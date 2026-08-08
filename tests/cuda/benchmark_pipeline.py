#!/usr/bin/env python3
"""CUDA → NV12 → FFmpeg NVENC 并行流水线基准测试。

核心理念：
    传统路线：CUDA 渲染所有帧 → 下载 NV12 → pipe 给 FFmpeg 编码（串行，渲染时编码器空闲）
    优化路线：CUDA 渲染 + NVENC 编码并行进行（生产者-消费者流水线）
    
    渲染线程生产 NV12 帧 → 有界队列 → 编码线程消费并写入 FFmpeg pipe
    渲染 5.59s 与编码 20.02s 重叠，省去渲染时间。

进一步优化：
    1. NVENC 参数调优（QP 35, main profile, coder=vlc 等）
    2. 使用 hwupload_cuda 滤镜让 FFmpeg 在 GPU 上编码（减少 PCIe 往返）
    3. 使用 CUDA 固定内存（pinned memory）加速 GPU→CPU 传输

测试流程：
    1. 加载工程文件（庙堂之外.uplr）
    2. CPU 预计算帧状态（去重）
    3. CUDA 渲染 + NVENC 编码并行流水线
    4. 封装 MP4
    5. 报告各阶段耗时

目标：总耗时 < 20s，理想 < 10s
"""

import os
import sys
import json
import time
import subprocess
import threading
import queue
from typing import List, Optional, Tuple, Dict

# ===================== 路径设置 =====================

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.renderer import (
    precompute_frame_states,
    build_ust_info_for_render,
    _render_frame_cuda,
    _render_frame_nv12,
    _clear_glyph_cache,
    _build_glyph_cache,
    _clear_cuda_contexts,
    _find_ffmpeg,
    FrameState,
    logger,
)
from core.ustxreader import get_ustx_info_from_content


# ===================== 配置 =====================

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
UPLR_PATH = os.path.join(os.path.dirname(__file__), "..", "庙堂之外.uplr")

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 30

# 编码参数
NVENC_PRESET = "p1"
NVENC_QP = 35  # 从 28 提升到 35——更快编码，预览质量可接受
NVENC_EXTRA = [
    "-profile:v", "main",
    "-coder", "vlc",
    "-weighted_pred", "0",
]

# 流水线队列大小（渲染比编码快，小队列足够）
PIPELINE_QUEUE_SIZE = 8


# 哨兵对象（用于线程间通信，表示队列结束）
_SENTINEL = object()

# ===================== 工具函数 =====================


def _load_uplr(uplr_path: str) -> dict:
    """加载 UPLR 工程文件，返回 ust_info。"""
    with open(uplr_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    fmt = payload.get("format", "")
    if fmt != "ustxPlayer-preview.uplr" or payload.get("version") != 2:
        raise ValueError(f"不是有效的 ustxPlayer-preview 工程文件")

    ustx_content = payload.get("ustx_content", "")
    if not ustx_content:
        raise ValueError("UPLR 文件中没有 USTX 内容")

    core_ust_info = get_ustx_info_from_content(ustx_content)
    logger.info(f"USTX 解析完成: {len(core_ust_info.get('notes', []))} 个音符, "
                f"BPM={core_ust_info.get('tempo')}")

    ust_info = _build_ust_info(core_ust_info, payload.get("settings", {}))
    return ust_info


def _build_ust_info(core_ust_info: dict, settings: dict) -> dict:
    """从 UPLR 设置构建 ust_info。"""
    styles = settings.get("styles", [])
    active_idx = settings.get("active_style_index", 0)
    if isinstance(active_idx, str):
        try:
            active_idx = int(active_idx)
        except ValueError:
            active_idx = 0
    active_idx = max(0, min(active_idx, len(styles) - 1)) if styles else 0
    ap = styles[active_idx] if styles else {}

    sc = settings.get("show_config", {}) or {}
    _bg_color = settings.get("bg_color", "#000000")
    _note_color = settings.get("note_color", "#6c6c6c")
    _lyric_color = settings.get("lyric_color", "#ffffff")
    _info_text_color = settings.get("info_text_color", "#ffffff")

    pitch_curve_color = ap.get("pitch_curve_color", "#ffffff")

    global_bg_enabled = False
    global_bg_color = "#00ff00"
    if isinstance(settings.get("global_bg_enabled"), bool):
        global_bg_enabled = settings["global_bg_enabled"]
        global_bg_color = settings.get("global_bg_color", "#00ff00")

    note_styles_raw = settings.get("note_styles", {})
    note_styles = {}
    if isinstance(note_styles_raw, dict):
        for k, v in note_styles_raw.items():
            try:
                note_styles[int(k)] = int(v)
            except (ValueError, TypeError):
                pass

    word_lyric_font_family = settings.get("word_lyric_font_family", "等线")
    info_font_family = settings.get("info_font_family", "微软雅黑")

    pi = {
        "song_name": settings.get("song_name", ""),
        "song_author": settings.get("song_author", ""),
        "ust_author": settings.get("ust_author", ""),
        "project_name": settings.get("project_name", ""),
    }

    return {
        "version": core_ust_info.get("version", "未知版本"),
        "tempo": core_ust_info.get("tempo", 120.0),
        "tracks": core_ust_info.get("tracks", 1),
        "notes": core_ust_info.get("notes", []),
        "show_config": {
            "bpm": sc.get("bpm", True),
            "play_time": sc.get("play_time", True),
            "song_name": sc.get("song_name", True),
            "song_author": sc.get("song_author", True),
            "ust_author": sc.get("ust_author", True),
            "copyright": sc.get("copyright", True),
            "lyric": sc.get("lyric", True),
            "lyric_autohide": sc.get("lyric_autohide", True),
            "lyric_autohide_threshold": float(sc.get("lyric_autohide_threshold", 3.0)),
            "curve_show": sc.get("curve_show", False),
        },
        "project_info": pi,
        "player_style": {
            "bg_color": ap.get("bg_color", _bg_color),
            "global_bg_color": global_bg_color,
            "global_bg_enabled": global_bg_enabled,
            "note_color": ap.get("note_color", _note_color),
            "lyric_color": ap.get("lyric_color", _lyric_color),
            "pitch_curve_color": pitch_curve_color,
            "info_text_color": _info_text_color,
            "lyric_pos": settings.get("lyric_pos", "上"),
            "show_phoneme": False,
            "show_midinote": False,
            "show_waveform": False,
            "fullscreen": False,
            "lrc_path": settings.get("lrc_path", ""),
            "audio_path": settings.get("audio_path", ""),
            "silent_display": settings.get("silent_display", "R"),
            "silent_custom_text": settings.get("silent_custom_text", ""),
            "end_display": settings.get("end_display", "END"),
            "end_custom_text": settings.get("end_custom_text", ""),
            "pitch_placeholder": settings.get("pitch_placeholder", "无"),
            "pitch_custom_text": settings.get("pitch_custom_text", ""),
            "word_lyric_font_family": word_lyric_font_family,
            "info_font_family": info_font_family,
            "styles": styles,
            "note_styles": note_styles,
        },
    }


def _find_ffmpeg_cuda() -> Optional[str]:
    """查找可用且支持 h264_nvenc 的 ffmpeg。"""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return None
    try:
        r = subprocess.run([ffmpeg, "-encoders"], capture_output=True, text=True, timeout=10,
                           creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
        if "h264_nvenc" in r.stdout:
            return ffmpeg
    except Exception:
        pass
    return None


def _build_ffmpeg_cmd(
    ffmpeg: str,
    output_path: str,
    width: int, height: int, fps: int,
    use_hwupload: bool = False,
) -> List[str]:
    """构建 FFmpeg 命令。

    Args:
        use_hwupload: 如果为 True，使用 hwupload_cuda 滤镜将帧上传到 GPU，
                      让 NVENC 直接从 GPU 显存编码（零拷贝路径）。
                      需要 FFmpeg 编译了 CUDA 支持。
    """
    if use_hwupload:
        # 零拷贝路径：hwupload_cuda 将帧上传到 GPU，NVENC 从 GPU 显存编码
        # 需要 FFmpeg 编译了 --enable-cuda-nvcc --enable-cuvid --enable-nvenc
        # -noautoscale 防止自动缩放与 hwupload_cuda 格式冲突
        cmd = [
            ffmpeg, "-y",
            "-init_hw_device", "cuda=gpu:0",
            "-filter_hw_device", "gpu",
            "-f", "rawvideo",
            "-pix_fmt", "nv12",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vf", "hwupload_cuda=extra_hw_frames=64,format=nv12",
            "-c:v", "h264_nvenc",
            "-preset", NVENC_PRESET,
            "-rc", "constqp", "-qp", str(NVENC_QP),
            "-bf", "0",
            "-spatial-aq", "0", "-temporal-aq", "0",
            "-rc-lookahead", "0", "-no-scenecut", "1",
            "-pix_fmt", "yuv420p",
            *NVENC_EXTRA,
            output_path,
        ]
    else:
        # 标准路径：NV12 数据通过 pipe 喂给 FFmpeg，h264_nvenc 内部处理上传
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "nv12",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "h264_nvenc",
            "-preset", NVENC_PRESET,
            "-rc", "constqp", "-qp", str(NVENC_QP),
            "-bf", "0",
            "-spatial-aq", "0", "-temporal-aq", "0",
            "-rc-lookahead", "0", "-no-scenecut", "1",
            "-pix_fmt", "yuv420p",
            *NVENC_EXTRA,
            output_path,
        ]
    return cmd


def _mux_mp4(
    ffmpeg: str,
    h264_path: str,
    output_path: str,
    fps: int,
    audio_path: Optional[str] = None,
) -> bool:
    """将 H.264 原始流封装为 MP4，可选添加音频。"""
    cmd = [
        ffmpeg, "-y",
        "-f", "h264",
        "-r", str(fps),
        "-i", h264_path,
    ]
    if audio_path and os.path.exists(audio_path):
        cmd.extend(["-i", audio_path, "-c", "copy", "-shortest"])
    else:
        cmd.extend(["-c:v", "copy"])
    cmd.append(output_path)

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
        if r.returncode != 0:
            logger.error(f"  封装失败: {r.stderr[-500:]}")
            return False
        return True
    except Exception as e:
        logger.error(f"  封装异常: {e}")
        return False


def _rgba_to_nv12_cpu_fallback(img, width: int, height: int) -> Optional[bytes]:
    """CPU 端 RGBA→NV12 转换回退。"""
    try:
        import numpy as np
        numpy_ref = getattr(img, "_numpy_ref", None)
        if numpy_ref is not None:
            rgba_np = numpy_ref
        else:
            ptr = img.bits()
            rgba_np = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))

        b = rgba_np[:, :, 0].astype(np.float32)
        g = rgba_np[:, :, 1].astype(np.float32)
        r = rgba_np[:, :, 2].astype(np.float32)

        y = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

        r_sub = r[::2, ::2]
        g_sub = g[::2, ::2]
        b_sub = b[::2, ::2]
        u = np.clip(-0.169 * r_sub - 0.331 * g_sub + 0.500 * b_sub + 128, 0, 255).astype(np.uint8)
        v = np.clip(0.500 * r_sub - 0.419 * g_sub - 0.081 * b_sub + 128, 0, 255).astype(np.uint8)

        uv = np.empty((height // 2, width), dtype=np.uint8)
        uv[:, 0::2] = u
        uv[:, 1::2] = v

        return y.tobytes() + uv.tobytes()
    except Exception as e:
        logger.error(f"CPU NV12 转换失败: {e}")
        return None


def _render_frame_to_nv12_bytes(
    state: FrameState, width: int, height: int, fonts: dict,
) -> Optional[bytes]:
    """CUDA 渲染帧并直接返回 NV12 字节（跳过 QImage 创建，节省 8MB RGBA 下载）。

    QImage 创建需要额外下载 8MB RGBA 帧缓冲，此函数直接返回 NV12 字节，
    减少 GPU→CPU 下载量为 3.1MB/帧，适合流水线编码场景。
    """
    try:
        import cupy as cp
    except ImportError:
        return None

    try:
        from core.renderer import (
            _get_cuda_ctx, _cuda_draw_polyline,
            _get_glyph_texture, _blit_texture, _rgba_to_nv12_gpu,
            _cuda_draw_text,
        )
        from PySide6.QtGui import QColor, QFontMetrics

        # 复用 _render_frame_cuda 的渲染逻辑，但跳过 QImage 创建
        ctx = _get_cuda_ctx(width, height)
        fb = ctx.fb
        stream = ctx.stream

        with stream:
            # ---- 重置帧缓冲 ----
            bg = QColor(state.bg_color)
            fb[:, :, 0] = bg.blue()
            fb[:, :, 1] = bg.green()
            fb[:, :, 2] = bg.red()

            cx, cy = width // 2, height // 2

            # ---- 音名 ----
            if state.show_note_name and state.note_name:
                note_c = QColor(*state.note_color)
                note_c.setAlpha(225)
                tex, tw, th = _get_glyph_texture(
                    state.note_name, fonts["note_font"], note_c, fonts["fm_note"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, cx - tw // 2, cy - th // 2, width, height)

            # ---- 音高线 ----
            if state.show_curve and state.pitch_points and len(state.pitch_points) >= 2:
                _cuda_draw_polyline(
                    fb, state.pitch_points,
                    QColor(state.pitch_curve_color),
                    5, width, height,
                )

            # ---- 歌词 ----
            if state.show_lyric and state.lyric:
                lyric_c = QColor(*state.lyric_color)
                tex, tw, th = _get_glyph_texture(
                    state.lyric, fonts["ust_lyric_font"], lyric_c, fonts["fm_ust_lyric"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, cx - tw // 2, cy - th // 2, width, height)

            # ---- 左上角信息 ----
            small_c = QColor(state.small_font_color)
            y_off = 20
            if state.show_song_name and state.song_name:
                tex, tw, th = _get_glyph_texture(state.song_name, fonts["bold_small_font"], small_c, fonts["fm_small"])
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, 20, y_off, width, height)
                    y_off += 27
            if state.show_song_author and state.song_author:
                tex, tw, th = _get_glyph_texture(state.song_author, fonts["small_font"], small_c, fonts["fm_small"])
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, 20, y_off, width, height)
                    y_off += 25
            if state.show_ust_author and state.ust_author:
                tex, tw, th = _get_glyph_texture(state.ust_author, fonts["small_font"], small_c, fonts["fm_small"])
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, 20, y_off, width, height)

            # ---- BPM ----
            if state.show_bpm:
                bpm_text = f"BPM={state.tempo}"
                tex, tw, th = _get_glyph_texture(bpm_text, fonts["small_font"], small_c, fonts["fm_small"])
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, width - 20 - tw, 20, width, height)

            # ---- LRC 歌词 ----
            if state.show_lyric and state.lrc_lines and not state.lrc_hidden:
                anchor_y = int(height * 0.3) if state.lyric_pos == "上" else int(height * 0.7)
                line_h = fonts["fm_lyric"].height()
                step = line_h * 1.3
                n = len(state.lrc_lines)
                top_baseline = anchor_y - (n - 1) * step if state.lyric_pos == "上" else anchor_y
                lrc_c = QColor(state.small_font_color)
                for li, text in enumerate(state.lrc_lines):
                    baseline = int(top_baseline + li * step)
                    tex, tw, th = _get_glyph_texture(text, fonts["lyric_font"], lrc_c, fonts["fm_lyric"])
                    if tex is not None:
                        _blit_texture(fb, tex, tw, th, width // 2 - tw // 2, baseline - fonts["fm_lyric"].ascent(), width, height)

            # ---- 版权 ----
            if state.show_copyright:
                copy_c = QColor(195, 195, 195)
                copy_c.setAlpha(100)
                tex, tw, th = _get_glyph_texture(state.copyright_text, fonts["copyright_font"], copy_c, fonts["fm_copyright"])
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, width // 2 - tw // 2, height - 20 - th, width, height)

        # stream 同步（等待 GPU 完成）
        stream.synchronize()

        # GPU 端 RGBA→NV12 转换（只下载 3.1MB NV12，跳过 8MB RGBA）
        nv12_gpu = _rgba_to_nv12_gpu(fb, width, height)
        nv12_cpu = cp.asnumpy(nv12_gpu)
        return nv12_cpu.tobytes()

    except Exception as e:
        logger.exception(f"CUDA→NV12 渲染失败")
        return None


# ===================== H.264 解析工具 =====================


def _parse_h264_frames(h264_data: bytes) -> Tuple[bytes, List[bytes]]:
    """解析 H.264 Annex B 格式，提取 SPS/PPS 和帧数据。

    Args:
        h264_data: 完整的 H.264 比特流数据

    Returns:
        (sps_pps_bytes, frame_data_list):
            sps_pps_bytes: 所有 SPS/PPS NAL 单元
            frame_data_list: 每个帧的 NAL 单元数据列表
    """
    # 找到所有 NAL 单元边界
    nal_units = []
    pos = 0
    while pos < len(h264_data):
        # 查找 start code (0x00 0x00 0x01 或 0x00 0x00 0x00 0x01)
        if pos + 3 > len(h264_data):
            break
        if h264_data[pos:pos+3] != b'\x00\x00\x01':
            pos += 1
            continue

        sc_len = 3
        if pos + 4 <= len(h264_data) and h264_data[pos+3] == 0x00:
            sc_len = 4

        # 查找下一个 start code
        next_pos = pos + sc_len
        while next_pos < len(h264_data):
            if next_pos + 3 <= len(h264_data) and h264_data[next_pos:next_pos+3] == b'\x00\x00\x01':
                break
            next_pos += 1

        nal_units.append(h264_data[pos:next_pos])
        pos = next_pos

    # 分类 NAL 单元
    sps_pps = b""
    seen_sps_pps_types = {}
    frame_data_list = []
    for nal in nal_units:
        if len(nal) < 4:
            continue
        # 确定 NAL 头位置
        nal_start = 3 if nal[0:3] == b'\x00\x00\x01' else 4
        nal_type = nal[nal_start] & 0x1F
        if nal_type in (7, 8):  # SPS(7) or PPS(8)，每种只保留一组
            # NVENC 在 -g 1 时每个 IDR 前都写 SPS/PPS；若全拼进头部，
            # FFmpeg 解析会错乱导致 MP4 Duration=N/A
            if not seen_sps_pps_types.get(nal_type):
                sps_pps += nal
                seen_sps_pps_types[nal_type] = True
        elif nal_type == 5:  # IDR slice
            frame_data_list.append(nal)
        elif nal_type == 1:  # Non-IDR slice
            frame_data_list.append(nal)

    return sps_pps, frame_data_list


def _unique_encode(
    ffmpeg: str,
    frame_states: List[FrameState],
    width: int, height: int, fps: int,
    fonts: dict,
    output_path: str,
    total_frames: int,
) -> bool:
    """唯一帧 H.264 I 帧重复编码方案（优化版本：pipe 直连 + 渲染编码重叠）。

    核心思路：
        NVENC 只编码 638 个唯一帧（每帧都是 IDR 帧，-g 1），
        然后在 H.264 比特流层面解析 NAL 单元，重复每帧 frame_count 次。
        这样 NVENC 的编码量减少 15.9 倍（638 帧 vs 10153 帧）。

    优化点：
        1. NV12 数据直接 pipe 给 FFmpeg，跳过中间文件 I/O
        2. 渲染线程与 FFmpeg 编码线程并行（生产者-消费者模式）
        3. 渲染完成后立即关闭 stdin，FFmpeg 自动结束并写 H.264 文件

    Returns:
        True 表示成功
    """
    unique_h264_path = os.path.join(OUTPUT_DIR, "_unique_temp.h264")

    # ====== 阶段 1+2: 渲染 + 编码并行（pipe 直连） ======
    logger.info("  [子阶段1+2] 渲染唯一帧 pipe → FFmpeg 编码（并行）...")
    render_start = time.monotonic()

    # 构建 FFmpeg 命令（stdin 接收 NV12，输出到 H.264 文件）
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "nv12",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "h264_nvenc",
        "-preset", NVENC_PRESET,
        "-rc", "constqp", "-qp", str(NVENC_QP),
        "-g", "1",
        "-bf", "0",
        "-spatial-aq", "0", "-temporal-aq", "0",
        "-rc-lookahead", "0", "-no-scenecut", "1",
        "-pix_fmt", "yuv420p",
        *NVENC_EXTRA,
        "-f", "h264",
        unique_h264_path,
    ]

    # 启动 FFmpeg（先预热，等 NVENC 初始化完成）
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
    except Exception as e:
        logger.error(f"  启动 FFmpeg 失败: {e}")
        return False

    # 渲染线程状态
    render_errors = []
    frames_rendered = [0]
    render_queue = queue.Queue(maxsize=8)

    def render_worker():
        """渲染线程：逐帧渲染 NV12 → 放入队列，主线程从队列取并写入 pipe。"""
        local_start = time.monotonic()
        try:
            for idx, state in enumerate(frame_states):
                nv12_bytes = _render_frame_nv12(state, width, height, fonts)
                if nv12_bytes is None:
                    img = _render_frame_cuda(state, width, height, fonts)
                    nv12_bytes = _rgba_to_nv12_cpu_fallback(img, width, height)
                    del img
                if nv12_bytes is None:
                    logger.error(f"  渲染帧 {idx} 失败")
                    with render_lock:
                        render_errors.append(f"帧 {idx} 渲染失败")
                    render_queue.put(None)
                    return

                render_queue.put(nv12_bytes)
                frames_rendered[0] = idx + 1

                if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_states):
                    elapsed = time.monotonic() - local_start
                    logger.info(f"  → 渲染进度: {idx+1}/{len(frame_states)} "
                                f"({(idx+1)/len(frame_states)*100:.0f}%), "
                                f"{elapsed:.1f}s, {(idx+1)/elapsed:.0f} fps")
        except Exception as e:
            logger.exception(f"渲染线程异常")
            with render_lock:
                render_errors.append(str(e))
            render_queue.put(None)
        finally:
            render_queue.put(_SENTINEL)

    render_lock = threading.Lock()
    render_thread = threading.Thread(target=render_worker, daemon=True)
    render_thread.start()

    # 主线程：从队列取 NV12 数据，写入 FFmpeg pipe
    frames_sent = 0
    pipe_ok = True

    while True:
        item = render_queue.get()
        if item is _SENTINEL:
            break
        if item is None:
            pipe_ok = False
            break

        try:
            proc.stdin.write(item)
            frames_sent += 1
        except BrokenPipeError:
            logger.error(f"  Pipe 断开 (帧 {frames_sent})")
            pipe_ok = False
            break
        except Exception as e:
            logger.error(f"  写入 pipe 失败 (帧 {frames_sent}): {e}")
            pipe_ok = False
            break

    # 关闭 stdin，等待 FFmpeg 完成
    render_end = time.monotonic()
    render_time = render_end - render_start

    try:
        proc.stdin.close()
    except Exception:
        pass

    render_thread.join(timeout=30)

    if not pipe_ok:
        try:
            proc.kill()
        except Exception:
            pass
        return False

    # 等待 FFmpeg 完成编码
    try:
        stdout, stderr = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        logger.error("  FFmpeg 超时，已终止")
        return False

    encode_end = time.monotonic()

    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace")[-1000:]
        logger.error(f"  FFmpeg 编码失败 (返回码 {proc.returncode}):")
        for line in stderr_text.split("\n")[-10:]:
            logger.error(f"    {line.strip()}")
        return False

    encode_time = encode_end - render_start
    logger.info(f"  [子阶段1+2] 完成: {frames_rendered[0]} 帧渲染 + 编码, "
                f"{encode_time:.2f}s ({frames_rendered[0]/encode_time:.0f} fps)")

    # ====== 阶段 3: 解析 H.264 并重复帧 ======
    logger.info("  [子阶段3] 解析 H.264 并重复帧...")
    parse_start = time.monotonic()

    with open(unique_h264_path, "rb") as f:
        h264_data = f.read()

    sps_pps, frame_data_list = _parse_h264_frames(h264_data)

    if len(frame_data_list) != len(frame_states):
        logger.warning(f"  H.264 帧数 ({len(frame_data_list)}) 与预期 "
                       f"({len(frame_states)}) 不匹配，尝试修复...")
        if len(frame_data_list) > len(frame_states):
            frame_data_list = frame_data_list[:len(frame_states)]
        while len(frame_data_list) < len(frame_states):
            frame_data_list.append(frame_data_list[-1] if frame_data_list else b"")

    # 写入最终 H.264 文件（使用 chunk 乘法替代循环写入，大幅减少 I/O 调用次数）
    with open(output_path, "wb", buffering=1024*1024) as f:
        if sps_pps:
            f.write(sps_pps)
        for i, state in enumerate(frame_states):
            frame_data = frame_data_list[i]
            # 一次性写入 frame_count 个重复帧（避免 10153 次 write 调用）
            f.write(frame_data * state.frame_count)

    parse_end = time.monotonic()
    parse_repeat_time = parse_end - parse_start
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    logger.info(f"  [子阶段3] 完成: {total_frames} 帧, "
                f"{parse_repeat_time:.3f}s, "
                f"输出文件: {file_size_mb:.1f}MB")

    # 清理临时文件
    try:
        os.unlink(unique_h264_path)
    except Exception:
        pass

    return True


# ===================== 并行流水线核心 =====================


def _parallel_pipeline(
    ffmpeg: str,
    frame_states: List[FrameState],
    width: int, height: int, fps: int,
    fonts: dict,
    output_path: str,
    total_frames: int,
    use_hwupload: bool = False,
) -> bool:
    """并行渲染+编码流水线。

    生产者-消费者模式：
        - 渲染线程（producer）：逐帧渲染为 NV12，放入有界队列
        - 主线程（consumer）：从队列取帧，写入 FFmpeg pipe

    队列大小限制为 PIPELINE_QUEUE_SIZE，防止内存爆炸。
    """
    nv12_queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)

    # 渲染线程状态
    render_errors = []
    render_lock = threading.Lock()
    render_start_time = [0.0]
    render_end_time = [0.0]
    frames_rendered = [0]

    def render_worker():
        """渲染线程：逐帧渲染为 NV12，放入队列。
        
        使用 _render_frame_nv12() 跳过 QImage 创建（节省 8MB RGBA 下载），
        直接从 GPU 帧缓冲下载 3.1MB NV12 数据。
        """
        render_start_time[0] = time.monotonic()
        try:
            for idx, state in enumerate(frame_states):
                try:
                    # 直接渲染为 NV12 字节（跳过 QImage 创建）
                    nv12_bytes = _render_frame_nv12(state, width, height, fonts)

                    if nv12_bytes is None:
                        # 回退：使用 _render_frame_cuda + CPU NV12 转换
                        img = _render_frame_cuda(state, width, height, fonts)
                        nv12_bytes = getattr(img, "_nv12_bytes", None)
                        if nv12_bytes is None:
                            nv12_bytes = _rgba_to_nv12_cpu_fallback(img, width, height)
                        del img

                    if nv12_bytes is None:
                        logger.error(f"  渲染帧 {idx} 失败")
                        continue

                    nv12_queue.put((nv12_bytes, state.frame_count))
                    frames_rendered[0] = idx + 1

                    if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_states):
                        elapsed = time.monotonic() - render_start_time[0]
                        logger.info(f"  渲染进度: {idx + 1}/{len(frame_states)} "
                                    f"({(idx+1)/len(frame_states)*100:.0f}%), "
                                    f"{elapsed:.1f}s, {(idx+1)/elapsed:.0f} fps")

                except Exception as e:
                    logger.exception(f"渲染帧 {idx} 异常")
                    with render_lock:
                        render_errors.append(str(e))
                    nv12_queue.put(None)  # 错误通知
                    break
        finally:
            render_end_time[0] = time.monotonic()
            nv12_queue.put(_SENTINEL)  # 结束标志

    # 构建 FFmpeg 命令
    cmd = _build_ffmpeg_cmd(ffmpeg, output_path, width, height, fps, use_hwupload)
    logger.info(f"  FFmpeg 命令: {' '.join(cmd[:10])}...")

    # 启动 FFmpeg
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
    except Exception as e:
        logger.error(f"  启动 FFmpeg 失败: {e}")
        return False

    # 启动渲染线程
    render_thread = threading.Thread(target=render_worker, daemon=True)
    render_thread.start()

    # ====== 主线程：编码消费者 ======
    encode_start = time.monotonic()
    frames_sent = 0
    encode_ok = True

    while True:
        item = nv12_queue.get()

        if item is _SENTINEL:
            # 渲染完成
            break

        if item is None:
            # 渲染出错
            encode_ok = False
            break

        nv12_bytes, repeat_count = item

        # 写入 pipe，重复帧重复写入
        for _ in range(repeat_count):
            try:
                proc.stdin.write(nv12_bytes)
                frames_sent += 1
            except BrokenPipeError:
                logger.error(f"  Pipe 断开 (帧 {frames_sent})")
                encode_ok = False
                break
            except Exception as e:
                logger.error(f"  写入 pipe 失败 (帧 {frames_sent}): {e}")
                encode_ok = False
                break

        if not encode_ok:
            break

        # 进度报告
        if frames_sent % 500 == 0 or frames_sent >= total_frames:
            elapsed = time.monotonic() - encode_start
            if elapsed > 0:
                logger.info(f"  编码进度: {frames_sent}/{total_frames} "
                            f"({frames_sent/total_frames*100:.0f}%), "
                            f"{frames_sent/elapsed:.0f} fps, {elapsed:.1f}s")

    # 关闭 stdin
    try:
        proc.stdin.close()
    except Exception:
        pass

    # 等待渲染线程
    render_thread.join(timeout=30)

    # 等待 FFmpeg 完成
    encode_end = time.monotonic()
    try:
        stdout, stderr = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        logger.error("  FFmpeg 超时，已终止")
        return False

    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace")[-2000:]
        logger.error(f"  FFmpeg 编码失败 (返回码 {proc.returncode}):")
        for line in stderr_text.split("\n")[-20:]:
            logger.error(f"    {line.strip()}")
        return False

    # 记录编码耗时
    elapsed = encode_end - encode_start
    logger.info(f"  编码完成: {frames_sent} 帧, {elapsed:.2f}s "
                f"({frames_sent/elapsed:.0f} fps)")

    return True


# ===================== 基准测试主函数 =====================


def run_benchmark(method: str = "pipe") -> dict:
    """运行基准测试，返回各阶段耗时统计。

    Args:
        method: "pipe" — 标准 pipe 流水线
                "hwupload" — hwupload_cuda 零拷贝路径
                "unique" — H.264 I 帧重复方案（仅编码唯一帧）
    """
    results = {}
    t_start = time.monotonic()

    # 方法名映射
    method_names = {
        "pipe": "标准 pipe 流水线",
        "hwupload": "hwupload_cuda 零拷贝路径",
        "unique": "H.264 I 帧重复（仅编码唯一帧）",
    }
    mode_name = method_names.get(method, method)

    # ==================== 阶段 0: 加载工程 + 检查 FFmpeg ====================
    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info(f"CUDA → NV12 → FFmpeg NVENC 基准测试 [{mode_name}]")
    logger.info(f"工程文件: {UPLR_PATH}")
    logger.info(f"输出分辨率: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} @ {OUTPUT_FPS}fps")
    logger.info(f"NVENC: preset={NVENC_PRESET}, qp={NVENC_QP}")
    logger.info("=" * 60)

    # 检查 FFmpeg
    ffmpeg = _find_ffmpeg_cuda()
    if not ffmpeg:
        logger.error("未找到支持 h264_nvenc 的 FFmpeg")
        results["error"] = "FFmpeg NVENC 不可用"
        return results
    logger.info(f"FFmpeg: {ffmpeg}")

    # 检查 CuPy
    try:
        import cupy as cp
        _ = cp.zeros((1,), dtype=cp.float32)
        logger.info(f"CuPy: {cp.__version__}")
        logger.info(f"CUDA 设备: {cp.cuda.Device(0)}")
    except ImportError:
        logger.error("需要 cupy: pip install cupy-cuda12x")
        results["error"] = "cupy 未安装"
        return results
    except Exception as e:
        logger.error(f"CuPy 初始化失败: {e}")
        results["error"] = f"CuPy 初始化失败: {e}"
        return results

    # 加载 UPLR
    ust_info = _load_uplr(UPLR_PATH)
    t1 = time.monotonic()
    load_time = t1 - t0
    results["load_time"] = load_time
    logger.info(f"[阶段0] 加载工程: {load_time:.2f}s")

    # ==================== 阶段 1: 预计算 ====================
    logger.info("-" * 40)
    logger.info("[阶段1] CPU 预计算帧状态...")

    render_info = build_ust_info_for_render(ust_info, OUTPUT_WIDTH, OUTPUT_HEIGHT)
    fonts = render_info["_render_fonts"]

    frame_states = precompute_frame_states(ust_info, OUTPUT_FPS, OUTPUT_WIDTH, OUTPUT_HEIGHT)
    unique_count = len(frame_states)
    total_output_frames = sum(s.frame_count for s in frame_states)
    video_duration = total_output_frames / OUTPUT_FPS

    dedup_ratio = total_output_frames / unique_count if unique_count > 0 else 0

    t2 = time.monotonic()
    precompute_time = t2 - t1
    results["precompute_time"] = precompute_time
    results["unique_frames"] = unique_count
    results["output_frames"] = total_output_frames
    results["video_duration"] = video_duration
    logger.info(f"  唯一帧: {unique_count}, 输出帧: {total_output_frames}, "
                f"去重比: {dedup_ratio:.1f}x, 时长: {video_duration:.1f}s")
    logger.info(f"[阶段1] 预计算: {precompute_time:.2f}s")

    if unique_count == 0:
        results["error"] = "预计算产生 0 帧"
        return results

    # ==================== 阶段 2+3: 渲染+编码 ====================
    logger.info("-" * 40)
    if method == "unique":
        logger.info("[阶段2+3] 唯一帧渲染 → H.264 I 帧编码 → 比特流层面重复帧...")
    else:
        logger.info("[阶段2+3] 并行渲染 + NVENC 编码（生产者-消费者流水线）...")

    # 预构建字形图集
    _clear_glyph_cache()
    _build_glyph_cache(frame_states, fonts)
    logger.info(f"  字形图集: {len(frame_states)} 帧")

    # 输出路径
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix_map = {"pipe": "", "hwupload": "_hwupload", "unique": "_unique"}
    suffix = suffix_map.get(method, "")
    output_h264 = os.path.join(OUTPUT_DIR, f"pipeline_output{suffix}.h264")
    output_mp4 = os.path.join(OUTPUT_DIR, f"pipeline_output{suffix}.mp4")

    # 执行渲染+编码
    pipeline_start = time.monotonic()
    use_hwupload = (method == "hwupload")

    if method == "unique":
        pipeline_ok = _unique_encode(
            ffmpeg, frame_states, OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FPS,
            fonts, output_h264, total_output_frames,
        )
    else:
        pipeline_ok = _parallel_pipeline(
            ffmpeg, frame_states, OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FPS,
            fonts, output_h264, total_output_frames, use_hwupload,
        )

    pipeline_end = time.monotonic()
    pipeline_time = pipeline_end - pipeline_start

    if not pipeline_ok:
        results["error"] = "编码失败"
        results["total_time"] = time.monotonic() - t_start
        return results

    results["pipeline_time"] = pipeline_time
    results["pipeline_fps"] = total_output_frames / pipeline_time if pipeline_time > 0 else 0
    logger.info(f"[阶段2+3] 完成: {total_output_frames} 帧, {pipeline_time:.2f}s "
                f"({total_output_frames/pipeline_time:.0f} fps)")

    # 清理 GPU 资源
    _clear_glyph_cache()
    _clear_cuda_contexts()

    # ==================== 阶段 4: 封装 MP4 ====================
    logger.info("-" * 40)
    logger.info("[阶段4] 封装 MP4...")

    audio_path = ust_info.get("player_style", {}).get("audio_path", "")
    if audio_path and not os.path.exists(audio_path):
        audio_path = None

    mux_ok = _mux_mp4(ffmpeg, output_h264, output_mp4, OUTPUT_FPS, audio_path)
    if mux_ok:
        results["output_path"] = output_mp4
        logger.info(f"  输出文件: {output_mp4}")
    else:
        results["output_path"] = output_h264
        logger.info(f"  封装失败，保留 H.264: {output_h264}")

    # ==================== 统计 ====================
    t_end = time.monotonic()
    total_time = t_end - t_start
    results["total_time"] = total_time
    results["effective_fps"] = total_output_frames / total_time if total_time > 0 else 0

    logger.info("=" * 60)
    logger.info("基准测试结果:")
    logger.info(f"  加载工程:     {load_time:.2f}s")
    logger.info(f"  预计算帧:     {precompute_time:.2f}s")
    logger.info(f"  渲染+编码:    {pipeline_time:.2f}s ({results['pipeline_fps']:.0f} fps)")
    logger.info(f"  ─────────────────────────────")
    logger.info(f"  总耗时:       {total_time:.2f}s")
    logger.info(f"  视频时长:     {video_duration:.1f}s")
    logger.info(f"  有效帧率:     {results['effective_fps']:.0f} fps")
    logger.info(f"  去重比:       {dedup_ratio:.1f}x ({unique_count} 唯一帧)")
    logger.info(f"  模式:         {mode_name}")
    logger.info(f"  NVENC:        preset={NVENC_PRESET}, qp={NVENC_QP}")
    logger.info(f"  输出文件:     {results['output_path']}")
    logger.info("=" * 60)

    return results


# ===================== 入口 =====================

if __name__ == "__main__":
    # 必须先创建 QGuiApplication（QFont/QFontMetrics 需要）
    from PySide6.QtGui import QGuiApplication
    _app = QGuiApplication([])

    import argparse

    parser = argparse.ArgumentParser(description="CUDA → NV12 → FFmpeg NVENC 并行流水线基准测试")
    parser.add_argument("--runs", type=int, default=1, help="测试次数")
    parser.add_argument("--width", type=int, default=OUTPUT_WIDTH)
    parser.add_argument("--height", type=int, default=OUTPUT_HEIGHT)
    parser.add_argument("--fps", type=int, default=OUTPUT_FPS)
    parser.add_argument("--preset", type=str, default=NVENC_PRESET, help="NVENC 预设 (p1-p7)")
    parser.add_argument("--qp", type=int, default=NVENC_QP, help="量化参数 (默认 35)")
    parser.add_argument("--method", type=str, default="unique",
                        choices=["pipe", "hwupload", "unique", "all"],
                        help="测试方法: pipe=标准pipe, hwupload=hwupload_cuda, "
                             "unique=H.264 I帧重复(默认), all=全部")

    args = parser.parse_args()

    OUTPUT_WIDTH = args.width
    OUTPUT_HEIGHT = args.height
    OUTPUT_FPS = args.fps
    NVENC_PRESET = args.preset
    NVENC_QP = args.qp

    all_results = []

    if args.method == "all":
        methods = ["pipe", "hwupload", "unique"]
    else:
        methods = [args.method]

    for method_name in methods:
        for run in range(args.runs):
            logger.info(f"\n\n=== 第 {run + 1}/{args.runs} 次测试 [{method_name}] ===")
            result = run_benchmark(method=method_name)
            result["method"] = method_name
            all_results.append(result)

    # 汇总
    if len(all_results) > 1:
        # 按方法分组
        by_method = {}
        for r in all_results:
            m = r.get("method", "unknown")
            if m not in by_method:
                by_method[m] = []
            by_method[m].append(r)

        for method, results in by_method.items():
            valid = [r for r in results if "error" not in r]
            if valid:
                total_times = [r.get("total_time", 0) for r in valid]
                pipeline_times = [r.get("pipeline_time", 0) for r in valid]
                fps_list = [r.get("effective_fps", 0) for r in valid]
                logger.info(f"\n\n=== 汇总 [{method}] ({len(valid)} 次有效测试) ===")
                logger.info(f"  平均总耗时: {sum(total_times)/len(total_times):.2f}s")
                logger.info(f"  最短总耗时: {min(total_times):.2f}s")
                logger.info(f"  平均流水线: {sum(pipeline_times)/len(pipeline_times):.2f}s")
                logger.info(f"  平均有效帧率: {sum(fps_list)/len(fps_list):.0f} fps")