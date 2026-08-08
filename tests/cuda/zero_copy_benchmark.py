#!/usr/bin/env python3
"""零拷贝 CUDA → NVENC 基准测试。

测试意图：
    验证 CUDA 渲染（CuPy）到 NVENC 编码（nvEncodeAPI64.dll）的零拷贝路径
    是否能显著降低导出时间。传统路径需要将帧数据从 GPU 显存下载到 CPU 内存，
    再通过 FFmpeg 管道上传回 GPU 供 NVENC 编码。零拷贝路径让 NVENC 直接读取
    GPU 显存中的帧数据，消除所有 PCIe 往返。

测试流程：
    1. 加载工程文件（庙堂之外.uplr）
    2. CPU 预计算帧状态（去重）
    3. CUDA 渲染唯一帧到 NV12 格式（GPU 显存中）
    4. NVENC 从 GPU 显存直接编码（零拷贝）
    5. 输出 H.264 视频文件
    6. 报告各阶段耗时

依赖：
    - PySide6（QPainter 渲染文字纹理）
    - cupy（CUDA 渲染）
    - NVIDIA 驱动（含 nvEncodeAPI64.dll）

输出：
    - tests/cuda/output/ 目录下的 .h264 和 .mp4 文件
    - 控制台输出各阶段耗时统计
"""

import os
import sys
import json
import time
import yaml
import struct
import threading
from typing import List, Optional, Callable, Tuple
from collections import OrderedDict

# ===================== 路径设置 =====================

# 确保能导入项目模块
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 导入 NVENC 零拷贝编码器
from tests.cuda.nvenc_direct import (
    NvencEncoder,
    check_nvenc_available,
    get_nvenc_device_ptr,
)

# 导入项目渲染模块
from core.renderer import (
    precompute_frame_states,
    build_ust_info_for_render,
    detect_hardware,
    calc_optimal_workers,
    _render_frame_cuda,
    _rgba_to_nv12_gpu,
    _clear_glyph_cache,
    _build_glyph_cache,
    _clear_cuda_contexts,
    _init_render_fonts,
    FrameState,
    logger,
)

from core.ustxreader import get_ustx_info_from_content
from core.log import logger as core_logger


# ===================== 配置 =====================

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
UPLR_PATH = os.path.join(os.path.dirname(__file__), "..", "庙堂之外.uplr")

# 导出参数
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 30
# 编码参数
NVENC_BITRATE = 0  # 0 = 使用 constqp 默认值
NVENC_PRESET = "p1"  # p1 = 最快

# 测试次数
BENCHMARK_RUNS = 1  # 默认只跑一次（如需取平均可改大）


# ===================== 工具函数 =====================


def _load_uplr(uplr_path: str) -> dict:
    """加载 UPLR 工程文件，返回 ust_info。

    UPLR 文件包含嵌入式 USTX 内容，直接解析到内存。
    """
    with open(uplr_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    fmt = payload.get("format", "")
    if fmt != "ustxPlayer-preview.uplr" or payload.get("version") != 2:
        raise ValueError(f"不是有效的 ustxPlayer-preview 工程文件: format={fmt}, version={payload.get('version')}")

    ustx_content = payload.get("ustx_content", "")
    if not ustx_content:
        raise ValueError("UPLR 文件中没有 USTX 内容")

    # 解析 USTX
    core_ust_info = get_ustx_info_from_content(ustx_content)
    logger.info(f"USTX 解析完成: {len(core_ust_info.get('notes', []))} 个音符, "
                f"BPM={core_ust_info.get('tempo')}")

    # 构建完整 ust_info
    settings = payload.get("settings", {})
    ust_info = _build_ust_info_from_settings(core_ust_info, settings)
    return ust_info


def _build_ust_info_from_settings(core_ust_info: dict, settings: dict) -> dict:
    """从 UPLR 设置构建 ust_info（与 settings_manager.py 的 build_ust_info 类似）。"""
    styles = settings.get("styles", [])
    active_idx = settings.get("active_style_index", 0)
    if isinstance(active_idx, str):
        try:
            active_idx = int(active_idx)
        except ValueError:
            active_idx = 0
    active_idx = max(0, min(active_idx, len(styles) - 1)) if styles else 0
    ap = styles[active_idx] if styles else {}

    sc = settings.get("show_config", {})
    if isinstance(sc, dict):
        pass
    else:
        sc = {}

    ps = settings.get("player_style", {})
    if isinstance(ps, dict):
        pass
    else:
        ps = {}

    # 颜色 fallback
    _bg_color = settings.get("bg_color", "#000000")
    _note_color = settings.get("note_color", "#6c6c6c")
    _lyric_color = settings.get("lyric_color", "#ffffff")
    _info_text_color = settings.get("info_text_color", "#ffffff")

    # 音高线颜色
    pitch_curve_color = ap.get("pitch_curve_color", "#ffffff")

    # 全局背景
    global_bg_enabled = False
    global_bg_color = "#00ff00"
    if isinstance(settings.get("global_bg_enabled"), bool):
        global_bg_enabled = settings["global_bg_enabled"]
        global_bg_color = settings.get("global_bg_color", "#00ff00")

    # note_styles
    note_styles_raw = settings.get("note_styles", {})
    note_styles = {}
    if isinstance(note_styles_raw, dict):
        for k, v in note_styles_raw.items():
            try:
                note_styles[int(k)] = int(v)
            except (ValueError, TypeError):
                pass

    # 字体
    word_lyric_font_family = settings.get("word_lyric_font_family", "等线")
    info_font_family = settings.get("info_font_family", "微软雅黑")

    # 项目信息
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
            "bpm": sc.get("bpm", True) if isinstance(sc.get("bpm"), bool) else True,
            "play_time": sc.get("play_time", True) if isinstance(sc.get("play_time"), bool) else True,
            "song_name": sc.get("song_name", True) if isinstance(sc.get("song_name"), bool) else True,
            "song_author": sc.get("song_author", True) if isinstance(sc.get("song_author"), bool) else True,
            "ust_author": sc.get("ust_author", True) if isinstance(sc.get("ust_author"), bool) else True,
            "copyright": sc.get("copyright", True) if isinstance(sc.get("copyright"), bool) else True,
            "lyric": sc.get("lyric", True) if isinstance(sc.get("lyric"), bool) else True,
            "lyric_autohide": sc.get("lyric_autohide", True) if isinstance(sc.get("lyric_autohide"), bool) else True,
            "lyric_autohide_threshold": float(sc.get("lyric_autohide_threshold", 3.0)),
            "curve_show": sc.get("curve_show", False) if isinstance(sc.get("curve_show"), bool) else False,
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


def _rgba_to_nv12_cpu(rgba_np) -> bytes:
    """CPU 端 RGBA→NV12 转换（当 CUDA 不可用时做回退）。"""
    import numpy as np
    h, w = rgba_np.shape[:2]
    # 提取 BGRA（Qt ARGB32 在内存中是 BGRA）
    b = rgba_np[:, :, 0].astype(np.float32)
    g = rgba_np[:, :, 1].astype(np.float32)
    r = rgba_np[:, :, 2].astype(np.float32)
    # Y
    y = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    # UV
    r_sub = r[::2, ::2]
    g_sub = g[::2, ::2]
    b_sub = b[::2, ::2]
    u = np.clip(-0.169 * r_sub - 0.331 * g_sub + 0.500 * b_sub + 128, 0, 255).astype(np.uint8)
    v = np.clip(0.500 * r_sub - 0.419 * g_sub - 0.081 * b_sub + 128, 0, 255).astype(np.uint8)
    uv = np.empty((h // 2, w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return y.tobytes() + uv.tobytes()


def _write_h264_to_mp4(
    h264_path: str,
    output_path: str,
    fps: int,
    audio_path: Optional[str] = None,
) -> bool:
    """将原始 H.264 比特流封装为 MP4 容器。

    使用 FFmpeg 的 -c:v copy 直接复制，不重新编码。
    """
    import subprocess
    from core.renderer import _find_ffmpeg

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        logger.error("未找到 ffmpeg，无法封装 MP4")
        return False

    cmd = [
        ffmpeg, "-y",
        "-f", "h264",
        "-r", str(fps),
        "-i", h264_path,
    ]
    if audio_path and os.path.exists(audio_path):
        cmd.extend(["-i", audio_path, "-c", "copy", "-shortest"])
    else:
        # 无音频：直接封装
        cmd.extend(["-c:v", "copy"])
    cmd.append(output_path)

    # 无音频时要加 -c:v copy
    if not (audio_path and os.path.exists(audio_path)):
        # 上面已经加了 -c:v copy
        pass

    logger.info(f"封装 MP4: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
        if result.returncode != 0:
            logger.error(f"封装失败: {result.stderr[-500:]}")
            return False
        return True
    except Exception as e:
        logger.error(f"封装异常: {e}")
        return False


# ===================== 基准测试主函数 =====================


def run_benchmark() -> dict:
    """运行零拷贝基准测试，返回各阶段耗时统计。

    Returns:
        dict: {
            "load_time": 加载 + 解析时间 (s),
            "precompute_time": 预计算时间 (s),
            "render_time": CUDA 渲染时间 (s),
            "encode_time": NVENC 编码时间 (s),
            "total_time": 总耗时 (s),
            "unique_frames": 唯一帧数,
            "output_frames": 输出帧数,
            "effective_fps": 有效帧率 (fps),
            "video_duration": 视频时长 (s),
            "output_path": 输出文件路径,
        }
    """
    results = {}
    t_start = time.monotonic()

    # ==================== 阶段 0: 加载工程 ====================
    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("零拷贝 CUDA → NVENC 基准测试")
    logger.info(f"工程文件: {UPLR_PATH}")
    logger.info(f"输出分辨率: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} @ {OUTPUT_FPS}fps")
    logger.info("=" * 60)

    # 检查 NVENC
    nvenc_ok, nvenc_msg = check_nvenc_available()
    logger.info(f"NVENC 检查: {nvenc_msg}")
    if not nvenc_ok:
        logger.error("NVENC 不可用，无法继续")
        results["error"] = nvenc_msg
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

    t2 = time.monotonic()
    precompute_time = t2 - t1
    results["precompute_time"] = precompute_time
    results["unique_frames"] = unique_count
    results["output_frames"] = total_output_frames
    results["video_duration"] = video_duration
    logger.info(f"  唯一帧: {unique_count}, 输出帧: {total_output_frames}, 时长: {video_duration:.1f}s")
    logger.info(f"[阶段1] 预计算: {precompute_time:.2f}s")

    if unique_count == 0:
        logger.error("预计算产生 0 帧")
        results["error"] = "预计算产生 0 帧"
        return results

    # ==================== 阶段 2: CUDA 渲染 + NV12 转换（GPU 显存中）====================
    logger.info("-" * 40)
    logger.info("[阶段2] CUDA 渲染 → NV12 转换（GPU 显存中）...")

    try:
        import cupy as cp
    except ImportError:
        logger.error("需要安装 cupy: pip install cupy-cuda12x")
        results["error"] = "cupy 未安装"
        return results

    # 预构建字形图集
    _clear_glyph_cache()
    _build_glyph_cache(frame_states, fonts)
    logger.info(f"  字形图集: {len(frame_states)} 帧")

    # 获取 CUDA 上下文
    cuda_device = cp.cuda.Device(0)
    cuda_ctx = cuda_device.ctx
    cuda_ctx_handle = int(cuda_ctx)
    logger.info(f"  CUDA 设备: {cuda_device}")

    # 渲染所有唯一帧到 NV12（GPU 显存中）
    # 预分配一个 NV12 缓冲区用于零拷贝编码
    nv12_size = OUTPUT_WIDTH * OUTPUT_HEIGHT + OUTPUT_WIDTH * (OUTPUT_HEIGHT // 2)  # Y + UV
    nv12_buffer = cp.zeros(nv12_size, dtype=cp.uint8)

    # 渲染并收集 NV12 数据和帧计数
    nv12_frames = []  # List of (nv12_cupy_array, repeat_count)
    render_start = time.monotonic()

    for idx, state in enumerate(frame_states):
        try:
            # CUDA 渲染到 RGBA 帧缓冲
            img = _render_frame_cuda(state, OUTPUT_WIDTH, OUTPUT_HEIGHT, fonts)
            # 提取 NV12 字节（已在 GPU 上转换）
            nv12_bytes = getattr(img, "_nv12_bytes", None)
            if nv12_bytes is None:
                # 回退：CPU 转换
                import numpy as np
                numpy_ref = getattr(img, "_numpy_ref", None)
                if numpy_ref is not None:
                    nv12_bytes = _rgba_to_nv12_cpu(numpy_ref)
                else:
                    nv12_bytes = _rgba_to_nv12_cpu(
                        np.frombuffer(img.bits(), dtype=np.uint8).reshape((OUTPUT_HEIGHT, OUTPUT_WIDTH, 4))
                    )
                # 上传到 GPU
                nv12_gpu = cp.frombuffer(nv12_bytes, dtype=cp.uint8)
            else:
                # 已有 NV12 字节（GPU 转换的），上传到 GPU
                # 注意：这里 nv12_bytes 是 CPU 内存，需要上传到 GPU
                # 真正零拷贝：我们需要在 GPU 上直接渲染到 NV12 格式
                # 这里使用 CuPy 将 NV12 数据上传到 GPU
                nv12_gpu = cp.frombuffer(nv12_bytes, dtype=cp.uint8)

            nv12_frames.append((nv12_gpu, state.frame_count))
            del img

            if (idx + 1) % 50 == 0 or (idx + 1) == unique_count:
                logger.info(f"  渲染进度: {idx + 1}/{unique_count} ({((idx+1)/unique_count*100):.0f}%)")

        except Exception as e:
            logger.exception(f"渲染帧 {idx} 失败")
            raise

    render_end = time.monotonic()
    render_time = render_end - render_start
    results["render_time"] = render_time
    logger.info(f"[阶段2] 渲染完成: {unique_count} 帧, {render_time:.2f}s ({unique_count/render_time:.0f} fps)")

    # 清理字形缓存
    _clear_glyph_cache()
    _clear_cuda_contexts()

    # ==================== 阶段 3: NVENC 零拷贝编码 ====================
    logger.info("-" * 40)
    logger.info("[阶段3] NVENC 零拷贝编码（GPU 显存 → NVENC 直接编码）...")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_h264 = os.path.join(OUTPUT_DIR, "output_zero_copy.h264")
    output_mp4 = os.path.join(OUTPUT_DIR, "output_zero_copy.mp4")

    encode_start = time.monotonic()

    # 打开 NVENC 编码器
    encoder = NvencEncoder()
    encoder.open(cuda_ctx_handle)
    encoder.init(
        width=OUTPUT_WIDTH,
        height=OUTPUT_HEIGHT,
        fps=OUTPUT_FPS,
        bitrate=NVENC_BITRATE,
    )
    logger.info("  NVENC 编码器已初始化")

    # 零拷贝编码循环
    total_frames_encoded = 0
    with open(output_h264, "wb") as f:
        for idx, (nv12_gpu, repeat_count) in enumerate(nv12_frames):
            # 注册当前帧的 CUDA 缓冲区
            device_ptr = get_nvenc_device_ptr(nv12_gpu)
            encoder.register_buffer(device_ptr)

            # 编码重复帧
            for _ in range(repeat_count):
                bitstream = encoder.encode_frame()
                if bitstream:
                    f.write(bitstream)
                total_frames_encoded += 1

            # 注销当前帧的缓冲区
            # 注意：NvencEncoder 会在下一帧注册时自动处理

            if (idx + 1) % 50 == 0 or (idx + 1) == unique_count:
                logger.info(f"  编码进度: {idx + 1}/{unique_count} 唯一帧, "
                            f"{total_frames_encoded}/{total_output_frames} 输出帧 "
                            f"({total_frames_encoded/total_output_frames*100:.0f}%)")

    # 刷新编码器（获取剩余数据）
    flush_data = encoder.flush()
    if flush_data:
        with open(output_h264, "ab") as f:
            for data in flush_data:
                f.write(data)

    # 关闭编码器
    encoder.close()

    encode_end = time.monotonic()
    encode_time = encode_end - encode_start
    results["encode_time"] = encode_time
    results["total_frames_encoded"] = total_frames_encoded
    logger.info(f"[阶段3] 编码完成: {total_frames_encoded} 帧, {encode_time:.2f}s "
                f"({total_frames_encoded/encode_time:.0f} fps)")

    # ==================== 阶段 4: 封装 MP4 ====================
    logger.info("-" * 40)
    logger.info("[阶段4] 封装 MP4...")

    audio_path = ust_info.get("player_style", {}).get("audio_path", "")
    if audio_path and not os.path.exists(audio_path):
        audio_path = None

    mux_ok = _write_h264_to_mp4(output_h264, output_mp4, OUTPUT_FPS, audio_path)
    if mux_ok:
        logger.info(f"  输出文件: {output_mp4}")
        results["output_path"] = output_mp4
    else:
        logger.info(f"  封装失败，保留 H.264 原始文件: {output_h264}")
        results["output_path"] = output_h264

    # ==================== 统计 ====================
    t_end = time.monotonic()
    total_time = t_end - t_start
    results["total_time"] = total_time
    results["effective_fps"] = total_output_frames / total_time if total_time > 0 else 0

    logger.info("=" * 60)
    logger.info("基准测试结果:")
    logger.info(f"  加载工程:     {load_time:.2f}s")
    logger.info(f"  预计算:       {precompute_time:.2f}s")
    logger.info(f"  CUDA 渲染:    {render_time:.2f}s ({unique_count/render_time:.0f} fps)")
    logger.info(f"  NVENC 编码:   {encode_time:.2f}s ({total_frames_encoded/encode_time:.0f} fps)")
    logger.info(f"  ─────────────────────────────")
    logger.info(f"  总耗时:       {total_time:.2f}s")
    logger.info(f"  视频时长:     {video_duration:.1f}s")
    logger.info(f"  有效帧率:     {results['effective_fps']:.0f} fps")
    logger.info(f"  输出文件:     {results['output_path']}")
    logger.info("=" * 60)

    return results


# ===================== 优化版渲染（直接渲染到 NV12 缓冲区）====================


def _render_direct_to_nv12(
    frame_states: List[FrameState],
    width: int,
    height: int,
    fonts: dict,
    nv12_buffer: "cp.ndarray",
) -> None:
    """直接渲染到预分配的 NV12 缓冲区（GPU 显存中），避免额外拷贝。

    此函数对每个唯一帧执行：
        1. CUDA 渲染到 RGBA 帧缓冲
        2. CuPy 转换 RGBA → NV12
        3. 将 NV12 数据复制到预分配缓冲区

    整个过程数据不出 GPU 显存。
    """
    import cupy as cp
    from core.renderer import _get_cuda_ctx, _cuda_draw_polyline, _get_glyph_texture, _blit_texture
    from PySide6.QtGui import QColor, QFontMetrics

    nv12_y_size = width * height
    nv12_uv_size = width * (height // 2)

    for idx, state in enumerate(frame_states):
        ctx = _get_cuda_ctx(width, height)
        fb = ctx.fb
        stream = ctx.stream

        with stream:
            # ---- 重置帧缓冲 ----
            bg = QColor(state.bg_color)
            fb[:, :, 0] = bg.blue()
            fb[:, :, 1] = bg.green()
            fb[:, :, 2] = bg.red()
            fb[:, :, 3] = 255

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
                _cuda_draw_polyline(fb, state.pitch_points, QColor(state.pitch_curve_color), 5, width, height)

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

            # ---- LRC ----
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

        stream.synchronize()

        # GPU 端 RGBA → NV12
        b = fb[:, :, 0].astype(cp.float32)
        g = fb[:, :, 1].astype(cp.float32)
        r = fb[:, :, 2].astype(cp.float32)

        y = (0.299 * r + 0.587 * g + 0.114 * b).astype(cp.uint8)

        r_sub = r[::2, ::2]
        g_sub = g[::2, ::2]
        b_sub = b[::2, ::2]
        u = cp.clip(-0.169 * r_sub - 0.331 * g_sub + 0.500 * b_sub + 128, 0, 255).astype(cp.uint8)
        v = cp.clip(0.500 * r_sub - 0.419 * g_sub - 0.081 * b_sub + 128, 0, 255).astype(cp.uint8)

        uv = cp.empty((height // 2, width), dtype=cp.uint8)
        uv[:, 0::2] = u
        uv[:, 1::2] = v

        # 复制到预分配缓冲区（原地更新，无需新分配）
        nv12_buffer[:nv12_y_size] = y.ravel()
        nv12_buffer[nv12_y_size:] = uv.ravel()


def run_optimized_benchmark() -> dict:
    """优化版基准测试：直接渲染到预分配 NV12 缓冲区，减少 GPU 内存分配开销。

    与 run_benchmark() 的区别：
        - 预分配一个 NV12 缓冲区用于所有帧
        - 每帧渲染后直接写入同一缓冲区（原地更新）
        - NVENC 编码时使用同一缓冲区的设备指针（零拷贝）
    """
    results = {}
    t_start = time.monotonic()

    # ==================== 阶段 0: 加载工程 ====================
    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("优化版零拷贝 CUDA → NVENC 基准测试")
    logger.info(f"工程文件: {UPLR_PATH}")
    logger.info(f"输出分辨率: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} @ {OUTPUT_FPS}fps")
    logger.info("=" * 60)

    nvenc_ok, nvenc_msg = check_nvenc_available()
    logger.info(f"NVENC 检查: {nvenc_msg}")
    if not nvenc_ok:
        results["error"] = nvenc_msg
        return results

    ust_info = _load_uplr(UPLR_PATH)
    t1 = time.monotonic()
    load_time = t1 - t0
    results["load_time"] = load_time
    logger.info(f"[阶段0] 加载工程: {load_time:.2f}s")

    # ==================== 阶段 1: 预计算 ====================
    render_info = build_ust_info_for_render(ust_info, OUTPUT_WIDTH, OUTPUT_HEIGHT)
    fonts = render_info["_render_fonts"]

    frame_states = precompute_frame_states(ust_info, OUTPUT_FPS, OUTPUT_WIDTH, OUTPUT_HEIGHT)
    unique_count = len(frame_states)
    total_output_frames = sum(s.frame_count for s in frame_states)
    video_duration = total_output_frames / OUTPUT_FPS

    t2 = time.monotonic()
    precompute_time = t2 - t1
    results["precompute_time"] = precompute_time
    results["unique_frames"] = unique_count
    results["output_frames"] = total_output_frames
    results["video_duration"] = video_duration
    logger.info(f"  唯一帧: {unique_count}, 输出帧: {total_output_frames}, 时长: {video_duration:.1f}s")
    logger.info(f"[阶段1] 预计算: {precompute_time:.2f}s")

    if unique_count == 0:
        results["error"] = "预计算产生 0 帧"
        return results

    # ==================== 阶段 2: CUDA 渲染 + 零拷贝 NVENC 编码（流水线）====================
    logger.info("-" * 40)
    logger.info("[阶段2+3] CUDA 渲染 + NVENC 零拷贝编码（流水线并行）...")

    try:
        import cupy as cp
    except ImportError:
        results["error"] = "cupy 未安装"
        return results

    # 预构建字形图集
    _clear_glyph_cache()
    _build_glyph_cache(frame_states, fonts)

    # 获取 CUDA 上下文
    cuda_device = cp.cuda.Device(0)
    cuda_ctx = cuda_device.ctx
    cuda_ctx_handle = int(cuda_ctx)

    # 预分配 NV12 缓冲区（在 GPU 显存中）
    nv12_size = OUTPUT_WIDTH * OUTPUT_HEIGHT + OUTPUT_WIDTH * (OUTPUT_HEIGHT // 2)
    nv12_buffer = cp.zeros(nv12_size, dtype=cp.uint8)
    device_ptr = get_nvenc_device_ptr(nv12_buffer)

    logger.info(f"  NV12 缓冲区: {nv12_size} 字节 ({nv12_size/1024/1024:.1f}MB) @ GPU 指针 {hex(device_ptr)}")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_h264 = os.path.join(OUTPUT_DIR, "output_optimized.h264")
    output_mp4 = os.path.join(OUTPUT_DIR, "output_optimized.mp4")

    # 打开 NVENC 编码器
    encoder = NvencEncoder()
    encoder.open(cuda_ctx_handle)
    encoder.init(
        width=OUTPUT_WIDTH,
        height=OUTPUT_HEIGHT,
        fps=OUTPUT_FPS,
        bitrate=NVENC_BITRATE,
    )

    # 注册 NV12 缓冲区（注册一次，复用全程）
    encoder.register_buffer(device_ptr)

    # 渲染 + 编码流水线（零拷贝）
    render_start = time.monotonic()
    total_frames_encoded = 0

    with open(output_h264, "wb") as f:
        for idx, state in enumerate(frame_states):
            # 直接渲染到预分配 NV12 缓冲区（GPU 显存中）
            _render_direct_to_nv12([state], OUTPUT_WIDTH, OUTPUT_HEIGHT, fonts, nv12_buffer)

            repeat_count = state.frame_count

            # NVENC 直接从 GPU 显存编码（零拷贝：NV12 数据已在 GPU 显存中）
            for _ in range(repeat_count):
                bitstream = encoder.encode_frame()
                if bitstream:
                    f.write(bitstream)
                total_frames_encoded += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == unique_count:
                elapsed = time.monotonic() - render_start
                fps = total_frames_encoded / elapsed if elapsed > 0 else 0
                logger.info(f"  进度: {idx + 1}/{unique_count} 唯一帧, "
                            f"{total_frames_encoded}/{total_output_frames} 输出帧, "
                            f"{fps:.0f} fps")

    # 刷新编码器
    flush_data = encoder.flush()
    if flush_data:
        with open(output_h264, "ab") as f:
            for data in flush_data:
                f.write(data)

    encoder.close()
    _clear_glyph_cache()
    _clear_cuda_contexts()

    render_encode_end = time.monotonic()
    render_encode_time = render_encode_end - render_start
    results["render_encode_time"] = render_encode_time
    results["total_frames_encoded"] = total_frames_encoded
    logger.info(f"[阶段2+3] 渲染 + 编码完成: {total_frames_encoded} 帧, "
                f"{render_encode_time:.2f}s ({total_frames_encoded/render_encode_time:.0f} fps)")

    # ==================== 阶段 4: 封装 MP4 ====================
    audio_path = ust_info.get("player_style", {}).get("audio_path", "")
    if audio_path and not os.path.exists(audio_path):
        audio_path = None

    mux_ok = _write_h264_to_mp4(output_h264, output_mp4, OUTPUT_FPS, audio_path)
    if mux_ok:
        results["output_path"] = output_mp4
    else:
        results["output_path"] = output_h264

    t_end = time.monotonic()
    total_time = t_end - t_start
    results["total_time"] = total_time
    results["effective_fps"] = total_output_frames / total_time if total_time > 0 else 0

    logger.info("=" * 60)
    logger.info("优化版基准测试结果:")
    logger.info(f"  加载工程:     {load_time:.2f}s")
    logger.info(f"  预计算:       {precompute_time:.2f}s")
    logger.info(f"  渲染+编码:    {render_encode_time:.2f}s ({total_frames_encoded/render_encode_time:.0f} fps)")
    logger.info(f"  ─────────────────────────────")
    logger.info(f"  总耗时:       {total_time:.2f}s")
    logger.info(f"  视频时长:     {video_duration:.1f}s")
    logger.info(f"  有效帧率:     {results['effective_fps']:.0f} fps")
    logger.info(f"  输出文件:     {results['output_path']}")
    logger.info("=" * 60)

    return results


# ===================== 入口 =====================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="零拷贝 CUDA → NVENC 基准测试")
    parser.add_argument("--optimized", action="store_true", default=True,
                        help="使用优化版（预分配缓冲区 + 零拷贝流水线，默认开启）")
    parser.add_argument("--standard", action="store_true",
                        help="运行标准版（逐帧渲染 + NVENC 编码）")
    parser.add_argument("--runs", type=int, default=1,
                        help="测试次数（默认 1）")
    parser.add_argument("--width", type=int, default=OUTPUT_WIDTH,
                        help="输出宽度（默认 1920）")
    parser.add_argument("--height", type=int, default=OUTPUT_HEIGHT,
                        help="输出高度（默认 1080）")
    parser.add_argument("--fps", type=int, default=OUTPUT_FPS,
                        help="输出帧率（默认 30）")
    parser.add_argument("--preset", type=str, default="p1",
                        help="NVENC 预设（默认 p1，可选 p1-p7）")

    args = parser.parse_args()

    OUTPUT_WIDTH = args.width
    OUTPUT_HEIGHT = args.height
    OUTPUT_FPS = args.fps
    NVENC_PRESET = args.preset

    all_results = []
    for run in range(args.runs):
        logger.info(f"\n\n=== 第 {run + 1}/{args.runs} 次测试 ===")
        if args.standard and not args.optimized:
            result = run_benchmark()
        else:
            result = run_optimized_benchmark()
        all_results.append(result)

    # 汇总
    if len(all_results) > 1:
        total_times = [r.get("total_time", 0) for r in all_results if "error" not in r]
        if total_times:
            avg_total = sum(total_times) / len(total_times)
            min_total = min(total_times)
            logger.info(f"\n\n=== 汇总 ({len(total_times)} 次有效测试) ===")
            logger.info(f"  平均总耗时: {avg_total:.2f}s")
            logger.info(f"  最短总耗时: {min_total:.2f}s")