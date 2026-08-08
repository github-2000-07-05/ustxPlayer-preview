#!/usr/bin/env python3
"""CUDA → NV12 → FFmpeg NVENC 高速基准测试。

核心理念：
    传统路线：CUDA 渲染 → RGBA→NV12 (GPU) → 下载到 CPU → 存 PNG → FFmpeg 读 PNG → 编码
    优化路线：CUDA 渲染 → RGBA→NV12 (GPU) → 下载到 CPU → 直接 pipe 给 FFmpeg → 编码

    优化点：
    1. 省去 PNG 压缩/解压（CPU 瓶颈）
    2. 省去 drawtext 滤镜（GPU 瓶颈，原 >60s 的元凶）
    3. 省去磁盘 I/O
    4. NV12 直接喂给 h264_nvenc，无需颜色转换

测试流程：
    1. 加载工程文件（庙堂之外.uplr）
    2. CPU 预计算帧状态（去重）
    3. CUDA 渲染唯一帧到 NV12（GPU 显存）
    4. 下载 NV12 到 CPU 内存
    5. 通过 pipe 流式喂给 FFmpeg h264_nvenc 编码
    6. 封装 MP4
    7. 报告各阶段耗时

目标：总耗时 < 20s，理想 < 10s
"""

import os
import sys
import json
import time
import subprocess
import threading
from typing import List, Optional, Tuple, Dict
import numpy as np

# ===================== 路径设置 =====================

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.renderer import (
    precompute_frame_states,
    build_ust_info_for_render,
    _render_frame_cuda,
    _rgba_to_nv12_gpu,
    _clear_glyph_cache,
    _build_glyph_cache,
    _clear_cuda_contexts,
    _init_render_fonts,
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

# 编码参数（极致速度）
NVENC_PRESET = "p1"
NVENC_QP = 28  # 质量可接受的最快设置


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
    ps = settings.get("player_style", {}) or {}

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
    # 验证 NVENC 支持
    try:
        r = subprocess.run([ffmpeg, "-encoders"], capture_output=True, text=True, timeout=10,
                           creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
        if "h264_nvenc" in r.stdout:
            return ffmpeg
    except Exception:
        pass
    return None


def _stream_encode(
    ffmpeg: str,
    nv12_frames: List[Tuple[bytes, int]],  # [(nv12_bytes, repeat_count), ...]
    output_path: str,
    total_frames: int,
    width: int,
    height: int,
    fps: int,
    progress_cb: Optional[callable] = None,
) -> bool:
    """通过 pipe 将 NV12 帧流式喂给 FFmpeg h264_nvenc 编码。

    使用 hwupload_cuda 将帧数据上传到 GPU 显存，再由 NVENC 直接编码。
    每帧 3.1MB，21600 帧约 67GB 数据通过 pipe，pipe 缓冲区自动反压。
    """
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "nv12",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        # 硬件上传 + NVENC 编码
        "-c:v", "h264_nvenc",
        "-preset", NVENC_PRESET,
        "-rc", "constqp", "-qp", str(NVENC_QP),
        "-bf", "0",
        "-spatial-aq", "0", "-temporal-aq", "0",
        "-rc-lookahead", "0", "-no-scenecut", "1",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    logger.info(f"  FFmpeg 命令: {' '.join(cmd[:8])}...")

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

    frames_sent = 0
    write_start = time.monotonic()

    for idx, (nv12_bytes, repeat_count) in enumerate(nv12_frames):
        for _ in range(repeat_count):
            try:
                proc.stdin.write(nv12_bytes)
                frames_sent += 1
            except BrokenPipeError:
                logger.error(f"  Pipe 断开 (帧 {frames_sent})")
                break
            except Exception as e:
                logger.error(f"  写入 pipe 失败 (帧 {frames_sent}): {e}")
                break

        # 进度回调
        if progress_cb and (idx % 50 == 0 or idx == len(nv12_frames) - 1):
            elapsed = time.monotonic() - write_start
            progress_cb(frames_sent, total_frames, elapsed)

    # 关闭 stdin 等待 FFmpeg 完成
    try:
        proc.stdin.close()
    except Exception:
        pass

    stdout, stderr = proc.communicate(timeout=120)

    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace")[-2000:]
        logger.error(f"  FFmpeg 编码失败 (返回码 {proc.returncode}):")
        for line in stderr_text.split("\n")[-20:]:
            logger.error(f"    {line.strip()}")
        return False

    return True


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


# ===================== 基准测试 =====================


def run_benchmark() -> dict:
    """运行完整基准测试，返回各阶段耗时统计。"""
    results = {}
    t_start = time.monotonic()

    # ==================== 阶段 0: 加载工程 + 检查 FFmpeg ====================
    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("CUDA → NV12 → FFmpeg NVENC 高速基准测试")
    logger.info(f"工程文件: {UPLR_PATH}")
    logger.info(f"输出分辨率: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} @ {OUTPUT_FPS}fps")
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

    # 检查是否有去重效果
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

    # ==================== 阶段 2: CUDA 渲染 ====================
    logger.info("-" * 40)
    logger.info("[阶段2] CUDA 渲染 → NV12 转换...")

    # 预构建字形图集
    _clear_glyph_cache()
    _build_glyph_cache(frame_states, fonts)
    logger.info(f"  字形图集构建完成: {len(frame_states)} 帧")

    # 逐帧渲染为 NV12 字节
    nv12_frames: List[Tuple[bytes, int]] = []
    render_start = time.monotonic()

    for idx, state in enumerate(frame_states):
        try:
            # CUDA 渲染到 RGBA 帧缓冲 → GPU 上转 NV12 → 下载到 CPU
            img = _render_frame_cuda(state, OUTPUT_WIDTH, OUTPUT_HEIGHT, fonts)

            # 获取 NV12 字节（_render_frame_cuda 已经在 GPU 上转了 NV12）
            nv12_bytes = getattr(img, "_nv12_bytes", None)
            if nv12_bytes is None:
                # 回退：CPU 转换
                import numpy as np
                nv12_bytes = _rgba_to_nv12_cpu_fallback(img, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                if nv12_bytes is None:
                    raise RuntimeError("无法获取 NV12 数据")

            nv12_frames.append((nv12_bytes, state.frame_count))
            del img

            if (idx + 1) % 50 == 0 or (idx + 1) == unique_count:
                logger.info(f"  渲染进度: {idx + 1}/{unique_count} "
                            f"({(idx+1)/unique_count*100:.0f}%)")

        except Exception as e:
            logger.exception(f"渲染帧 {idx} 失败")
            raise

    render_end = time.monotonic()
    render_time = render_end - render_start
    results["render_time"] = render_time
    results["render_fps"] = unique_count / render_time if render_time > 0 else 0
    logger.info(f"[阶段2] 渲染完成: {unique_count} 帧, {render_time:.2f}s "
                f"({unique_count/render_time:.0f} fps)")

    # 释放 GPU 资源
    _clear_glyph_cache()
    _clear_cuda_contexts()

    # ==================== 阶段 3: 流式编码 ====================
    logger.info("-" * 40)
    logger.info("[阶段3] NV12 → FFmpeg NVENC 流式编码...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_h264 = os.path.join(OUTPUT_DIR, "benchmark_output.h264")
    output_mp4 = os.path.join(OUTPUT_DIR, "benchmark_output.mp4")

    encode_start = time.monotonic()

    def progress_cb(sent: int, total: int, elapsed: float):
        if sent > 0 and elapsed > 0:
            logger.info(f"  编码进度: {sent}/{total} ({sent/total*100:.0f}%), "
                        f"{sent/elapsed:.0f} fps, 已用 {elapsed:.1f}s")

    # 先编码到 H.264 文件（避免 pipe 数据量过大导致内存问题）
    encode_ok = _stream_encode(
        ffmpeg, nv12_frames, output_h264, total_output_frames,
        OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FPS, progress_cb,
    )

    if not encode_ok:
        results["error"] = "编码失败"
        results["total_time"] = time.monotonic() - t_start
        return results

    encode_end = time.monotonic()
    encode_time = encode_end - encode_start
    results["encode_time"] = encode_time
    results["encode_fps"] = total_output_frames / encode_time if encode_time > 0 else 0
    logger.info(f"[阶段3] 编码完成: {total_output_frames} 帧, {encode_time:.2f}s "
                f"({total_output_frames/encode_time:.0f} fps)")

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
    logger.info(f"  CUDA 渲染:    {render_time:.2f}s ({results['render_fps']:.0f} fps)")
    logger.info(f"  NVENC 编码:   {encode_time:.2f}s ({results['encode_fps']:.0f} fps)")
    logger.info(f"  ─────────────────────────────")
    logger.info(f"  总耗时:       {total_time:.2f}s")
    logger.info(f"  视频时长:     {video_duration:.1f}s")
    logger.info(f"  有效帧率:     {results['effective_fps']:.0f} fps")
    logger.info(f"  去重比:       {dedup_ratio:.1f}x ({unique_count} 唯一帧)")
    logger.info(f"  输出文件:     {results['output_path']}")
    logger.info("=" * 60)

    return results


def _rgba_to_nv12_cpu_fallback(img, width: int, height: int) -> Optional[bytes]:
    """CPU 端 RGBA→NV12 转换回退。"""
    try:
        import numpy as np
        # 从 QImage 提取 numpy 数组
        numpy_ref = getattr(img, "_numpy_ref", None)
        if numpy_ref is not None:
            rgba_np = numpy_ref
        else:
            ptr = img.bits()
            rgba_np = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))

        # Qt ARGB32 in memory = BGRA
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


# ===================== 入口 =====================


if __name__ == "__main__":
    # 必须先创建 QGuiApplication（QFont/QFontMetrics 需要）
    from PySide6.QtGui import QGuiApplication
    _app = QGuiApplication([])

    import argparse

    parser = argparse.ArgumentParser(description="CUDA → NV12 → FFmpeg NVENC 基准测试")
    parser.add_argument("--runs", type=int, default=1, help="测试次数")
    parser.add_argument("--width", type=int, default=OUTPUT_WIDTH)
    parser.add_argument("--height", type=int, default=OUTPUT_HEIGHT)
    parser.add_argument("--fps", type=int, default=OUTPUT_FPS)
    parser.add_argument("--preset", type=str, default=NVENC_PRESET, help="NVENC 预设")

    args = parser.parse_args()

    OUTPUT_WIDTH = args.width
    OUTPUT_HEIGHT = args.height
    OUTPUT_FPS = args.fps
    NVENC_PRESET = args.preset

    all_results = []
    for run in range(args.runs):
        logger.info(f"\n\n=== 第 {run + 1}/{args.runs} 次测试 ===")
        result = run_benchmark()
        all_results.append(result)

    # 汇总
    if len(all_results) > 1:
        valid = [r for r in all_results if "error" not in r]
        if valid:
            total_times = [r.get("total_time", 0) for r in valid]
            render_times = [r.get("render_time", 0) for r in valid]
            encode_times = [r.get("encode_time", 0) for r in valid]
            logger.info(f"\n\n=== 汇总 ({len(valid)} 次有效测试) ===")
            logger.info(f"  平均总耗时: {sum(total_times)/len(total_times):.2f}s")
            logger.info(f"  最短总耗时: {min(total_times):.2f}s")
            logger.info(f"  平均渲染:   {sum(render_times)/len(render_times):.2f}s")
            logger.info(f"  平均编码:   {sum(encode_times)/len(encode_times):.2f}s")