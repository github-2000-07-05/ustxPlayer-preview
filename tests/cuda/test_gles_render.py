#!/usr/bin/env python3
"""GLES 渲染后端集成测试：验证 FBO 离屏渲染 + NV12 直出管线。

测试目标：
    1. 功能正确：GLES 渲染器能正确创建上下文、FBO、渲染帧并回读
    2. 性能：1080p 单帧渲染耗时 < 20ms（目标 10ms）
    3. 管线集成：render_video 使用 GLES 后端能成功导出（逐帧渲染编码）

用法：
    python tests/cuda/test_gles_render.py
"""
import os
import sys
import json
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.renderer import (
    render_video, precompute_frame_states, _GLES_RENDERER, logger,
    _draw_with_painter, _init_render_fonts, FrameState,
)
from core.ustxreader import get_ustx_info_from_content
from PySide6.QtGui import QGuiApplication, QImage, QColor

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
UPLR_PATH = os.path.join(os.path.dirname(__file__), "..", "庙堂之外.uplr")
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 30


def _load_uplr(uplr_path: str) -> dict:
    """加载 UPLR 工程文件，返回 ust_info。"""
    with open(uplr_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    fmt = payload.get("format", "")
    if fmt != "ustxPlayer-preview.uplr" or payload.get("version") != 2:
        raise ValueError("不是有效的 ustxPlayer-preview 工程文件")

    ustx_content = payload.get("ustx_content", "")
    core_ust_info = get_ustx_info_from_content(ustx_content)
    logger.info(f"USTX 解析完成: {len(core_ust_info.get('notes', []))} 个音符, "
                f"BPM={core_ust_info.get('tempo')}")

    settings = payload.get("settings", {})
    styles = settings.get("styles", [])
    active_idx = settings.get("active_style_index", 0)
    active_idx = max(0, min(active_idx, len(styles) - 1)) if styles else 0
    ap = styles[active_idx] if styles else {}

    sc = settings.get("show_config", {}) or {}
    pi = {
        "song_name": settings.get("song_name", ""),
        "song_author": settings.get("song_author", ""),
        "ust_author": settings.get("ust_author", ""),
        "project_name": settings.get("project_name", ""),
    }

    note_styles_raw = settings.get("note_styles", {})
    note_styles = {}
    if isinstance(note_styles_raw, dict):
        for k, v in note_styles_raw.items():
            try:
                note_styles[int(k)] = int(v)
            except (ValueError, TypeError):
                pass

    audio_path = settings.get("audio_path", "")
    if not audio_path:
        default_audio = r"C:/Users/Danny/Desktop/ustPlayer-main/新建文件夹/庙堂之外 (Live) - 胡彦斌.flac"
        if os.path.exists(default_audio):
            audio_path = default_audio

    ust_info = {
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
            "curve_show": True,
        },
        "project_info": pi,
        "player_style": {
            "bg_color": ap.get("bg_color", settings.get("bg_color", "#000000")),
            "global_bg_color": settings.get("global_bg_color", "#00ff00"),
            "global_bg_enabled": bool(settings.get("global_bg_enabled", False)),
            "note_color": ap.get("note_color", settings.get("note_color", "#6c6c6c")),
            "lyric_color": ap.get("lyric_color", settings.get("lyric_color", "#ffffff")),
            "pitch_curve_color": ap.get("pitch_curve_color", "#ffffff"),
            "info_text_color": settings.get("info_text_color", "#ffffff"),
            "lyric_pos": settings.get("lyric_pos", "上"),
            "show_phoneme": False,
            "show_midinote": False,
            "show_waveform": False,
            "fullscreen": False,
            "lrc_path": settings.get("lrc_path", ""),
            "audio_path": audio_path,
            "silent_display": settings.get("silent_display", "R"),
            "silent_custom_text": settings.get("silent_custom_text", ""),
            "end_display": settings.get("end_display", "END"),
            "end_custom_text": settings.get("end_custom_text", ""),
            "pitch_placeholder": settings.get("pitch_placeholder", "无"),
            "pitch_custom_text": settings.get("pitch_custom_text", ""),
            "word_lyric_font_family": settings.get("word_lyric_font_family", "等线"),
            "info_font_family": settings.get("info_font_family", "微软雅黑"),
            "styles": styles,
            "note_styles": note_styles,
        },
    }
    return ust_info


def test_gles_single_frame():
    """测试 GLES 渲染器单帧渲染正确性和性能。"""
    print("\n" + "=" * 60)
    print("测试 1: GLES 单帧渲染")
    print("=" * 60)

    # 创建测试状态
    state = FrameState(
        lyric="测试歌词",
        note_name="C4",
        bg_color="#1a1a2e",
        lyric_color=(255, 255, 255),
        note_color=(200, 200, 200),
        pitch_curve_color="#ffffff",
        pitch_points=[(100, 500), (200, 450), (300, 400), (400, 350)],
        show_lyric=True,
        show_note_name=True,
        show_curve=True,
        song_name="测试歌曲",
        song_author="测试作者",
        ust_author="UST作者",
        tempo=120,
        show_bpm=True,
        show_song_name=True,
        show_song_author=True,
        show_ust_author=True,
        show_copyright=True,
        show_play_time=True,
        lrc_lines=["第一行歌词", "第二行歌词"],
        lrc_hidden=False,
        lyric_pos="上",
        word_lyric_font_family="等线",
        info_font_family="微软雅黑",
        small_font_color="#ffffff",
        copyright_text="测试版权 © 2026",
        start_time=0.0,
        duration=1.0,
        frame_count=60,
        is_duplicate=False,
        cache_key="test_key",
    )

    fonts = _init_render_fonts({
        "word_lyric_font_family": "等线",
        "info_font_family": "微软雅黑",
    }, OUTPUT_WIDTH, OUTPUT_HEIGHT)

    # 预热：第一次渲染会初始化 GLES 上下文
    print("  [预热] 初始化 GLES 上下文...")
    _GLES_RENDERER.ensure_init(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    img = _GLES_RENDERER.render_frame(state, OUTPUT_WIDTH, OUTPUT_HEIGHT, fonts)
    if img is None or img.isNull():
        print("  ✗ 预热渲染失败！")
        return False
    print(f"  [预热] 完成，QImage 尺寸: {img.width()}x{img.height()}")

    # 多次渲染测试性能
    N = 10
    times = []
    for i in range(N):
        t0 = time.monotonic()
        img = _GLES_RENDERER.render_frame(state, OUTPUT_WIDTH, OUTPUT_HEIGHT, fonts)
        elapsed = time.monotonic() - t0
        times.append(elapsed)

        if img is None or img.isNull():
            print(f"  ✗ 第 {i+1} 次渲染返回空图像！")
            return False

    avg_time = sum(times) / len(times) * 1000  # ms
    max_time = max(times) * 1000  # ms
    min_time = min(times) * 1000  # ms

    print(f"  渲染 {N} 帧统计:")
    print(f"    平均: {avg_time:.2f}ms")
    print(f"    最大: {max_time:.2f}ms")
    print(f"    最小: {min_time:.2f}ms")
    print(f"    等效帧率: {1000/avg_time:.0f} fps")

    # 验证 QImage 格式正确（Format_RGBA8888）
    if img.format() != QImage.Format.Format_RGBA8888:
        print(f"  ✗ QImage 格式错误: {img.format()}, 期望 RGBA8888")
        return False
    print(f"  ✓ QImage 格式正确: Format_RGBA8888")

    # 验证像素数据非空
    ptr = img.bits()
    if ptr is None or len(ptr) == 0:
        print("  ✗ 像素数据为空！")
        return False
    print(f"  ✓ 像素数据正确: {len(ptr)} 字节")

    # 保存一帧验证视觉效果
    output_path = os.path.join(OUTPUT_DIR, "gles_test_frame.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img.save(output_path)
    print(f"  ✓ 测试帧已保存: {output_path}")

    success = avg_time < 20  # 目标 < 20ms
    print(f"  {'✓ 通过' if success else '✗ 未达标'}: 平均 {avg_time:.2f}ms {'<' if success else '>='} 20ms")
    return success


def test_gles_render_pipeline():
    """测试 GLES 渲染管线完整导出。"""
    print("\n" + "=" * 60)
    print("测试 2: GLES 渲染管线完整导出")
    print("=" * 60)

    if not os.path.exists(UPLR_PATH):
        print(f"  ✗ 工程文件不存在: {UPLR_PATH}")
        return False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "gles_render_pipeline.mp4")

    ust_info = _load_uplr(UPLR_PATH)
    states = precompute_frame_states(ust_info, OUTPUT_FPS, OUTPUT_WIDTH, OUTPUT_HEIGHT)
    curved = [s for s in states if s.pitch_points and
              max(p[1] for p in s.pitch_points) - min(p[1] for p in s.pitch_points) > 1]
    print(f"  [检查] 时间区间帧 {len(states)} 个，其中 {len(curved)} 个含转音曲线")

    t0 = time.monotonic()
    ok = render_video(
        ust_info, output_path,
        fps=OUTPUT_FPS, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT,
        mode="frame_by_frame", render_backend="opengl",
        progress_callback=None,
    )
    total = time.monotonic() - t0

    if ok and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  ✓ 导出成功: {output_path}")
        print(f"    总耗时: {total:.2f}s, 文件大小: {size_mb:.1f}MB")
        return True
    else:
        from core.renderer import get_last_render_error
        print(f"  ✗ 导出失败: {get_last_render_error()}")
        return False


def main() -> int:
    if not os.path.exists(UPLR_PATH):
        print(f"工程文件不存在: {UPLR_PATH}")
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 单帧测试
    frame_ok = test_gles_single_frame()

    # 管线测试
    pipeline_ok = test_gles_render_pipeline()

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"  单帧渲染: {'✓ 通过' if frame_ok else '✗ 失败'}")
    print(f"  管线导出: {'✓ 通过' if pipeline_ok else '✗ 失败'}")

    # 清理 GLES 资源
    _GLES_RENDERER._cleanup()

    return 0 if (frame_ok and pipeline_ok) else 1


if __name__ == "__main__":
    _app = QGuiApplication([])
    sys.exit(main())