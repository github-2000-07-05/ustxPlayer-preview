#!/usr/bin/env python3
"""验证 GLES 渲染器在后台线程（主项目 _RenderWorker 场景）中正常工作。

主项目的 render_video 由 _RenderWorker 在 threading.Thread 中调用，
GLES 上下文也在该线程中首次创建并渲染。本测试复现该场景。

用法：
    python tests/cuda/test_gles_thread.py
"""
import os
import sys
import time
import threading

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PySide6.QtGui import QGuiApplication

from core.renderer import (
    _GLES_RENDERER, _init_render_fonts, FrameState, logger,
)

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

_state = FrameState(
    lyric="后台线程测试",
    note_name="C4",
    bg_color="#1a1a2e",
    lyric_color=(255, 255, 255),
    note_color=(200, 200, 200),
    pitch_curve_color="#ffffff",
    pitch_points=[(100, 500), (200, 450), (300, 400), (400, 350)],
    show_lyric=True, show_note_name=True, show_curve=True,
    song_name="测试歌曲", song_author="测试作者", ust_author="UST作者",
    tempo=120, show_bpm=True, show_song_name=True, show_song_author=True,
    show_ust_author=True, show_copyright=True, show_play_time=True,
    lrc_lines=["第一行歌词", "第二行歌词"], lrc_hidden=False,
    lyric_pos="上", word_lyric_font_family="等线", info_font_family="微软雅黑",
    small_font_color="#ffffff", copyright_text="测试版权 © 2026",
    start_time=0.0, duration=1.0, frame_count=60,
    is_duplicate=False, cache_key="test_key",
)

_fonts = _init_render_fonts({
    "word_lyric_font_family": "等线",
    "info_font_family": "微软雅黑",
}, OUTPUT_WIDTH, OUTPUT_HEIGHT)

result = {"ok": False, "err": ""}


def _worker():
    try:
        # 在后台线程中首次初始化 GLES 上下文并渲染（模拟主项目 Worker）
        _GLES_RENDERER.ensure_init(OUTPUT_WIDTH, OUTPUT_HEIGHT)
        times = []
        for _ in range(10):
            t0 = time.monotonic()
            img = _GLES_RENDERER.render_frame(
                _state, OUTPUT_WIDTH, OUTPUT_HEIGHT, _fonts)
            times.append((time.monotonic() - t0) * 1000)
            if img is None or img.isNull():
                result["err"] = "渲染返回空图像"
                return
        avg = sum(times) / len(times)
        logger.info(f"后台线程 GLES 渲染 10 帧, avg={avg:.2f}ms ({1000/avg:.0f}fps)")
        result["ok"] = True
    except Exception as e:
        logger.exception("后台线程 GLES 渲染失败")
        result["err"] = f"{type(e).__name__}: {e}"
    finally:
        _GLES_RENDERER._cleanup()


def main() -> int:
    print("=" * 60)
    print("后台线程 GLES 渲染验证（主项目 Worker 场景）")
    print("=" * 60)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=120)

    if result["ok"]:
        print("  ✓ 后台线程 GLES 渲染成功")
        return 0
    print(f"  ✗ 后台线程 GLES 渲染失败: {result['err']}")
    return 1


if __name__ == "__main__":
    _app = QGuiApplication([])
    sys.exit(main())
