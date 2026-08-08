#!/usr/bin/env python3
"""GLES vs CPU 渲染性能对比基准测试。

测试 1920x1080@30fps 单帧渲染耗时，对比各后端。
"""
import os, sys, time, json
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.renderer import (
    _GLES_RENDERER, _render_frame_cpu, _render_frame_opengl,
    _init_render_fonts, FrameState,
)
from PySide6.QtGui import QGuiApplication, QPainter, QImage, QColor

# 需要先创建 QGuiApplication
_app = QGuiApplication([])

W, H = 1920, 1080

state = FrameState(
    lyric="测试歌词",
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

fonts = _init_render_fonts({
    "word_lyric_font_family": "等线",
    "info_font_family": "微软雅黑",
}, W, H)

def bench(name, func, n=20):
    # 预热
    func(state, W, H, fonts)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        func(state, W, H, fonts)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)
    avg = sum(times) / len(times)
    print(f"  {name:20s}  avg={avg:6.2f}ms  min={min(times):6.2f}ms  max={max(times):6.2f}ms  {1000/avg:5.0f}fps")
    return avg

print("=" * 60)
print("渲染后端性能对比 (1920x1080)")
print("=" * 60)

# CPU 渲染
cpu_avg = bench("CPU (QPainter→QImage)", _render_frame_cpu)

# GLES 渲染
gles_avg = bench("GLES (OpenGL)", _render_frame_opengl)

# 清理
_GLES_RENDERER._cleanup()

print("-" * 60)
ratio = cpu_avg / gles_avg
print(f"CPU/GLES 比例: {ratio:.2f}x")
print(f"730 帧预估: CPU={730*cpu_avg/1000:.1f}s  GLES={730*gles_avg/1000:.1f}s")
print(f"1460 帧预估: CPU={1460*cpu_avg/1000:.1f}s  GLES={1460*gles_avg/1000:.1f}s")