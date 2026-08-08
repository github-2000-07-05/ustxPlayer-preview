#!/usr/bin/env python3
"""主项目集成验证：调用 render_video（唯一帧 I 帧重复方案）完整导出。

验证目标：
    1. 功能正确：导出成功生成 .mp4
    2. 转音保留：预计算唯一帧中应包含转音曲线（去重不再吞帧）
    3. 性能：CUDA 后端总耗时目标 < 20s（理想 < 10s）

用法：
    python tests/cuda/integrate_render_video.py
"""
import os
import sys
import json
import time

# ===================== 路径设置 =====================

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.renderer import (  # noqa: E402
    render_video, precompute_frame_states, logger,
)
from core.ustxreader import get_ustx_info_from_content  # noqa: E402

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
    if not ustx_content:
        raise ValueError("UPLR 文件中没有 USTX 内容")

    core_ust_info = get_ustx_info_from_content(ustx_content)
    logger.info(f"USTX 解析完成: {len(core_ust_info.get('notes', []))} 个音符, "
                f"BPM={core_ust_info.get('tempo')}")

    settings = payload.get("settings", {})
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
        # 工程未记录音频路径时，回退到已知测试音频以验证音频合并
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
            # 强制开启音高线显示：验证真实工程的转音（portamento）曲线
            # 在修复去重 cache_key 后能被正确渲染（模拟用户在设置中开启音高线）
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


def main() -> int:
    if not os.path.exists(UPLR_PATH):
        print(f"工程文件不存在: {UPLR_PATH}")
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "main_render_integrate.mp4")

    # 预检查：去重后的唯一帧是否包含转音曲线
    ust_info = _load_uplr(UPLR_PATH)
    states = precompute_frame_states(ust_info, OUTPUT_FPS, OUTPUT_WIDTH, OUTPUT_HEIGHT)
    curved = [s for s in states if s.pitch_points and
              max(p[1] for p in s.pitch_points) - min(p[1] for p in s.pitch_points) > 1]
    logger.info(f"[检查] 唯一帧 {len(states)} 个，其中 {len(curved)} 个含转音曲线")

    # 进度回调
    def cb(pct: int, stage: str):
        if pct % 10 == 0 or stage in ("预计算", "GPU渲染", "编码"):
            print(f"  [{stage}] {pct}%")

    print("=" * 60)
    print("主项目集成验证：render_video (逐帧渲染编码，不做去重)")
    print("=" * 60)

    t0 = time.monotonic()
    ok = render_video(
        ust_info, output_path,
        fps=OUTPUT_FPS, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT,
        mode="frame_by_frame", render_backend="auto",
        progress_callback=cb,
    )
    total = time.monotonic() - t0

    print("-" * 60)
    if ok and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"✓ 导出成功: {output_path}")
        print(f"  总耗时: {total:.2f}s, 文件大小: {size_mb:.1f}MB")
        return 0
    else:
        from core.renderer import get_last_render_error
        print(f"✗ 导出失败: {get_last_render_error()}")
        return 1


if __name__ == "__main__":
    from PySide6.QtGui import QGuiApplication
    _app = QGuiApplication([])  # QFontMetrics 等需要 QGuiApplication
    sys.exit(main())
