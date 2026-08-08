"""验证去重 cache_key 修复：同文本不同转音曲线的帧不再被错误合并。

背景 bug：
    precompute_frame_states 的去重 cache_key 之前只用 str(len(pitch_points))，
    当两个相邻时间段显示文本（歌词/音名）相同、但 pitch 曲线不同且点数恰好
    相同时，后一个状态会被合并进前一个状态，其转音曲线随之丢失。

本测试：
    构造两个连续 C4 音符（歌词相同 "啦"）：
        - 音符 A: pitch_bend = [0, 0, 0, 0, 0]         （平直，5 点）
        - 音符 B: pitch_bend = [0, 25, 50, 25, 0]      （转音，5 点）
    修复前：cache_key 相同 → 合并为 1 个状态，转音丢失。
    修复后：应生成 2 个状态，且第二个状态保留转音曲线。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.renderer import precompute_frame_states  # noqa: E402


def _build_test_ust_info() -> dict:
    """构造最小可用 ust_info：两个连续同音高 C4 音符，曲线开启。"""
    return {
        "version": "test",
        "tempo": 120.0,
        "tracks": 1,
        "notes": [
            {
                "index": "0000",
                "position": 0,
                "length": 480,
                "lyric": "啦",
                "note_num": 60,  # C4
                "pitch_bend": [0, 0, 0, 0, 0],       # 平直
            },
            {
                "index": "0001",
                "position": 480,
                "length": 480,
                "lyric": "啦",
                "note_num": 60,  # C4（同音高、同歌词 → 修复前 cache_key 相同）
                "pitch_bend": [0, 25, 50, 25, 0],    # 转音（上滑后回落）
            },
        ],
        "show_config": {
            "bpm": True,
            "play_time": True,
            "song_name": True,
            "song_author": True,
            "ust_author": True,
            "copyright": True,
            "lyric": True,
            "lyric_autohide": False,
            "lyric_autohide_threshold": 3.0,
            "curve_show": True,   # 关键：显示音高线
        },
        "project_info": {
            "song_name": "测试", "song_author": "tester",
            "ust_author": "tester", "project_name": "测试",
        },
        "player_style": {
            "bg_color": "#000000",
            "global_bg_color": "#00ff00",
            "global_bg_enabled": False,
            "note_color": "#6c6c6c",
            "lyric_color": "#ffffff",
            "pitch_curve_color": "#ffffff",
            "info_text_color": "#ffffff",
            "lyric_pos": "上",
            "show_phoneme": False,
            "show_midinote": False,
            "show_waveform": False,
            "fullscreen": False,
            "silent_display": "R",
            "silent_custom_text": "",
            "end_display": "END",
            "end_custom_text": "",
            "pitch_placeholder": "无",
            "pitch_custom_text": "",
            "word_lyric_font_family": "等线",
            "info_font_family": "微软雅黑",
            "styles": [],
            "note_styles": {},
        },
    }


def main() -> int:
    ust_info = _build_test_ust_info()
    states = precompute_frame_states(ust_info, fps=30, width=1920, height=1080)

    print("=" * 60)
    print("去重 cache_key 修复验证")
    print("=" * 60)
    print(f"唯一帧数量: {len(states)}（期望 ≥ 2：两个音符曲线不同，不应合并）")

    for i, s in enumerate(states):
        ys = [p[1] for p in s.pitch_points] if s.pitch_points else []
        print(f"  state[{i}]: lyric={s.lyric!r}, note={s.note_name!r}, "
              f"pts={len(ys)}, y 范围={min(ys):.0f}~{max(ys):.0f}")

    if len(states) < 2:
        print("\n✗ 失败：两个不同转音曲线的帧被错误合并，转音丢失！")
        return 1

    # 确认至少存在一个带转音（非平直）的曲线
    has_curve = any(
        len(s.pitch_points) >= 2
        and max(p[1] for p in s.pitch_points) - min(p[1] for p in s.pitch_points) > 1
        for s in states
    )
    if not has_curve:
        print("\n✗ 失败：未找到转音曲线，转音丢失！")
        return 1

    print("\n✓ 通过：转音曲线已正确保留，未被去重合并吞掉")
    return 0


if __name__ == "__main__":
    from PySide6.QtGui import QGuiApplication
    _app = QGuiApplication([])  # QFontMetrics 需要 QGuiApplication
    sys.exit(main())
