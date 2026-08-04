# ustxreader.py — USTX 文件解析器
"""OpenUtau USTX (YAML) 文件解析模块。

提取 USTX 文件中的版本、速度、轨道数和音符信息，
供播放器和主窗口使用。
"""

from typing import List, Dict, Union
import os

import yaml

from core.log import logger


# ===================== 音轨检测 =====================


def get_ustx_tracks(ustx_path: str) -> List[Dict]:
    """检测 USTX 文件中可解析的音轨（含音符的 voice_parts）。

    供 UI 在存在多条可解析音轨时弹出选择窗口使用。

    Args:
        ustx_path: USTX 文件路径（.ustx 或 .txt）

    Returns:
        list: 按文件内顺序排列的可解析音轨信息 dict：
            [{"index": int, "name": str, "note_count": int}, ...]
        没有含音符的 voice_parts 时返回空列表。

    Raises:
        FileNotFoundError: USTX 文件不存在
    """
    if not os.path.exists(ustx_path):
        raise FileNotFoundError(f"USTX 文件不存在: {ustx_path}")

    with open(ustx_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        data = {}

    tracks: List[Dict] = []
    for i, part in enumerate(data.get('voice_parts', [])):
        if not isinstance(part, dict):
            continue
        notes = part.get('notes', [])
        if not isinstance(notes, list) or not notes:
            continue
        tracks.append({
            "index": i,
            "name": str(part.get('name', '') or f"音轨 {i + 1}"),
            "note_count": len(notes),
        })
    return tracks


# ===================== 主解析函数 =====================


def get_ustx_info(ustx_path: str, track_index: Union[int, None] = None) -> Dict[str, Union[str, float, int, List[Dict]]]:
    """解析 USTX 文件（YAML 格式），提取版本、速度、轨道数和音符列表。

    USTX 是 OpenUtau 使用的 YAML 格式文件，包含更丰富的信息。
    多音轨文件可通过 track_index 指定仅解析某一条音轨（voice_parts 下标），
    默认（None）合并全部音轨，保持向后兼容。

    Args:
        ustx_path: USTX 文件路径（.ustx 或 .txt）
        track_index: 需要解析的音轨下标（voice_parts 中的位置）；None 表示合并全部音轨

    Returns:
        dict:
            version (str):    USTX 版本号
            tempo (float):    速度 (BPM)
            tracks (int):     轨道数
            track_name (str): 当前解析音轨名称（合并全部音轨时为 "全部音轨"）
            notes (list):     音符列表 [{index, length, lyric, note_num, pitch_bend}]

    Raises:
        FileNotFoundError: USTX 文件不存在
    """
    if not os.path.exists(ustx_path):
        raise FileNotFoundError(f"USTX 文件不存在: {ustx_path}")

    with open(ustx_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        # 空文件或非映射（如纯列表/标量）→ 视为无内容，避免 AttributeError
        data = {}

    ustx_version = data.get('ustx_version', 'unknown')
    ustx_tempo = float(data.get('bpm', 120.0))
    ustx_tracks = max(1, len(data.get('tracks', [])))

    note_list: List[Dict] = []

    # 按需过滤音轨：指定 track_index 时仅解析该音轨，否则合并全部
    track_name = "全部音轨"
    voice_parts = data.get('voice_parts', [])
    if track_index is not None:
        if 0 <= track_index < len(voice_parts):
            part0 = voice_parts[track_index]
            track_name = str(part0.get('name', '') or f"音轨 {track_index + 1}")
            voice_parts = [part0]
        else:
            logger.warning(f"音轨下标越界: {track_index}，已回退为合并全部音轨")

    for part in voice_parts:
        part_pos = part.get('position', 0)
        notes = part.get('notes', [])

        # 从 voice_part.curves 提取 pitd（pitch deviation）曲线数据
        # 规范上每个声部至多一条 pitd 曲线；防御性处理：若出现多条
        # （部分第三方导出工具可能产生冗余/重复曲线），全部收集并合并，
        # 避免只取第一条导致的数据丢失。
        pitd_xs: List[int] = []
        pitd_ys: List[int] = []
        pitd_count = 0
        for curve in part.get('curves', []):
            if isinstance(curve, dict) and curve.get('abbr') == 'pitd':
                pitd_count += 1
                xs = curve.get('xs', [])
                ys = curve.get('ys', [])
                if isinstance(xs, list) and isinstance(ys, list):
                    pitd_xs.extend(xs)
                    pitd_ys.extend(ys)
        if pitd_count > 1:
            logger.warning(f"voice_part 包含 {pitd_count} 条 pitd 曲线，已全部合并数据点")

        # 构建 tick→pitch 查找表，并预排序 tick（每个 part 仅排序一次，避免每音符重复排序）
        tick_pitch = {}
        if len(pitd_xs) != len(pitd_ys):
            logger.warning(f"pitd 曲线 xs/ys 长度不一致: {len(pitd_xs)} vs {len(pitd_ys)}，按短的截断")
        for x, y in zip(pitd_xs, pitd_ys):
            tick_pitch[int(x)] = int(y)
        sorted_ticks = sorted(tick_pitch.keys())

        for i, note in enumerate(notes):
            note_num = note.get('tone', 0)
            lyric = note.get('lyric', '')
            duration = note.get('duration', 0)
            note_pos = part_pos + note.get('position', 0)
            note_end = note_pos + duration

            # 从 tick_pitch 提取该音符范围内的 pitch_bend
            pitch_bend: List[int] = []
            if sorted_ticks:
                for t in sorted_ticks:
                    if note_pos <= t <= note_end:
                        pitch_bend.append(tick_pitch[t])
            if not pitch_bend:
                pitch_bend = [0]
            elif len(pitch_bend) == 1:
                pitch_bend = pitch_bend * 2

            note_list.append({
                "index": f"{i:04d}",
                "position": note_pos,
                "length": duration,
                "lyric": lyric,
                "note_num": note_num,
                "pitch_bend": pitch_bend,
            })

    logger.info(f"USTX 解析完成: {len(note_list)} 个音符, BPM={ustx_tempo}")
    if note_list:
        logger.info(f"USTX 音符区间: pos={note_list[0]['position']}~{note_list[-1]['position']}, "
                    f"共 {note_list[-1]['position'] + note_list[-1]['length']} ticks")

    return {
        "version": ustx_version,
        "tempo": ustx_tempo,
        "tracks": ustx_tracks,
        "track_name": track_name,
        "notes": note_list,
    }
