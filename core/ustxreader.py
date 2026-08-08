# ustxreader.py — USTX 文件解析器
"""OpenUtau USTX (YAML) 文件解析模块。

提取 USTX 文件中的版本、速度、轨道数和音符信息，
供播放器和主窗口使用。

支持两种输入方式：
- 从文件路径读取（get_ustx_tracks / get_ustx_info）
- 从字符串内容直接解析（get_ustx_tracks_from_content / get_ustx_info_from_content）
  用于导入 uplr 工程时避免创建缓存文件
"""

from typing import List, Dict, Tuple, Optional, Union
import os

import yaml

from core.log import logger


# ===================== 公共解析逻辑 =====================


def _parse_ustx_tracks(data: dict) -> List[Dict]:
    """从已解析的 USTX dict 中检测可解析音轨（内部共用）。"""
    if not isinstance(data, dict):
        return []

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


def _interpolate_envelope(
    points: List[Tuple[float, float]], tick: float,
) -> float:
    """在 per-note pitch 包络点上线性插值，返回 tick 处的偏移值 (cents)。

    Args:
        points: [(absolute_tick, offset_cents), ...]，按 tick 升序
        tick: 需要插值的 tick 位置

    Returns:
        插值后的偏移值 (cents)；tick 超出范围时返回最近端点的值
    """
    if not points:
        return 0.0
    if len(points) == 1:
        return points[0][1]
    # 二分查找 tick 所在的区间
    lo, hi = 0, len(points) - 1
    if tick <= points[lo][0]:
        return points[lo][1]
    if tick >= points[hi][0]:
        return points[hi][1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if tick < points[mid][0]:
            hi = mid
        else:
            lo = mid
    # 线性插值
    x0, y0 = points[lo]
    x1, y1 = points[hi]
    if x1 == x0:
        return y0
    ratio = (tick - x0) / (x1 - x0)
    return y0 + ratio * (y1 - y0)


def _parse_ustx_info(data: dict, track_index: Union[int, None] = None) -> Dict[str, Union[str, float, int, List[Dict]]]:
    """从已解析的 USTX dict 中提取版本、速度、轨道数和音符列表（内部共用）。

    Args:
        data: yaml.safe_load 返回的 dict
        track_index: 需要解析的音轨下标；None 表示合并全部音轨

    Returns:
        dict: version, tempo, tracks, track_name, notes
    """
    if not isinstance(data, dict):
        data = {}

    ustx_version = data.get('ustx_version', 'unknown')
    ustx_tempo = float(data.get('bpm', 120.0))
    ustx_tracks = max(1, len(data.get('tracks', [])))

    note_list: List[Dict] = []

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

        # 构建 tick→pitch 查找表
        tick_pitch = {}
        if len(pitd_xs) != len(pitd_ys):
            logger.warning(f"pitd 曲线 xs/ys 长度不一致: {len(pitd_xs)} vs {len(pitd_ys)}，按短的截断")
        for x, y in zip(pitd_xs, pitd_ys):
            tick_pitch[int(x)] = int(y)
        sorted_ticks = sorted(tick_pitch.keys())

        # 预计算所有正常音符的 per-note pitch 结束值（用于 snap_first）
        _prev_note_pitch_at_end: Optional[float] = None

        for i, note in enumerate(notes):
            note_num = note.get('tone', 0)
            lyric = note.get('lyric', '')
            duration = note.get('duration', 0)
            note_pos = part_pos + note.get('position', 0)
            note_end = note_pos + duration

            pitch_bend: List[int] = []
            if sorted_ticks:
                for t in sorted_ticks:
                    if note_pos <= t <= note_end:
                        pitch_bend.append(tick_pitch[t])
            if not pitch_bend:
                pitch_bend = [0]
            elif len(pitch_bend) == 1:
                pitch_bend = pitch_bend * 2

            # ---- 合并 per-note pitch 数据（转音/过渡音高） ----
            # USTX 每个音符都包含 pitch.data 定义 per-note 音高包络，
            # 但旧代码只提取了 voice_part 级别的 pitd 曲线，忽略了 per-note 数据，
            # 导致过渡音符的转音（portamento）丢失。
            # 以下将 per-note pitch 包络插值到音符的每个 tick 上，叠加到 pitd 值。
            note_pitch_data = note.get('pitch')
            if isinstance(note_pitch_data, dict):
                pdata = note_pitch_data.get('data', [])
                snap_first = note_pitch_data.get('snap_first', False)
                if pdata and isinstance(pdata, list) and len(pdata) >= 2:
                    # 将 per-note pitch 包络点转换为 (absolute_tick, offset_cents)
                    note_center = note_pos + duration / 2.0
                    envelope_points: List[Tuple[float, float]] = []
                    for pi, pt in enumerate(pdata):
                        x = pt.get('x', 0)
                        y = pt.get('y', 0)
                        abs_tick = note_center + x
                        if pi == 0 and snap_first and _prev_note_pitch_at_end is not None:
                            # snap_first: 第一个点的 y 相对前一个音符的结束音高
                            # 偏移 = (前音符结束音高 - 当前音符基音) + y
                            cur_base = note_num * 100.0
                            offset = (_prev_note_pitch_at_end - cur_base) + y
                        else:
                            offset = float(y)
                        envelope_points.append((abs_tick, offset))

                    # 计算前音符结束音高（供下一个音符的 snap_first 使用）
                    # 包络最后一个点的 offset 相对当前音符基音
                    if envelope_points:
                        _prev_note_pitch_at_end = note_num * 100.0 + envelope_points[-1][1]

                    # 将 per-note 包络插值到每个 pitd tick 上，叠加到 pitch_bend
                    if duration > 0 and len(pitch_bend) > 0 and len(sorted_ticks) > 0:
                        note_ticks = [t for t in sorted_ticks if note_pos <= t <= note_end]
                        if note_ticks:
                            # 对每个 tick 插值 per-note 包络
                            for ti, tick in enumerate(note_ticks):
                                # 插值：找到包络上 tick 左右的两个点
                                offset = _interpolate_envelope(envelope_points, tick)
                                # 叠加到 pitd 值
                                pitch_bend[ti] = pitch_bend[ti] + int(round(offset))
                else:
                    # 没有足够包络点，但仍记录前音符结束音高
                    if len(pdata) == 1:
                        _prev_note_pitch_at_end = note_num * 100.0 + pdata[0].get('y', 0)
            else:
                # 没有 per-note pitch 数据（极少数情况）
                # 用 pitd 曲线的最后一个值作为前音符结束音高
                if pitch_bend:
                    _prev_note_pitch_at_end = note_num * 100.0 + pitch_bend[-1]

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


# ===================== 从字符串内容解析（供 uplr 导入使用，无需创建缓存文件）=====================


def get_ustx_tracks_from_content(content: str) -> List[Dict]:
    """从 USTX 字符串内容检测可解析的音轨。

    供导入 uplr 工程时使用，直接在内存中解析，无需写缓存文件。

    Args:
        content: USTX 文件的 YAML 文本内容

    Returns:
        list: 可解析音轨信息 dict 列表
    """
    data = yaml.safe_load(content)
    return _parse_ustx_tracks(data)


def get_ustx_info_from_content(
    content: str, track_index: Union[int, None] = None
) -> Dict[str, Union[str, float, int, List[Dict]]]:
    """从 USTX 字符串内容直接解析音符信息。

    供导入 uplr 工程时使用，直接在内存中解析，无需写缓存文件。

    Args:
        content: USTX 文件的 YAML 文本内容
        track_index: 需要解析的音轨下标；None 表示合并全部音轨

    Returns:
        dict: version, tempo, tracks, track_name, notes
    """
    data = yaml.safe_load(content)
    return _parse_ustx_info(data, track_index)


# ===================== 从文件路径解析（原有接口，保持兼容）=====================


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
    return _parse_ustx_tracks(data)


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
    return _parse_ustx_info(data, track_index)