"""分析 USTX 文件中过渡音符的字段和 pitch_bend 数据。"""
import json
import os
import yaml

uplr_path = os.path.join(os.path.dirname(__file__), '..', '庙堂之外.uplr')
with open(uplr_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

ustx_content = data['ustx_content']
yaml_data = yaml.safe_load(ustx_content)

vp = yaml_data['voice_parts'][0]
notes = vp['notes']

# 查看过渡音符的原始字段
trans_notes = [n for n in notes if n.get('lyric', '') in ('-', '+')]
print(f"过渡音符数量: {len(trans_notes)}")

# 查看第一个过渡音符的所有字段
tn = trans_notes[0]
print(f"\n第一个过渡音符的字段:")
for k, v in tn.items():
    print(f"  {k} = {repr(v)}")

# 查看它前后的正常音符
note_idx = notes.index(tn)
prev_notes = notes[max(0, note_idx-2):note_idx]
next_notes = notes[note_idx+1:note_idx+3]
print(f"\n过渡音符前后的音符:")
for i, n in enumerate(prev_notes):
    print(f"  prev[{i}]: {dict(n)}")
print(f"  current: {dict(tn)}")
for i, n in enumerate(next_notes):
    print(f"  next[{i}]: {dict(n)}")

# 检查所有过渡音符是否有 duration 字段
has_duration = sum(1 for n in trans_notes if 'duration' in n)
has_length = sum(1 for n in trans_notes if 'length' in n)
print(f"\n过渡音符: 有duration={has_duration}, 有length={has_length}")

# 检查所有普通音符是否有 duration 字段
normal_notes = [n for n in notes if n.get('lyric', '') not in ('-', '+')]
has_duration_n = sum(1 for n in normal_notes if 'duration' in n)
has_length_n = sum(1 for n in normal_notes if 'length' in n)
print(f"普通音符: 有duration={has_duration_n}, 有length={has_length_n}")

# 查看过渡音符的 duration 值
if trans_notes:
    durations = [n.get('duration', None) for n in trans_notes]
    lengths = [n.get('length', None) for n in trans_notes]
    print(f"\n过渡音符 duration 值: 有 {sum(1 for d in durations if d is not None)}/{len(durations)} 非空")
    print(f"过渡音符 length 值: 有 {sum(1 for d in lengths if d is not None)}/{len(lengths)} 非空")
    print(f"duration 样本(前10): {durations[:10]}")
    print(f"length 样本(前10): {lengths[:10]}")

# 检查 ustxreader 中的 _calc_note_tick_ranges
# 它使用 note.get('length', 480)，但 USTX 中可能是 duration
# 这会导致过渡音符的 tick 区间计算错误！
normal_with_duration = [n for n in normal_notes if 'duration' in n]
if normal_with_duration:
    print(f"\n普通音符 duration 样本(前5):")
    for n in normal_with_duration[:5]:
        print(f"  {n.get('lyric')}: duration={n.get('duration')}, length={n.get('length')}")