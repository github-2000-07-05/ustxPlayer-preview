"""分析 USTX 文件的 per-note pitch 数据与 pitd 的差异。
检查 ustxreader 是否遗漏了 per-note pitch 数据。
"""
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

# 统计有 pitch 数据的音符
with_pitch = [n for n in notes if 'pitch' in n]
print(f"有 pitch 数据的音符: {len(with_pitch)}/{len(notes)}")

# 统计过渡音符的 pitch 数据
trans_notes = [n for n in notes if n.get('lyric', '') in ('-', '+')]
trans_with_pitch = [n for n in trans_notes if 'pitch' in n]
print(f"过渡音符有 pitch 数据: {len(trans_with_pitch)}/{len(trans_notes)}")

# 统计普通音符的 pitch 数据
normal_notes = [n for n in notes if n.get('lyric', '') not in ('-', '+')]
normal_with_pitch = [n for n in normal_notes if 'pitch' in n]
print(f"普通音符有 pitch 数据: {len(normal_with_pitch)}/{len(normal_notes)}")

# 查看 pitch 数据样本
if trans_with_pitch:
    tn = trans_with_pitch[0]
    print(f"\n过渡音符 pitch 数据样本:")
    print(f"  lyric={tn['lyric']}, pos={tn['position']}, duration={tn['duration']}, tone={tn['tone']}")
    print(f"  pitch={json.dumps(tn['pitch'], indent=4)}")

if normal_with_pitch:
    nn = normal_with_pitch[0]
    print(f"\n普通音符 pitch 数据样本:")
    print(f"  lyric={nn['lyric']}, pos={nn['position']}, duration={nn['duration']}, tone={nn['tone']}")
    print(f"  pitch={json.dumps(nn['pitch'], indent=4)}")

# 检查 ustxreader 中是否提取了 per-note pitch 数据
from core.ustxreader import get_ustx_info_from_content
result = get_ustx_info_from_content(ustx_content, track_index=0)
parsed_notes = result['notes']

# 查看第一个过渡音符的解析结果
trans_parsed = [n for n in parsed_notes if n.get('lyric') in ('+', '-')]
if trans_parsed:
    print(f"\n\nustxreader 解析的过渡音符 pitch_bend 样本:")
    for tn in trans_parsed[:3]:
        pb = tn.get('pitch_bend', [])
        print(f"  index={tn['index']}, lyric={tn['lyric']}, pos={tn['position']}, len={tn.get('length')}, pb_len={len(pb)}, pb_first5={pb[:5]}")

# 结论
print(f"\n\n{'='*60}")
print(f"BUG 分析: ustxreader 只提取了 voice_part 级别的 pitd 曲线")
print(f"但忽略了 per-note 级别的 pitch 数据 (note.pitch.data)")
print(f"这会导致 per-note 的 pitch 过渡 (转音) 丢失")
print(f"{'='*60}")