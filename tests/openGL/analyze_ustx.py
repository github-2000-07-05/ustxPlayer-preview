"""分析 USTX 文件中的 pitd 曲线和过渡音符数据。"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import yaml

# 加载 uplr 文件
uplr_path = os.path.join(os.path.dirname(__file__), '..', '庙堂之外.uplr')
with open(uplr_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

ustx_content = data['ustx_content']
yaml_data = yaml.safe_load(ustx_content)

vp = yaml_data['voice_parts'][0]
notes = vp['notes']

# 统计歌词类型
lyric_types = {}
for n in notes:
    l = n.get('lyric', '')
    lyric_types[l] = lyric_types.get(l, 0) + 1
print('歌词类型统计:')
for k, v in sorted(lyric_types.items(), key=lambda x: -x[1])[:20]:
    print(f'  {repr(k)}: {v}个')

# 查找过渡音符
trans_notes = [n for n in notes if n.get('lyric', '') in ('-', '+')]
print(f'\n过渡音符(-/+) 数量: {len(trans_notes)}')
if trans_notes:
    tn = trans_notes[0]
    print(f'第一个过渡音符: pos={tn.get("position")}, len={tn.get("length")}, tone={tn.get("tone")}')

# 音符位置和音高范围
positions = [n.get('position', 0) for n in notes]
tones = [n.get('tone', 0) for n in notes]
print(f'\n音符位置范围: {min(positions)}~{max(positions)}')
print(f'音高范围: {min(tones)}~{max(tones)}')

# pitd 曲线
curves = vp.get('curves', [])
for c in curves:
    if isinstance(c, dict) and c.get('abbr') == 'pitd':
        xs = c['xs']
        ys = c['ys']
        print(f'\npitd 曲线: xs_len={len(xs)}, 范围 {min(xs)}~{max(xs)}')

        # 检查过渡音符附近的 pitd 数据
        if trans_notes:
            tn = trans_notes[0]
            tn_pos = tn['position']
            tn_len = tn.get('length', 0)
            print(f'\n第一个过渡音符: pos={tn_pos}, len={tn_len}, tone={tn.get("tone")}')
            nearby = [(x, y) for x, y in zip(xs, ys) if tn_pos <= x <= tn_pos + tn_len]
            print(f'  附近 pitd 数据点: {len(nearby)} 个')
            if nearby:
                print(f'  前 10 个: {nearby[:10]}')

# 使用 ustxreader 解析
from core.ustxreader import get_ustx_info_from_content
print('\n\n=== 使用 ustxreader 解析结果 ===')
result = get_ustx_info_from_content(ustx_content, track_index=0)
notes_parsed = result['notes']
print(f'解析到 {len(notes_parsed)} 个音符')

# 统计 pitch_bend 数据
pb_counts = {}
for n in notes_parsed:
    pb = n.get('pitch_bend', [])
    key = len(pb)
    pb_counts[key] = pb_counts.get(key, 0) + 1
print('pitch_bend 长度分布:')
for k in sorted(pb_counts.keys()):
    print(f'  len={k}: {pb_counts[k]} 个音符')

# 检查过渡音符的 pitch_bend
trans_parsed = [n for n in notes_parsed if n.get('index') in [str(tn['index']) for tn in trans_notes]]
print(f'\n过渡音符的 pitch_bend 数据:')
for tn in trans_parsed[:5]:
    pb = tn.get('pitch_bend', [])
    print(f'  index={tn["index"]}, lyric={tn.get("lyric")}, pos={tn.get("position")}, pb_len={len(pb)}, pb={pb[:10]}...' if len(pb) > 10 else f'  index={tn["index"]}, lyric={tn.get("lyric")}, pos={tn.get("position")}, pb_len={len(pb)}, pb={pb}')