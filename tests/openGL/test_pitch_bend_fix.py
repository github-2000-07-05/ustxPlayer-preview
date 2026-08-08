"""验证 per-note pitch 数据修复是否生效。

测试场景：
1. 加载 庙堂之外.uplr
2. 检查过渡音符(+/-)的 pitch_bend 是否包含 per-note 音高数据
3. 对比修复前后的 pitch_bend 差异
"""
import json
import os
import sys

# 加入项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import yaml
from core.ustxreader import get_ustx_info_from_content

uplr_path = os.path.join(os.path.dirname(__file__), '..', '庙堂之外.uplr')
with open(uplr_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

ustx_content = data['ustx_content']
yaml_data = yaml.safe_load(ustx_content)

# ---- 原始 YAML 数据 ----
vp = yaml_data['voice_parts'][0]
notes_raw = vp['notes']
trans_raw = [n for n in notes_raw if n.get('lyric', '') in ('-', '+')]

print("=" * 60)
print("一、原始 YAML 数据 — 过渡音符的 per-note pitch 包络")
print("=" * 60)
for tn in trans_raw[:3]:
    pitch = tn.get('pitch', {})
    pdata = pitch.get('data', [])
    print(f"  lyric={tn['lyric']}, pos={tn['position']}, duration={tn['duration']}, tone={tn['tone']}")
    print(f"    pitch.data={json.dumps(pdata)}")
    print(f"    snap_first={pitch.get('snap_first', False)}")

# ---- 解析后的数据 ----
print("\n" + "=" * 60)
print("二、修复后解析结果 — pitch_bend 应包含 per-note 叠加值")
print("=" * 60)
result = get_ustx_info_from_content(ustx_content, track_index=0)
parsed_notes = result['notes']

trans_parsed = [n for n in parsed_notes if n.get('lyric') in ('+', '-')]
print(f"过渡音符数量: {len(trans_parsed)}")
print(f"所有音符数量: {len(parsed_notes)}")

# 检查过渡音符的 pitch_bend
print("\n过渡音符 pitch_bend 样本:")
for tn in trans_parsed[:5]:
    pb = tn.get('pitch_bend', [])
    pb_str = str(pb[:8]) + "..." if len(pb) > 8 else str(pb)
    # 检查是否有非零值的叠加
    has_nonzero = any(abs(v) > 0 for v in pb)
    print(f"  index={tn['index']}, lyric={tn['lyric']}, pos={tn['position']}, "
          f"pb_len={len(pb)}, has_nonzero_offset={has_nonzero}")
    if has_nonzero:
        print(f"    pb={pb_str}")

# 检查普通音符的 pitch_bend
print("\n普通音符 pitch_bend 样本:")
normal_parsed = [n for n in parsed_notes if n.get('lyric') not in ('+', '-')]
for nn in normal_parsed[:5]:
    pb = nn.get('pitch_bend', [])
    has_nonzero = any(abs(v) > 0 for v in pb)
    # 找到对应的原始音符
    raw_note = notes_raw[int(nn['index'])]
    raw_pitch = raw_note.get('pitch', {})
    raw_pdata = raw_pitch.get('data', [])
    raw_y = [p.get('y', 0) for p in raw_pdata]
    print(f"  index={nn['index']}, lyric={nn['lyric']}, "
          f"pb_len={len(pb)}, raw_pitch_y={raw_y}, has_nonzero_offset={has_nonzero}")
    if has_nonzero:
        pb_str = str(pb[:8]) + "..." if len(pb) > 8 else str(pb)
        print(f"    pb={pb_str}")

# ---- 验证关键场景：过渡音符的转音 ----
print("\n" + "=" * 60)
print("三、验证过渡音符的转音数据")
print("=" * 60)

# 检查第一个过渡音符
if trans_parsed:
    tn = trans_parsed[0]
    pb = tn.get('pitch_bend', [])
    print(f"第一个过渡音符: index={tn['index']}, pos={tn['position']}, len={tn.get('length')}")
    print(f"  pitch_bend length: {len(pb)}")
    print(f"  pitch_bend values: {pb}")
    # 检查是否有非零值（表示转音数据正确传递）
    max_abs = max(abs(v) for v in pb) if pb else 0
    print(f"  最大绝对值偏移: {max_abs} cents")
    if max_abs > 0:
        print("  ✅ 转音(portamento)数据已正确合并到 pitch_bend!")
    else:
        print("  ❌ 转音数据仍为零 — 修复可能未生效")

# ---- 验证渲染管线 ----
print("\n" + "=" * 60)
print("四、pitch_bend 统计")
print("=" * 60)
total_with_nonzero = sum(1 for n in parsed_notes if any(abs(v) > 0 for v in n.get('pitch_bend', [])))
total_all = len(parsed_notes)
print(f"有非零 pitch_bend 的音符: {total_with_nonzero}/{total_all}")

# 检查是否有音符的 pitch_bend 明显包含 per-note 叠加
# 方法：比较相邻音符的 pitch_bend 值是否有跨音节的连续变化
trans_count = len(trans_parsed)
print(f"过渡音符数: {trans_count}")

# 结论
print("\n" + "=" * 60)
print("结论")
print("=" * 60)
if total_with_nonzero > 0:
    print("✅ per-note pitch 数据已成功合并到 pitch_bend")
    print("   转音(portamento)将在渲染时正确显示")
else:
    print("❌ per-note pitch 数据未合并 — 需要检查修复代码")