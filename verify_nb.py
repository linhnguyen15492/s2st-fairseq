import json, sys, ast
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open('s2st-fairseq.ipynb', encoding='utf-8'))
assert nb['nbformat'] == 4
assert isinstance(nb['cells'], list) and len(nb['cells']) == 24
for i, c in enumerate(nb['cells']):
    assert c['cell_type'] in ('code', 'markdown'), i
    assert isinstance(c['source'], list)
    assert 'metadata' in c and 'outputs' in c if c['cell_type'] == 'code' else True
print('JSON structure OK, nbformat=4, cells=24')

# compile mọi code cell không có magic (!, %, ?)
bad = []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    if any(l.lstrip().startswith(('!', '%', '?')) for l in src.splitlines()):
        continue
    try:
        ast.parse(src)
    except SyntaxError as e:
        bad.append((i, e))
print('syntax errors (non-magic cells):', bad if bad else 'NONE')

# kiểm tra nhanh các thay đổi kỳ vọng
s4 = ''.join(nb['cells'][4]['source'])
s14 = ''.join(nb['cells'][14]['source'])
s22 = ''.join(nb['cells'][22]['source'])
s23 = ''.join(nb['cells'][23]['source'])
checks = {
    'cell4 exec_cmd có log_file': 'log_file=None' in s4,
    'cell4 ghi log đầy đủ': 'Đã ghi log đầy đủ' in s4,
    'cell14 train ghi train.log': 'log_file=os.path.join(model_dir, "train.log")' in s14,
    'cell22 markdown mô tả ZIP': 'train + inference' in s22,
    'cell23 dùng globals() (fix bug dir())': 'globals().get' in s23,
    'cell23 không còn dir() check': 'in dir()\n' not in s23.replace('`name in dir()`', ''),
    'cell23 có auto-quét results/*': 'results_' in s23,
    'cell23 có AUDIO_STATS.csv': 'AUDIO_STATS.csv' in s23,
    'cell23 fallback wave': 'import wave' in s23,
    'cell23 typo đã sửa': 'rồi tải file' in s23,
    'cell23 exclude checkpoint mặc định': 'include_checkpoints: bool = False' in s23,
    'cell23 zip separator /': 'f"{arc_name}/{os.path.basename(f)}"' in s23,
}
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
assert all(checks.values())
print('\nALL CHECKS PASS')
