import json, sys, ast
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open('s2st-fairseq.ipynb', encoding='utf-8'))
assert nb['nbformat'] == 4
assert isinstance(nb['cells'], list) and len(nb['cells']) == 26
for i, c in enumerate(nb['cells']):
    assert c['cell_type'] in ('code', 'markdown'), i
    assert isinstance(c['source'], list)
    assert 'metadata' in c and 'outputs' in c if c['cell_type'] == 'code' else True
print('JSON structure OK, nbformat=4, cells=26')

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

# kiểm tra nhanh các thay đổi kỳ vọng (index theo notebook 26 cell hiện tại)
s0 = ''.join(nb['cells'][0]['source'])
s4 = ''.join(nb['cells'][4]['source'])
s6 = ''.join(nb['cells'][6]['source'])
s10 = ''.join(nb['cells'][10]['source'])
s12 = ''.join(nb['cells'][12]['source'])
s16 = ''.join(nb['cells'][16]['source'])
s18 = ''.join(nb['cells'][18]['source'])
s20 = ''.join(nb['cells'][20]['source'])
s24 = ''.join(nb['cells'][24]['source'])
s25 = ''.join(nb['cells'][25]['source'])
checks = {
    'cell0 dữ liệu 32 clip': 'Tải 32 clip' in s0,
    'cell0 nhắc cổng kiểm tra Mục 3.2': 'cổng kiểm tra units' in s0,
    'cell6 k-means dùng URL chính thức km100/km.bin': 'textless_nlp/gslm/hubert/km100/km.bin' in s6,
    'cell6 KHÔNG còn URL 403 cũ': 'hubert_base_ls960_L6_km100.bin"' not in s6.replace('pretrained_models/hubert/hubert_base_ls960_L6_km100.bin', ''),
    'cell6 bỏ mirror yfyeung (không còn URL tải mirror)': 'huggingface.co/yfyeung' not in s6,
    'cell6 download_file có expected_sha256': 'expected_sha256: str = None' in s6,
    'cell6 có verify sha256': '_sha256' in s6 and 'EXPECTED_SHA256' in s6,
    'cell6 hash k-means chính thức': 'f14a3104615485381fc489701d6761c9abdbb0d43607cf55518a1e2891023fe' in s6,
    'cell4 exec_cmd có log_file': 'log_file=None' in s4,
    'cell4 ghi log đầy đủ': 'Đã ghi log đầy đủ' in s4,
    'cell10 manifest dùng natural sort': 'key=_natural_key' in s10,
    'cell10 có rename_pred_by_id': 'def rename_pred_by_id' in s10,
    'cell10 có read_units_file': 'def read_units_file' in s10,
    'cell12 có cổng round-trip': 'def verify_unit_vocoder_roundtrip' in s12,
    'cell12 ngưỡng 50% frame acc': 'min_frame_acc: float = 0.5' in s12,
    'cell12 raise khi sai label space': 'Vocoder đang hiểu sai label space' in s12,
    'cell12 chạy cổng trước train': 'ROUNDTRIP_FRAME_ACC' in s12,
    'cell16 train ghi train.log': 'log_file=os.path.join(model_dir, "train.log")' in s16,
    'cell18 có read_tsv_ids': 'def read_tsv_ids' in s18,
    'cell18 unit rỗng thì raise (bỏ fallback 0)': "tokens if tokens else" not in s18 and 'hypothesis RỖNG' in s18,
    'cell20 rename model output theo id': 'rename_pred_by_id(RESULTS_PATH, dev_ids)' in s20,
    'cell20 rename GT theo id': 'rename_pred_by_id(REF_RESULTS, gt_ids)' in s20,
    'cell24 markdown mô tả ZIP': 'train + inference' in s24,
    'cell25 dùng globals() (fix bug dir())': 'globals().get' in s25,
    'cell25 không còn dir() check': 'in dir()\n' not in s25.replace('`name in dir()`', ''),
    'cell25 có auto-quét results/*': 'results_' in s25,
    'cell25 có AUDIO_STATS.csv': 'AUDIO_STATS.csv' in s25,
    'cell25 fallback wave': 'import wave' in s25,
    'cell25 typo đã sửa': 'rồi tải file' in s25,
    'cell25 exclude checkpoint mặc định': 'include_checkpoints: bool = False' in s25,
    'cell25 zip separator /': 'f"{arc_name}/{os.path.basename(f)}"' in s25,
    'cell25 README ghi round-trip': 'ROUNDTRIP_FRAME_ACC' in s25,
}
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
assert all(checks.values())
print('\nALL CHECKS PASS')
