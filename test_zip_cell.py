import json, sys, os, shutil, tempfile, wave, struct, math
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('s2st-fairseq.ipynb', encoding='utf-8'))
cells = nb['cells']
src23 = ''.join(cells[23]['source'])

# --- mô phỏng môi trường Kaggle ---
tmp = tempfile.mkdtemp(prefix='s2st_test_')
old_cwd = os.getcwd()
os.chdir(tmp)

def mkwav(path, dur=0.5, freq=440.0, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b''.join(struct.pack('<h', int(12000*math.sin(2*math.pi*freq*i/sr)))
                               for i in range(int(dur*sr))))

mkwav('results/s2ut_results/1.wav', 0.5, 440)
mkwav('results/s2ut_results/2.wav', 0.7, 550)
open('results/s2ut_results/generate-dev.txt', 'w').write('D-0\t0\t1 2 3\n')
open('results/s2ut_results/generate-dev.unit', 'w').write('1 2 3\n')
mkwav('results/ref_gt_units/1.wav', 0.5, 440)
mkwav('results/real_speech_resynthesis/1.wav', 0.5, 440)
os.makedirs('checkpoints/s2ut_model', exist_ok=True)
open('checkpoints/s2ut_model/train.log', 'w').write('epoch 1 ...\n')
open('checkpoints/s2ut_model/checkpoint_last.pt', 'wb').write(b'0' * 1024)

# --- chạy cell 23 (như trong notebook, scope global) ---
ns = {}
exec(compile(src23, '<cell23>', 'exec'), ns)
zip_path = ns['ZIP_PATH']
assert zip_path and os.path.isfile(zip_path), 'ZIP not created'

import zipfile
with zipfile.ZipFile(zip_path) as zf:
    names = sorted(zf.namelist())
    print('\nZIP contents:')
    for n in names:
        print('  ', n)
    assert any(n.startswith('model_output/') for n in names)
    assert any(n.startswith('reference_gt_units/') for n in names)
    assert any(n.startswith('real_speech_resynthesis/') for n in names)
    assert 'training/train.log' in names
    assert 'README.txt' in names
    assert not any(n.endswith('.pt') for n in names), 'checkpoint must be excluded by default'
    print('\nREADME preview:')
    print(zf.read('README.txt').decode('utf-8')[:700])

# --- test 2: include_checkpoints=True ---
zp2 = ns['package_results_zip'](include_checkpoints=True, output_root=os.path.join(tmp, 'out2'))
with zipfile.ZipFile(zp2) as zf:
    assert any(n.endswith('.pt') for n in zf.namelist()), 'checkpoint missing'
print('\n[TEST 2] include_checkpoints=True OK ->', zp2)

# --- test 3: không có folder kết quả nào -> thông báo rõ ---
shutil.rmtree('results')
shutil.rmtree('checkpoints')
os.chdir(tmp)
ns3 = {}
exec(compile(src23, '<cell23>', 'exec'), ns3)
print('\n[TEST 3] empty case handled, ZIP_PATH =', ns3['ZIP_PATH'])

os.chdir(old_cwd)
shutil.rmtree(tmp, ignore_errors=True)
print('\nALL TESTS PASSED')
