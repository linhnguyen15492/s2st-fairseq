# -*- coding: utf-8 -*-
"""Patch s2st-fairseq.ipynb:
1. Cell 4  : exec_cmd() ghi log đầy đủ ra file (trước đó chỉ in rút gọn).
2. Cell 14 : train ghi model_dir/train.log (kết quả train đầy đủ: loss/lr/GPU).
3. Cell 22 : markdown mô tả đúng nội dung ZIP.
4. Cell 23 : FIX bug `dir()` (scope local) -> globals(); đóng gói đúng folder kết quả.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
NB = "s2st-fairseq.ipynb"

nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]


def get_src(i):
    return "".join(cells[i]["source"])


def set_src(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)


# ---------------------------------------------------------------- Cell 4
old_exec = '''def exec_cmd(cmd, desc="", show_head=25, show_tail=8):
    print(f"\u25b6\ufe0f {desc}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    res = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"\u274c TH\u1ea4T B\u1ea0I: {desc}")
        print("--- STDOUT ---")
        print(res.stdout)
        print("--- STDERR ---")
        print(res.stderr)
        raise RuntimeError(f"L\u1ec7nh th\u1ea5t b\u1ea1i v\u1edbi m\u00e3 l\u1ed7i {res.returncode}:\\n{res.stderr}")
    combined = (res.stdout or "") + "\\n" + (res.stderr or "")
'''

new_exec = '''def exec_cmd(cmd, desc="", show_head=25, show_tail=8, log_file=None):
    print(f"\u25b6\ufe0f {desc}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    res = subprocess.run(cmd, env=env, text=True, capture_output=True)
    combined = (res.stdout or "") + "\\n" + (res.stderr or "")
    # Ghi log ĐẦY \u0110\u1ee6 (kh\u00f4ng r\u00fat g\u1ecdn) \u0111\u1ec3 \u0111\u01b0a v\u00e0o ZIP k\u1ebft qu\u1ea3
    if log_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            with open(log_file, "w", encoding="utf-8") as lf:
                lf.write(combined)
            print(f"   \U0001f4dd \u0110\u00e3 ghi log \u0111\u1ea7y \u0111\u1ee7: {log_file}")
        except OSError as e:
            print(f"   \u26a0\ufe0f Kh\u00f4ng ghi \u0111\u01b0\u1ee7c log {log_file}: {e}")
    if res.returncode != 0:
        print(f"\u274c TH\u1ea4T B\u1ea0I: {desc}")
        print("--- STDOUT ---")
        print(res.stdout)
        print("--- STDERR ---")
        print(res.stderr)
        raise RuntimeError(f"L\u1ec7nh th\u1ea5t b\u1ea1i v\u1edbi m\u00e3 l\u1ed7i {res.returncode}:\\n{res.stderr}")
'''

assert old_exec in get_src(4), "Cell 4: exec_cmd not found"
set_src(4, get_src(4).replace(old_exec, new_exec, 1))
print("[1/4] Cell 4 - exec_cmd(log_file=...) OK")

# ---------------------------------------------------------------- Cell 14
old_train = '''        exec_cmd(cmd, f"Hu\u1ea5n luy\u1ec7n m\u00f4 h\u00ecnh S2UT ({max_update} updates)", show_head=15, show_tail=25)'''
new_train = '''        exec_cmd(
            cmd,
            f"Hu\u1ea5n luy\u1ec7n m\u00f4 h\u00ecnh S2UT ({max_update} updates)",
            show_head=15,
            show_tail=25,
            log_file=os.path.join(model_dir, "train.log"),  # log \u0111\u1ea7y \u0111\u1ee7 \u2192 \u0111\u01b0\u1ee7c \u0111\u00f3ng g\u00f3i v\u00e0o ZIP
        )'''
assert old_train in get_src(14), "Cell 14: train exec_cmd not found"
set_src(14, get_src(14).replace(old_train, new_train, 1))
print("[2/4] Cell 14 - train.log OK")

# ---------------------------------------------------------------- Cell 22 (markdown)
md_22 = '''---
### 5.3 \U0001f4e6 \u0110\u00f3ng g\u00f3i k\u1ebft qu\u1ea3 train + inference th\u00e0nh ZIP \u0111\u1ec3 t\u1ea3i v\u1ec1

Ch\u1ec9 \u0111\u00f3ng g\u00f3i **c\u00e1c th\u01b0 m\u1ee5c k\u1ebft qu\u1ea3**, KH\u00d4NG l\u1ea5y to\u00e0n b\u1ed9 `/kaggle/working`:

| Th\u01b0 m\u1ee5c trong ZIP | N\u1ed9i dung | Ngu\u1ed3n |
|---|---|---|
| `model_output/` | audio do m\u00f4 h\u00ecnh S2UT d\u1ecbch + `generate-dev.txt/.unit` | M\u1ee5c 5.1 |
| `reference_gt_units/` | audio t\u1eeb ground-truth units qua HiFi-GAN (\u0111\u1ed1i chi\u1ebfu vocoder) | M\u1ee5c 5.1 |
| `real_speech_resynthesis/` | audio resynthesis ti\u1ebfng ng\u01b0\u1eddi th\u1eadt (Analysis-by-Synthesis) | M\u1ee5c 5.2 |
| `training/` | `train.log` \u0111\u1ea7y \u0111\u1ee7 (loss, lr, GPU util, gradient) | M\u1ee5c 4 |
| `AUDIO_STATS.csv` | duration/peak/RMS t\u1eebng file \u2192 d\u00f2 mode-collapse nhanh | t\u1ef1 \u0111\u1ed9ng |
| `README.txt` | m\u00f4 t\u1ea3 t\u1eebng th\u01b0 m\u1ee5c + danh s\u00e1ch file | t\u1ef1 \u0111\u1ed9ng |

> \u26a0\ufe0f Checkpoint `.pt` (0.5\u20131 GB) **kh\u00f4ng** \u0111\u01b0\u1ee7c k\u00e8m theo \u2014 n\u1ebfu c\u1ea7n: `package_results_zip(include_checkpoints=True)`.

\U0001f4e5 Tr\u00ean Kaggle: m\u1edf tab **Output** (ho\u1eb7c n\u00fat **Download**) \u2192 t\u1ea3i `s2st_results_package.zip`.
'''
set_src(22, md_22)
print("[3/4] Cell 22 - markdown OK")

# ---------------------------------------------------------------- Cell 23 (code)
cell_23 = '''# -*- coding: utf-8 -*-
import zipfile
import glob
import os
from datetime import datetime


def _g(name, default=None):
    """L\u1ea5y bi\u1ebfn t\u1ea1i GLOBAL scope.

    L\u1ed7i c\u0169: d\u00f9ng `name in dir()` b\u00ean trong h\u00e0m \u2192 dir() ch\u1ec9 tr\u1ea3i\u1ec3n scope LOCAL,
    n\u00ean bi\u1ebfn RESULTS_PATH/REF_RESULTS/... kh\u00f4ng bao gi\u1ed1i \u0111\u01b0\u1ee7c t\u00ecm th\u1ea5y \u2192 b\u00e1o "Ch\u01b0a t\u00ecm th\u1ea5y th\u01b0 m\u1ee5c k\u1ebft qu\u1ea3".
    """
    return globals().get(name, default)


def _files_of(path):
    """Li\u1ec7t k\u00ea file k\u1ebft qu\u1ea3 trong 1 th\u01b0 m\u1ee5c (wav + units/log), ho\u1eb7c ch\u00ednh n\u00f3 n\u1ebfu l\u00e0 file."""
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.wav")))
    extras = sorted(
        f for f in glob.glob(os.path.join(path, "*.*"))
        if f.lower().endswith((".unit", ".txt", ".log", ".json", ".yaml")) and f not in files
    )
    return files + extras


def _audio_stats(wav_path):
    """Tr\u1ea3 v\u1ec1 (duration, peak, RMS) \u2014 \u0111\u1ec3 \u0111\u1ed1i chi\u1ebfu mode-collapse khi xem ZIP."""
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if not len(data):
            return (0.0, 0.0, 0.0)
        return (len(data) / sr, float(np.max(np.abs(data))), float(np.sqrt(np.mean(data ** 2))))
    except Exception:
        return None


def package_results_zip(include_checkpoints: bool = False, output_root: str = None):
    """\u0110\u00f3ng g\u00f3i K\u1ebeT QU\u1ea2 train + inference (KH\u00d4NG l\u1ea5y to\u00e0n b\u1ed9 working dir) v\u00e0o 1 ZIP."""
    cwd = os.getcwd()
    if output_root is None:
        # \u01afu ti\u00ean /kaggle/working (th\u01b0 m\u1ee5c duy nh\u1ea5t Kaggle cho t\u1ea3i); fallback cwd khi ch\u1ea1y local
        output_root = "/kaggle/working" if os.path.isdir("/kaggle/working") else cwd
    os.makedirs(output_root, exist_ok=True)
    zip_path = os.path.join(output_root, "s2st_results_package.zip")

    # ---- 1) K\u1ebft qu\u1ea3 inference (M\u1ee5c 5.1 / 5.2) ----
    packages = {}
    for arc_name, var_name, rel in (
        ("model_output", "RESULTS_PATH", "results/s2ut_results"),
        ("reference_gt_units", "REF_RESULTS", "results/ref_gt_units"),
        ("real_speech_resynthesis", "REAL_RESYNT_DIR", "results/real_speech_resynthesis"),
    ):
        path = _g(var_name) or os.path.join(cwd, *rel.split("/"))
        if os.path.isdir(path):
            packages[arc_name] = path

    # T\u1ef1 d\u00f3 th\u00eam m\u1ecdi results/* kh\u00e1c ch\u1ee9a wav (ph\u00f2ng khi \u0111\u1eb7t t\u00ean kh\u00e1c / th\u00eam b\u01b0\u1edbc m\u1edbi)
    known = {os.path.abspath(p) for p in packages.values()}
    results_root = os.path.join(cwd, "results")
    if os.path.isdir(results_root):
        for sub in sorted(glob.glob(os.path.join(results_root, "*"))):
            if os.path.isdir(sub) and os.path.abspath(sub) not in known \\
                    and glob.glob(os.path.join(sub, "*.wav")):
                packages["results_" + os.path.basename(sub)] = sub

    # ---- 2) K\u1ebft qu\u1ea3 train (M\u1ee5c 4): train.log (+ checkpoint n\u1ebfu y\u00eau c\u1ea7u) ----
    model_dir = _g("MODEL_DIR") or os.path.join(cwd, "checkpoints", "s2ut_model")
    train_log = os.path.join(model_dir, "train.log")
    checkpoints = sorted(glob.glob(os.path.join(model_dir, "*.pt"))) if os.path.isdir(model_dir) else []
    if os.path.isfile(train_log):
        packages["training"] = train_log
    if include_checkpoints:
        for c in checkpoints:
            packages["training_checkpoints/" + os.path.basename(c)] = c

    if not packages:
        print("\\u26a0\\ufe0f Ch\u01b0a t\u00ecm th\u1ea5y th\u01b0 m\u1ee5c k\u1ebft qu\u1ea3 n\u00e0o \u0111\u1ec3 \u0111\u00f3ng g\u00f3i.")
        print("   H\u00e3y ch\u1ea1y M\u1ee5c 4 (train) v\u00e0 M\u1ee5c 5.1/5.2 (inference) tr\u01b0\u1edbc. Ki\u1ec3m tra:")
        for cand in (
            os.path.join(cwd, "results", "s2ut_results"),
            os.path.join(cwd, "results", "ref_gt_units"),
            os.path.join(cwd, "results", "real_speech_resynthesis"),
            model_dir,
        ):
            state = "C\\u00d3 T\\u1ed4N T\\u1ea0I" if os.path.isdir(cand) else "KH\\u00d4NG T\\u1ed2N T\\u1ea0I"
            print(f"   \\u2022 {cand}  \\u2192 {state}")
        return None

    manifest, stats_rows, n_files = [], [], 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc_name, src in packages.items():
            for f in _files_of(src):
                if not os.path.isfile(f):
                    continue
                arc = f"{arc_name}/{os.path.basename(f)}"   # d\u00f9ng '/' \u0111\u1ec3 zip \u0111\u01b0\u1ee7c m\u1edf tr\u00ean m\u1ecdi n\u00ean t\u1ea3ng
                zf.write(f, arc)
                n_files += 1
                manifest.append(arc)
                if f.lower().endswith(".wav"):
                    st = _audio_stats(f)
                    if st:
                        stats_rows.append((arc,) + tuple(round(x, 4) for x in st))

        # Th\u1ed1ng k\u00ea audio (duration/peak/RMS) \\u2192 d\u00f2 nhanh 2 file gi\u1ed1ng h\u1ebft nhau (mode-collapse)
        if stats_rows:
            csv_lines = ["file,duration_sec,peak,rms"]
            csv_lines += [",".join(str(x) for x in row) for row in stats_rows]
            zf.writestr("AUDIO_STATS.csv", "\\n".join(csv_lines) + "\\n")

        # README m\u00f4 t\u1ea3 t\u1eebng th\u01b0 m\u1ee5c + danh s\u00e1ch checkpoint (kh\u00f4ng k\u00e8m weights)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if checkpoints:
            ckpt_lines = [
                f"   {os.path.basename(c)} ({os.path.getsize(c) / 1024 ** 2:.0f} MB)" for c in checkpoints
            ]
        else:
            ckpt_lines = ["   (ch\u01b0a c\\u00f3 \\u2014 ch\\u1ea1y M\\u1ee5c 4)"]
        readme = (
            "S2ST Fairseq \\u2014 K\\u1ebft qu\\u1ea3 train + inference\\n"
            f"Created: {ts}\\n\\n"
            "model_output/            = audio d\u1ecbch b\u1edfi m\u00f4 h\u00ecnh S2UT (M\u1ee5c 5.1)\\n"
            "reference_gt_units/      = audio t\u1eeb ground-truth units qua HiFi-GAN \\u2014 tham chi\u1ebfu vocoder (M\u1ee5c 5.1)\\n"
            "real_speech_resynthesis/ = audio resynthesis ti\u1ebfng ng\u01b0\u1eddi th\u1eadt (M\u1ee5c 5.2)\\n"
            "training/                = train.log \u0111\u1ea7y \u0111\u1ee7: loss, lr, GPU util, gradient (M\u1ee5c 4)\\n"
            "AUDIO_STATS.csv          = duration/peak/RMS t\u1eebng file wav \\u2192 d\u00f2 mode-collapse\\n\\n"
            "Checkpoint (.pt) KH\u00d4NG k\u00e8m v\u00e0o ZIP (0.5\\u20131 GB). \\u0110ang l\u00e0m vi\u1ec7c t\u1ea1i:\\n"
            + "\\n".join(ckpt_lines) + "\\n"
            "   (mu\u1ed1n k\u00e8m: package_results_zip(include_checkpoints=True))\\n\\n"
            f"Files ({n_files}):\\n" + "\\n".join(manifest) + "\\n"
        )
        zf.writestr("README.txt", readme)

    size_mb = os.path.getsize(zip_path) / 1024 ** 2
    print(f"\\u2705 \u0110\u00e3 \u0111\u00f3ng g\u00f3i {n_files} file k\u1ebft qu\\u1ea3 \\u2192 {zip_path}  ({size_mb:.2f} MB)")
    for arc_name, src in packages.items():
        print(f"   \\u2022 {arc_name:24s} \\u2192 {src}")
    # B\u00e1o c\u00e1c b\u01b0\u1edbc ch\u01b0a ch\u1ea1y (thi\u1ebfu wav/log)
    for name, rel, hint in (
        ("model_output", "results/s2ut_results", "M\u1ee5c 5.1"),
        ("reference_gt_units", "results/ref_gt_units", "M\u1ee5c 5.1"),
        ("real_speech_resynthesis", "results/real_speech_resynthesis", "M\u1ee5c 5.2"),
    ):
        if name not in packages:
            print(f"   \\u26a0\\ufe0f Thi\u1ebfu {name} (\\u2192 {rel}) \\u2014 ch\u1ea1y {hint} \u0111\u1ec3 b\u1ed5 sung v\u00e0o ZIP m\u1edbi.")
    if not include_checkpoints and checkpoints:
        print(f"   \\u2139\\ufe0f B\u1ecf qua {len(checkpoints)} checkpoint (.pt) cho ZIP nh\u1eb9 \\u2014 ch\u1ea1y l\u1ea1i "
              f"package_results_zip(include_checkpoints=True) n\u1ebfu c\u1ea7n.")
    print("\\U0001f4e5 Tr\u00ean Kaggle: m\u1edf tab **Output** (ho\u1eb7c n\u00fat **Download**) r\u01a1i t\u1ea3i file `s2st_results_package.zip`.")
    return zip_path


ZIP_PATH = package_results_zip()
'''

set_src(23, cell_23)

# compile check
compile(cell_23, "<cell23>", "exec")
compile(get_src(14), "<cell14>", "exec")
s4 = get_src(4)
# Cell 4 c\u00f3 magic IPython (!git clone...) n\u00ean ch\u1ec9 compile ph\u1ea7n h\u00e0m exec_cmd
compile(s4[s4.index("def exec_cmd"):s4.index("# Ki\u1ec3m tra import fairseq")], "<cell4.exec_cmd>", "exec")
print("[4/4] Cell 23 - code OK (compile pass)")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB} ({len(cells)} cells)")
