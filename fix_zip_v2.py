# -*- coding: utf-8 -*-
"""Fix v2 cho Cell 23 (s2st-fairseq.ipynb) — dùng vị trí (index) thay vì match chuỗi escape:
1. Sửa typo 'rơi tải' -> 'rồi tải'.
2. _audio_stats(): thêm fallback stdlib `wave` để AUDIO_STATS.csv luôn có (kể cả thiếu soundfile).
3. Chấp nhận peak/RMS = None trong stats row.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
NB = "s2st-fairseq.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
src = "".join(nb["cells"][23]["source"])
orig = src

# 1) typo
assert "Download**) rơi tải file" in src, "typo marker not found"
src = src.replace("Download**) rơi tải file", "Download**) rồi tải file", 1)

# 2) thay toàn bộ hàm _audio_stats (từ 'def _audio_stats' đến 'def package_results_zip')
start = src.index("def _audio_stats")
end = src.index("def package_results_zip")
new_fn = '''def _audio_stats(wav_path):
    """Trả về (duration, peak, RMS) — để đối chiếu mode-collapse khi xem ZIP."""
    try:  # ưu tiên soundfile + numpy (chạy trên Kaggle)
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if not len(data):
            return (0.0, 0.0, 0.0)
        return (len(data) / sr, float(np.max(np.abs(data))), float(np.sqrt(np.mean(data ** 2))))
    except Exception:
        pass
    try:  # Fallback stdlib `wave` — không cần soundfile/numpy (wav PCM 16-bit)
        import wave
        import struct
        with wave.open(wav_path, "rb") as w:
            sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(n)
        duration = n / sr if sr else 0.0
        if sw != 2:
            return (duration, None, None)
        samples = struct.unpack("<%dh" % (len(raw) // 2), raw)
        if ch > 1:
            samples = samples[::ch]
        if not samples:
            return (duration, 0.0, 0.0)
        peak = max(abs(s) for s in samples) / 32768.0
        rms = (sum((s / 32768.0) ** 2 for s in samples) / len(samples)) ** 0.5
        return (duration, peak, rms)
    except Exception:
        return None


'''
src = src[:start] + new_fn + src[end:]

# 3) chấp nhận None trong stats row
old_row = 'stats_rows.append((arc,) + tuple(round(x, 4) for x in st))'
new_row = 'stats_rows.append((arc,) + tuple("" if x is None else round(x, 4) for x in st))'
assert old_row in src, "stats_rows line not found"
src = src.replace(old_row, new_row, 1)

assert src != orig
compile(src, "<cell23>", "exec")
nb["cells"][23]["source"] = src.splitlines(keepends=True)
json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("OK: cell 23 updated + compile pass")
