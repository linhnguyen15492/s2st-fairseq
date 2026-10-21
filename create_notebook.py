"""Generate / sync s2st_fairseq_demo.ipynb.

Lưu ý: nguồn sự thật hiện tại là file s2st_fairseq_demo.ipynb (đã fix các bug
phát hiện khi review + chạy experiment). Script này chỉ re-dump notebook đó
để giữ định dạng JSON ổn định. Nếu chỉnh notebook, chạy lại script này.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT / "s2st_fairseq_demo.ipynb"

notebook = json.loads(NB.read_text(encoding="utf-8"))
NB.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Synced {NB} ({len(notebook['cells'])} cells)")
