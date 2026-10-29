# CHANGELOG

Tất cả chỉnh sửa mã nguồn/notebook của dự án S2ST Fairseq đều được ghi lại tại đây.

---

## [2026-10-29] Dọn repo: xoá script/artefact chẩn đoán, chốt danh mục lỗi vào tài liệu

**File:** `.gitignore`, `README.md`, `NHAT_KY_DANH_GIA_CHAT_LUONG_AM_THANH.md` (thêm Mục 7), xoá 9 script, 5 thư mục chẩn đoán và 1 tài liệu trùng

Mục đích: repo không còn rối, nhưng **mọi lỗi vẫn tra được** — chi tiết số liệu giữ ở nhật ký Mục 6.7, danh mục tra cứu theo triệu chứng ở Mục 7.

- **Xoá script dùng một lần**: `create_notebook.py`, `apply_zip_fix.py`, `fix_zip_v2.py`, `test_zip_cell.py`, `run_s2st_exp.py` (đã áp dụng / đã bị notebook thay thế), `_patch_km.py`, `_proof_download.py`, `_nb_overview.py`, `_test_fairseq_import.py`.
- **Xoá script phân tích** (số liệu đã chốt trong nhật ký Mục 6.7): `analyze_package.py`, `analyze_mapping.py`, `analyze_stoi.py`, `stoi_vs_original.py`, `control_asr.py`, `asr_intelligibility.py`, `compare_kmeans.py`, `proof_ab_kmeans.py`.
- **Xoá artefact/venv tải lại được** (~2,3 GB): `_proofenv/`, `_proof_models/`, `_proof_run/`, `_orig_ljs/`, `_km/`.
- **Xoá bản trùng ở root**: `direct_s2st_discrete_units.md` — giống từng dòng (khác mỗi CRLF) với `fairseq/examples/speech_to_speech/docs/direct_s2st_discrete_units.md`, nên bản gốc trong `docs/` vẫn là nguồn duy nhất.
- **Giữ** `s2st_results_package/` (bằng chứng "trước khi sửa" cho Mục 6.7), `s2st_fairseq_demo.ipynb` (bản smoke-test cũ, đối tượng phân tích Mục 1–5) và `fairseq/examples/speech_to_speech/FLOW.md` (tài liệu luồng end-to-end, được nhật ký trích dẫn).
- **`.gitignore`**: thêm mục "S2ST Fairseq runtime artifacts" (`_proof*`, `_km/`, `_orig_ljs/`, `pretrained_models/`, `speech_dataset/`, `results/`, `checkpoints/`, `real_speech_sample/`, `s2st_results_package/`, `*.bak`, `.ipynb_checkpoints/`) để lần chạy sau không làm bẩn `git status`.
- **`README.md`**: bỏ quảng cáo "hỗ trợ mirror HuggingFace khi CDN 403" (chính là B1), trỏ notebook demo sang `s2st-fairseq.ipynb`, thêm mục tra cứu lỗi.
- **Nhật ký**: thêm **Mục 7 — Danh mục lỗi (B1–B7)** kèm quy trình chẩn đoán nhanh 5 bước (cổng round-trip → reference GT → ASR WER → chỉ tin metric khi tên file khớp).

---

## [2026-10-29] Chặn tận gốc 2 lỗi làm audio sai: cổng kiểm tra units ↔ vocoder + đặt tên file theo sample_id

**File:** `s2st-fairseq.ipynb` (24 → 26 cell), `NHAT_KY_DANH_GIA_CHAT_LUONG_AM_THANH.md` (Mục 6.7), `verify_nb.py`

Sau khi sửa k-means (mục dưới), rà lại `s2st_results_package/` phát hiện thêm 2 lỗi khiến mọi metric trong gói vô nghĩa, và thêm cơ chế chặn tự động:

- **P0 — Cổng kiểm tra mới ở Mục 3.2** (`verify_unit_vocoder_roundtrip`): round-trip `wav → units → wav → units` trên 1 clip LJSpeech, đo % frame khớp; `< 50%` thì `raise` với thông điệp chỉ đúng file k-means. Đo thực tế: `km100/km.bin` = **85.3 %** (ĐẠT), mirror `yfyeung` = **0.0 %** (bị chặn). Chạy trước Mục 4 nên không mất ~30 phút train.
- **P1 — Thứ tự file:** `create_wav_manifest` chuyển sang **natural sort** (`sample_0, sample_1, …, sample_10`), trước đây `sorted(glob)` cho thứ tự lexicographic lệch với `prep_s2ut_data.py` (thứ tự filesystem) → `model_output/7_pred.wav` và `reference_gt_units/7_pred.wav` là 2 câu khác nhau (hoán vị 0→36, 1→11, 2→27; kiểm chứng 40/40 bằng độ dài).
- **P1 — Đặt tên theo id:** thêm `rename_pred_by_id()`; wav kết quả đổi thành `<sample_id>_pred.wav` (model: id đọc từ `dev.tsv`; GT: id giữ từ tiền tố `dev.txt`). Hai thư mục ghép cặp được theo tên, không phụ thuộc thứ tự.
- **P2 — Bỏ fallback unit rỗng:** `extract_units_from_log` không ghi `'0'` thay hypothesis rỗng nữa mà `raise` (trước đây che lỗi model chưa hội tụ, vocoder vẫn "ra tiếng").
- **P2 — Trung thực khi báo cáo:** Mục 5.2 bỏ tuyên bố "MOS ~4.0", ghi rõ đây chỉ là kiểm chứng khâu units → wav; README trong ZIP ghi `%` round-trip.
- `verify_nb.py` cập nhật index cell (26 cell) + thêm 12 check cho cổng kiểm tra, rename, natural sort, raise-khi-rỗng.

Bằng chứng trong `NHAT_KY...` Mục 6.7: WER (faster-whisper `base.en`) của units trích bằng `km100/km.bin` = **4.2 % trung vị** (0/40 rỗng), mirror = **94.6 %** (2/40 rỗng), model_output trong gói = **90.2 %** (8/40 rỗng).

---

## [2026-10-29] FIX NGUYÊN NHÂN GỐC audio rè/mờ: dùng sai file k-means (Cell 6) — KHÔNG cần tăng epochs

**File:** `s2st-fairseq.ipynb` (Cell 6 — `MODEL_URLS` / `download_file`)

### Kết luận điều tra chất lượng âm thanh trong `s2st_results_package/`
Phân tích toàn diện (xem các script `analyze_*.py`, `*_asr*.json`, `compare_kmeans.py`) cho thấy **model S2UT đã hội tụ hoàn toàn**, việc tăng số vòng huấn luyện **không giải quyết được gì**:
- `train.log`: valid loss 8.873 → **0.002** (ppl 1) tại update 3000; loss train ~0.005, gnorm 0.14.
- `generate-dev.txt`: **39/40 câu khớp EXACT** ground-truth units; token accuracy **0.986**; length ratio 1.000. Identity map đã học được.
- `AUDIO_STATS.csv` + phân tích: **0 file trùng lặp** → mode-collapse đã bị loại bỏ (khác lần chạy trước).
- `reference_gt_units/` (GT units → HiFi-GAN, **không qua model S2UT**) cũng **nghèo/nhiễn y hệt**: Whisper WER vs LJSpeech gốc ~176%, STOI ~0.43 ≈ baseline xáo trộn (~0.45). Nghĩa là đường gọi vocoder "sạch" đã hỏng từ trước khi model dự đoán.

→ **Audio hỏng nằm ở tầng discrete units/vocoder, không phải ở mô hình S2UT.**

### Nguyên nhân gốc: sai file k-means (nhãn cụm bị hoán vị)
- Cell 6 tải k-means từ URL `.../textless_nlp/gslm/hubert/hubert_base_ls960_L6_km100.bin` → **HTTP 403**, sau đó **âm thầm** fall-back sang mirror `yfyeung/kmeans-librispeech-hubert-base` (`hubert-base-layer-6.kmeans.100.model`).
- So trực tiếp 2 file k-means (`compare_kmeans.py`, `_km/`): cùng shape (100×768), nhưng **nhãn cụm bị hoán vị** — cosine cùng index chỉ **0.199**, chỉ **~1% cụm** nằm đúng vị trí; cosine tốt nhất mọi cặp **0.918** (nghĩa là *cùng* cụm âm học nhưng *khác* nhãn).
- Vocoder HiFi-GAN `hubert_base_100_lj` (config `num_embeddings 100`, `code_hop_size 320`, `sampling_rate 16000`) được huấn luyện kèm **bản k-means chính thức**. feeding nhãn sai → embedding sai theo hoán vị → sinh âm thanh rè/mờ, mất lời.

### URL chính thức hợp lệ (đã kiểm chứng HTTP 200, 308469 bytes)
`https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin`
(ghi trong `fairseq/examples/textless_nlp/pgslm/README.md`). SHA-256:
`f14a3104615485381fc489701d6761c9abdbb0d43607cf55518a1e2891023fe`

### Thay đổi trong Cell 6
- **Đổi URL k-means** sang bản chính thức `km100/km.bin`; **bỏ hẳn** fallback `yfyeung` (danh sách fallback rỗng).
- **`download_file()` thêm kiểm tra SHA-256** (`expected_sha256`): file sai bị **xoá ngay**, không dùng; nếu không khớp thì thử URL tiếp theo/không im lặng dùng bản sai.
- Gắn `EXPECTED_SHA256` cho file k-means; các model khác (vocoder, HuBERT) giữ nguyên (không ép hash để tránh vỡ nếu server đổi đuôi).
- Thông báo lỗi cuối rõ ràng: hướng dẫn bật Internet trên Kaggle, và **cấm** thay mirror k-means khác (nhãn hoán vị → audio rè).

### Bằng chứng quyết định (thí nghiệm A/B, chạy local — xem `proof_ab_kmeans.py` / `_proof_run/proof_result.json`)
Cùng 40 clip LJSpeech gốc → trích HuBERT layer-6 units bằng **(A) k-means chính thức** vs **(B) k-means mirror** → vocode cả hai bằng **cùng** HiFi-GAN → đo Whisper WER vs audio gốc:
- **(A) Chính thức: WER thấp (audio rõ lời)** vs **(B) Mirror: WER cao (audio rè/mờ)**.
- Unit map chính thức→mirror là **hoán vị nhất quán** (mỗi unit official luôn đổi thành 1 unit mirror xác định) → khẳng định mirror chỉ *đổi nhãn*, làm sai hoàn toàn vocoder.

### Quyết định: KHÔNG tăng số vòng huấn luyện
- Model đã hội tụ (valid loss 0.002, 39/40 exact). Thêm epochs chỉ tốn thời gian, không đổi chất lượng.
- **Việc cần làm là sửa file k-means rồi TRAIN LẠI** (vì units sinh ra trước đó dùng sai codebook → đầu vào train cũng sai). `max_update=3000` giữ nguyên (~24 phút trên T4).

### Hướng dẫn chạy lại trên Kaggle (thứ tự)
1. Bật **Internet ON**.
2. Chạy lại từ **Mục 2** (tải models — sẽ lấy k-means chính thức + verify hash).
3. **Mục 3** (dữ liệu) → **Mục 3.1** (trích units bằng codebook đúng) → **Mục 3.2** (cổng kiểm tra) → **Mục 3.3** (chuẩn bị S2UT).
4. **Mục 4** train lại với `max_update=3000` (KHÔNG tăng thêm).
5. **Mục 5.1** (suy luận) → **Mục 5.2** (resynthesis) → **Mục 5.3** (ZIP).
6. Nghe `model_output/` và `reference_gt_units/`: cả hai giờ phải **rõ lời** (vocoder nay nhận đúng nhãn).

---

## [2026-10-28] Nhóm B — Sửa pipeline huấn luyện để identity map hội tụ (fix audio "rất khó nghe")

**File:** `s2st-fairseq.ipynb`
**Nguyên nhân:** Output audio khó nghe do **mode-collapse** — mô hình S2UT (~100M tham số) train 300 updates trên 8 clip với dropout/label-smoothing đã sinh ra output **gần như giống hệt nhau** cho mọi input (bằng chứng: file `[0]==[3]`, `[1]==[2]` cùng duration/peak/RMS trong lần chạy trước). Beam search collapse về cùng chuỗi unit "generic" → vocoder HiFi-GAN sinh ra tiếng rè kim loại không ra lời.

### Thay đổi

#### Cell 8 — Chuẩn bị dữ liệu
- **8 → 32 clip LJSpeech thật** (`num_samples=32`). Cung cấp **40 ứng viên** `LJ001-0001..LJ001-0040` (downloader tự bỏ qua file lỗi), để đảm bảo có ≥32 clip thành công.
- Lý do: ít clip → model dễ mode-collapse; 32 clip giúp identity map hội tụ tổng quát hơn.

#### Cell 14 — Hàm huấn luyện `train_s2ut_model`
- `max_update`: **300 → 3000** (đủ để ~100M tham số học identity map từ random init trên T4; ~30–45 phút).
- `num_workers`: **0 → 4** (fix bottleneck dataloader trước đây khiến GPU util chỉ 0–12%).
- `--label-smoothing`: **0.2 → 0** (bỏ regularization để model *ghi nhớ* 8/32 clip).
- `--dropout`, `--attention-dropout`, `--relu-dropout`: **0.1 → 0** (tương tự, cho phép overfit).
- `--warmup-updates`: **50 → 100** (lr scheduler hợp lý hơn với 3000 updates).
- Gọi `train_s2ut_model(..., max_update=3000)`.
- `exec_cmd(...)`: `show_head 30→15`, `show_tail 10→25` để thấy rõ loss cuối + valid loss trong log.

#### Cell 19 — Ô phát âm thanh `play_generated_audio`
- **Thêm bộ phát hiện mode-collapse:** đọc thống kê (duration/peak/RMS) của *toàn bộ* file output; nếu 2 file có bộ ba này gần như y hệt nhau → in cảnh báo `PHÁT HIỆN MODE-COLLAPSE` kèm tên 2 file trùng + gợi ý xử lý.
- Giúp tự kiểm chứng model đã học thật chưa sau khi tăng training.

#### Cell 0 & Cell 7 — Markdown
- Cập nhật mô tả: "8 clip → 32 clip", "~300 updates → ~3000 updates (dropout=0, label-smoothing=0)".

### Kết quả mong đợi
- Output của model không còn trùng lặp lẫn nhau (bộ phát hiện mode-collapse không báo).
- Độ dài audio sát bản gốc; âm thanh có nội dung/nghe được từng từ (vì overfit identity map src=tgt trên speech thật).
- Phần **Unit Resynthesis (Mục 5.2)** giữ nguyên làm tham chiếu cho âm thanh trong trẻo.

### Lưu ý
- Các ô vẫn còn **output lịch sử** của lần chạy trước (log với `max_update=300`, `dropout=0.1`). Đây chỉ là output cũ, sẽ bị ghi đè khi chạy lại trên Kaggle (chạy ở chế độ **GPU T4 + Internet ON**).
- Thời gian train dài hơn (~30–45 phút) — đảm bảo session Kaggle không bị timeout giữa chừng.

---

## [2026-10-28] Bổ sung cell đóng gói kết quả âm thanh ZIP (Mục 5.3)

**File:** `s2st-fairseq.ipynb` (thêm 2 cell mới ở cuối: markdown tiêu đề + code)

### Thay đổi
- **Cell markdown 5.3:** tiêu đề "📦 Đóng gói kết quả âm thanh thành ZIP để tải về".
- **Cell code `package_results_zip()`:**
  - Gom toàn bộ kết quả âm thanh vào1 file `s2st_results_package.zip` đặt tại `/kaggle/working` (thư mục duy nhất Kaggle cho tải về):
    - `model_output/` — audio model S2UT dịch (`RESULTS_PATH`)
    - `reference_gt_units/` — audio từ ground-truth units qua HiFi-GAN (`REF_RESULTS`)
    - `real_speech_resynthesis/` — audio resynthesis Analysis-by-Synthesis (`REAL_RESYNT_DIR`)
  - Kèm file units/log (`.unit`, `.txt`) của từng thư mục để tiện chẩn đoán.
  - Tự ghi `README.txt` trong ZIP mô tả từng thư mục + danh sách file + thời điểm tạo.
  - Chỉ include thư mục thực sự tồn tại; báo rõ dung lượng ZIP và hướng dẫn tải trên Kaggle (tab Output / nút Download).

### Mục đích
- Cho phép người dùng tải toàn bộ audio kết quả về một cách, sau đó cung cấp lại để **kiểm tra/chẩn đoán chất lượng** (so sánh model_output vs reference vs resynthesis).

### Kết quả mong đợi
- Trên Kaggle tab **Output** xuất hiện `s2st_results_package.zip` (~vài MB). Giải nén ra 3 thư mục con + README.txt.

---

## [2026-10-28] FIX lỗi "⚠️ Chưa tìm thấy thư mục kết quả nào để đóng gói" + mở rộng phạm vi ZIP (train + inference)

**File:** `s2st-fairseq.ipynb` (Cell 4, 14, 22, 23)

### Nguyên nhân lỗi
Cell 23 dùng `if "RESULTS_PATH" in dir()` **bên trong hàm `package_results_zip()`**. Trong Python, `dir()` không có đối số chỉ trả về **scope local** của hàm → các biến toàn cục (`RESULTS_PATH`, `REF_RESULTS`, `REAL_RESYNT_DIR`) **không bao giờ được tìm thấy** → `packages` rỗng → luôn in:
`⚠️ Chưa tìm thấy thư mục kết quả nào để đóng gói. Hãy chạy các mục 5.1/5.2 trước.`
(Lỗi này xảy ra **dù 5.1/5.2 đã chạy thành công**.)

### Thay đổi

#### Cell 4 — `exec_cmd()`
- Thêm tham số **`log_file=None`**: ghi **toàn bộ** stdout+stderr ra file (trước đó chỉ in rút gọn `show_head/show_tail` → log train đầy đủ bị mất).
- Ghi log **trước** khi raise lỗi → vẫn có log khi lệnh thất bại.

#### Cell 14 — `train_s2ut_model()`
- Gọi `exec_cmd(..., log_file=os.path.join(model_dir, "train.log"))` → kết quả train (loss, lr, GPU util, gradient) được lưu thành **`checkpoints/s2ut_model/train.log`** để đóng gói vào ZIP.

#### Cell 22 — Markdown 5.3
- Cập nhật bảng mô tả đúng nội dung ZIP (thư mục, nguồn ô, mục đích) + ghi rõ checkpoint `.pt` không kèm theo.

#### Cell 23 — `package_results_zip()` (viết lại)
- **Fix root cause:** dùng helper `_g(name)` = `globals().get(name, ...)` thay cho `in dir()`.
- **Chỉ đóng gói folder kết quả** (KHÔNG lấy toàn bộ `/kaggle/working`), kèm fallback đường dẫn chuẩn khi biến chưa được định nghĩa:
  - `model_output/` ← `results/s2ut_results` (Mục 5.1)
  - `reference_gt_units/` ← `results/ref_gt_units` (Mục 5.1)
  - `real_speech_resynthesis/` ← `results/real_speech_resynthesis` (Mục 5.2)
  - `training/` ← `checkpoints/s2ut_model/train.log` (Mục 4)
- **Tự quét thêm** mọi `results/*` khác chứa `.wav` (phòng đổi tên/thêm bước).
- **`AUDIO_STATS.csv`** trong ZIP: `file,duration_sec,peak,rms` từng file wav → dò mode-collapse ngay cả khi không mở notebook. `_audio_stats()` có **fallback `wave` stdlib** nếu thiếu soundfile/numpy.
- **`README.txt`** mô tả từng thư mục + danh sách file + danh sách checkpoint đang có (chỉ ghi tên/kích thước, **không** kèm weights).
- **Checkpoint `.pt` không mặc định kèm** (0.5–1 GB) — opt-in bằng `package_results_zip(include_checkpoints=True)` (đã test hoạt động).
- Kiểm tra ZIP dùng dấu phân cách **`/`** (không phải `os.path.join`) để file giải nén đúng trên mọi OS.
- Trường hợp chưa chạy 4/5.1/5.2: in **danh sách đường dẫn + CÓ/KHÔNG TỒN TẠI** từng thư mục để biết đúng ô nào thiếu, thay vì chỉ 1 câu chung chung.
- Sửa typo tiếng Việt trong thông báo: "rơi tải file" → **"rồi tải file"**.

### Kiểm chứng (chạy local)
- `test_zip_cell.py`: 3 test PASS — (1) ZIP đủ 4 nhóm file + README + CSV, không có `.pt`; (2) `include_checkpoints=True` có `.pt`; (3) không có kết quả → báo đúng đường dẫn thiếu.
- `verify_nb.py`: nbformat=4 hợp lệ, 24 cell, không có lỗi syntax, 12/12 check PASS.

### Kết quả mong đợi trên Kaggle
- Chạy hết Mục 4 → 5.1 → 5.2 → 5.3: in `✅ Đã đóng gói N file kết quả → /kaggle/working/s2st_results_package.zip (x.xx MB)` kèm danh sách thư mục đã gom.

---
