# CHANGELOG

Tất cả chỉnh sửa mã nguồn/notebook của dự án S2ST Fairseq đều được ghi lại tại đây.

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
