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
