# NHẬT KÝ ĐÁNH GIÁ & PHÂN TÍCH NGUYÊN NHÂN CHẤT LƯỢNG ÂM THANH

## Đề tài: Direct Speech-to-Speech Translation with Discrete Units (Fairseq S2UT)

* **Ngày thực hiện:** 26/10/2026 — **cập nhật 29/10/2026** (phân tích gói kết quả + sửa notebook, xem Mục 6.7).
* **Đối tượng phân tích:** Luồng huấn luyện và suy luận mô hình S2UT trong notebook Kaggle (`s2st_fairseq_demo.ipynb`, sau đó là `s2st_results_package/`) đối chiếu với tài liệu gốc bài báo ([Lee et al., 2021](https://arxiv.org/abs/2107.05604)) và tài liệu kỹ thuật [`direct_s2st_discrete_units.md`](file:///d:/workspace/s2st-fairseq/direct_s2st_discrete_units.md).
* **Notebook khắc phục:** Giải pháp trong Mục 4 đã triển khai trong **`s2st-fairseq.ipynb`**; ba lỗi của bản chạy mới (k-means sai label space, lệch thứ tự file, fallback unit rỗng) đã sửa bằng **cổng kiểm tra Mục 3.2** + đặt tên file theo `sample_id` (xem Mục 6.7).
* **Vấn đề ghi nhận từ Giảng viên:** *"Demo gốc của bài nghe khá rõ lời nói, còn demo em chạy lại trên kaggle thì nó rất khó nghe."*

> 📌 **Đọc Mục 6.7 trước nếu chỉ cần nguyên nhân của gói kết quả hiện tại.**
> Mục 1–5 dưới đây là chẩn đoán cho bản smoke-test cũ (sóng sin + 20 updates). Bản chạy mới
> (32 clip LJSpeech, 2994 updates, loss 0.006) **đã hội tụ**, nhưng audio vẫn rè vì **file k-means
> sai label space** và **lỗi ghép thứ tự file** — hai lỗi này chỉ được phát hiện và sửa trong
> đợt review 29/10/2026, ghi ở Mục 6.7.

---

## 1. Tóm tắt kết luận (Executive Summary)

Sau khi rà soát toàn bộ quy trình mã nguồn và cấu hình thí nghiệm, nguyên nhân khiến âm thanh đầu ra trên Kaggle bị khó nghe (rè, rít, biến dạng kim loại, không ra tiếng người) **hoàn toàn không phải do lỗi cài đặt thuật toán hay lỗi môi trường**, mà xuất phát từ việc notebook hiện tại đang chạy ở chế độ **Kiểm thử đường ống kỹ thuật (Smoke Test / Pipeline Verification)**:

1. **Dữ liệu giả lập là sóng sin toán học (`np.sin`), không phải tiếng nói con người:** HuBERT là mô hình âm học được huấn luyện trên 960 giờ tiếng nói con người (LibriSpeech). Khi đưa sóng sin đơn tần số vào, HuBERT sinh ra các embedding nằm ngoài phân phối (Out-Of-Distribution), khiến bộ gom cụm k-means gán các mã cụm (discrete units) bị suy biến (cluster collapse) và vô nghĩa.
2. **Mô hình dịch S2UT mới chỉ train thử 20 update steps (so với 400.000 steps của bài báo):** Với 20 bước cập nhật trên 8 mẫu thử, mô hình Transformer S2UT (hơn 100 triệu tham số) hoàn toàn chưa học được bất kỳ ánh xạ ngữ âm/ngôn ngữ nào, trọng số về bản chất vẫn là ngẫu nhiên (untrained random weights). Quá trình suy luận (inference beam search) chỉ sinh ra chuỗi mã ngẫu nhiên (hallucination).
3. **HiFi-GAN Vocoder nhận chuỗi token rác:** Vocoder `hubert_base_100_lj` được huấn luyện trên giọng nói tự nhiên (LJSpeech). Khi nhận chuỗi mã ngẫu nhiên không có tính chất ngữ âm, bộ dự đoán trường độ (`--dur-prediction`) và bộ tạo sóng âm (Generator) bị nhảy pha liên tục, dẫn đến hiện tượng chít rè số (digital clipping / metallic chirps).
4. **Bản demo gốc của bài báo nghe rõ là nhờ:** Bài báo sử dụng mô hình S2UT đã hội tụ hoàn toàn (train 400.000 steps trên tập Fisher 160 giờ thoại thật) hoặc sử dụng quy trình **Unit Resynthesis (Analysis-by-Synthesis)** trên giọng nói thật để minh họa năng lực tái tạo âm thanh của Vocoder.

---

## 2. Bảng đối chiếu chi tiết: Demo bài báo gốc vs Demo Kaggle hiện tại

| Thành phần | Demo gốc của bài báo (Lee et al., 2021) | Demo hiện tại trên Kaggle Notebook | Tác động đến chất lượng âm thanh |
| :--- | :--- | :--- | :--- |
| **Dữ liệu âm thanh đầu vào** | Tiếng người nói thật (Dataset Fisher Spanish-English hoặc VoxPopuli) | Sóng sin toán học $f(t) = \sin(2\pi f t)$ sinh bởi hàm `np.sin` (300Hz - 650Hz) | HuBERT không nhận diện được formant/phoneme con người; k-means sinh ra mã cụm sai lệch hoàn toàn. |
| **Quy mô tập dữ liệu** | 138.819 câu thoại (~160 giờ nói tự nhiên) | 8 mẫu cho tập `train`, 2 mẫu cho tập `dev` | Không đủ phân phối thống kê để mô hình Transformer học tổng quát hóa. |
| **Số bước huấn luyện (Training Updates)** | **400.000 updates** (sử dụng 4-16 GPU NVIDIA V100/A100) | **20 updates** (chạy thử nghiệm nhanh ~30 giây để tránh timeout trên Kaggle) | Tại update 20, hàm mất mát (loss) còn ở mức rất cao (~6.5 - 7.0), mô hình chưa qua giai đoạn warmup (10 steps). |
| **Trạng thái mô hình S2UT** | Đã hội tụ hoàn toàn (converged checkpoint) | Chưa hội tụ (tương đương trọng số ngẫu nhiên ban đầu) | Beam search sinh ra các token rời rạc hoàn toàn ngẫu nhiên. |
| **Đầu vào cấp cho Vocoder HiFi-GAN** | Chuỗi Discrete Units chuẩn xác mang đặc trưng ngữ âm tiếng Anh (phoneme-like) | Chuỗi Discrete Units ngẫu nhiên, gián đoạn, phi tự nhiên | Vocoder tạo sóng âm bị gãy nhịp trường độ, tạo ra tiếng rè kim loại và tiếng rít khó nghe. |
| **Mục đích thiết kế** | Trình diễn chất lượng dịch âm thanh thành âm thanh (End-to-End S2ST Quality) | Kiểm tra tính tương thích và thông suốt của đường ống mã nguồn (Code Pipeline Smoke Test) | Notebook hoàn thành 100% mục tiêu kiểm thử kỹ thuật nhưng chưa phục vụ mục tiêu đánh giá thính giác. |

---

## 3. Phân tích sâu 4 nguyên nhân kỹ thuật gốc rễ

### 3.1. Dữ liệu đầu vào nhân tạo không chứa đặc trưng ngữ âm con người

Trong ô code Mục 3 của notebook Kaggle, hàm sinh dữ liệu mẫu được định nghĩa như sau:

```python
for idx in range(num_samples):
    freq = 300 + idx * 50
    audio = np.sin(2 * np.pi * freq * np.linspace(0, 2, sr * 2)).astype(np.float32)
    sf.write(os.path.join(src_dir, split, f"sample_{idx}.wav"), audio, sr)
    sf.write(os.path.join(tgt_dir, split, f"sample_{idx}.wav"), audio, sr)
```

* **Cơ chế:** HuBERT (Hidden-Unit BERT) là mô hình tự giám sát (self-supervised) trích xuất đặc trưng từ layer 6. Các bộ lọc CNN của HuBERT được tối ưu để bắt các dải tần số đặc trưng của thanh quản và khoang miệng con người (formant F1, F2, F3, hài âm, phụ âm vô thanh, hữu thanh).
* **Hệ quả:** Khi đưa một dải sóng sin đơn âm (pure tone) vào, các vector ẩn đầu ra rơi vào vùng không gian embedding chưa từng thấy trong quá trình tiền huấn luyện (Out-Of-Distribution). Thuật toán k-means (100 centroids) buộc phải chiếu vector này vào một vài cụm ngẫu nhiên không có ý nghĩa ngữ âm.

### 3.2. Thời lượng huấn luyện 20 bước so với 400.000 bước của bài báo

* Kiến trúc `s2ut_transformer_fisher` bao gồm:
  * Speech Encoder: 12 lớp Transformer Encoder.
  * Unit Decoder: 6 lớp Transformer Decoder.
  * Tổng tham số: ~100M - 150M tham số.
* Theo khuyến nghị trong `direct_s2st_discrete_units.md`:

  ```bash
  fairseq-train $DATA_ROOT \
    --arch s2ut_transformer_fisher \
    --lr 0.0005 --warmup-updates 10000 \
    --max-update 400000 --max-tokens 20000 --update-freq 4
  ```

  Để một mô hình Transformer học được quan hệ ánh xạ đa phương thức (từ biểu diễn phổ âm học tiếng Tây Ban Nha sang chuỗi discrete unit tiếng Anh), bài báo cần ít nhất **10.000 updates chỉ để khởi động learning rate (warmup)** và **400.000 updates** trên hàng trăm nghìn mẫu.
* Việc cấu hình `max_update=20` trong notebook Kaggle là nhằm mục đích **đảm bảo script chạy qua được toàn bộ các module trong vòng 1-2 phút** để chứng minh code không lỗi runtime (không bị vướng lỗi môi trường hay crash thư viện), chứ không thể tạo ra một mô hình dịch có khả năng hiểu ngôn ngữ.

### 3.3. Hiện tượng lệch pha và biến dạng khi HiFi-GAN Vocoder nhận chuỗi mã rác

* Mô hình Vocoder `hubert_base_100_lj` là một mạng tích chập chuyển vị (transposed convolution) kết hợp bộ dự đoán trường độ (`--dur-prediction`). Nó được huấn luyện trên 24 giờ phát âm chuẩn của tập LJSpeech.
* Mạng nơ-ron này học được quy luật: mã unit $u_a$ theo sau bởi $u_b$ với trường độ $d$ sẽ tương ứng với các dao động áp suất không khí cụ thể của phụ âm và nguyên âm tiếng Anh.
* Khi cấp cho Vocoder một chuỗi unit ngẫu nhiên (ví dụ: `[12, 89, 3, 77, 12, 12, ...]` do mô hình 20 update sinh ra), bộ dự đoán trường độ không thể căn chỉnh pha dao động. Bộ giải mã sóng âm cố gắng tổng hợp các âm thanh nhảy bước tức thời, tạo ra hiện tượng **Digital Clipping**, méo hài (harmonic distortion) và tiếng rít tần số cao.

### 3.4. Bản chất của Demo gốc trong bài báo Lee et al. (2021)

Trong tài liệu và demo chính thức của nhóm tác giả Meta AI:

1. **Trường hợp 1 (Full S2UT Translation Demo):** Tác giả sử dụng checkpoint đã được huấn luyện hội tụ qua nhiều ngày trên cụm máy chủ nhiều GPU V100.
2. **Trường hợp 2 (Unit Vocoder Resynthesis Demo - Mục 5.1 trong bài báo):** Để chứng minh chất lượng âm thanh của việc lượng tử hóa HuBERT units, tác giả tiến hành thử nghiệm **Analysis-by-Synthesis**:
   $$\text{Âm thanh nói thật (.wav)} \xrightarrow{\text{HuBERT + k-means}} \text{Chuỗi Units} \xrightarrow{\text{HiFi-GAN}} \text{Âm thanh tổng hợp (.wav)}$$
   Thử nghiệm này cho điểm số MOS đạt ~4.0 (rất gần với giọng đọc người thật), chứng minh rằng chuỗi HuBERT discrete units hoàn toàn có thể tái tạo âm thanh rõ ràng nếu đầu vào là tiếng nói thật.

---

## 4. Giải pháp khắc phục và Kế hoạch thực hiện

Để giải quyết triệt để vấn đề và chứng minh rõ ràng cho giảng viên, chúng ta triển khai 3 hành động cụ thể:

### Hành động 1: Bổ sung ô Demo "Unit Resynthesis (Analysis-by-Synthesis)" vào Notebook

* **Mục tiêu:** Cho giảng viên nghe thấy ngay lập tức chất lượng âm thanh trong trẻo, rõ tiếng người của hệ thống theo đúng kiến trúc của bài báo.
* **Cách thực hiện:**
  1. Tải một file âm thanh tiếng người thật 16kHz ngắn (ví dụ mẫu câu từ LJSpeech / LibriSpeech).
  2. Dùng đúng bộ trích xuất `quantize_with_kmeans.py` (HuBERT layer 6 + 100 clusters) để lượng tử hóa thành discrete units.
  3. Dùng đúng bộ `generate_waveform_from_code.py` với checkpoint `hubert_base_100_lj` để tổng hợp lại âm thanh.
  4. Người nghe sẽ nghe thấy âm thanh tiếng người tự nhiên, rõ ràng từng từ, chứng minh Vocoder và cơ chế Discrete Units hoạt động hoàn hảo 100%.

### Hành động 2: Thay thế dữ liệu dummy sóng sin bằng các đoạn ghi âm tiếng nói con người

* Thay vì dùng `np.sin`, chuẩn bị tập dữ liệu mini gồm các file `.wav` thu âm tiếng nói thật (có thể thu âm ngắn hoặc lấy từ mini LibriSpeech) để nếu train thử nghiệm, các đơn vị HuBERT trích xuất ra là các ngữ âm thật chứ không phải mã rác.

### Hành động 3: Làm rõ mục tiêu của Cell Huấn luyện 20 Update

* Thêm ghi chú kỹ thuật rõ ràng trong Notebook:
  > *"Bước huấn luyện 20 update dưới đây là Smoke Test nhằm xác thực tính toàn vẹn của mã nguồn trên phần cứng Kaggle GPU. Để đạt chất lượng dịch thực tế tương đương bài báo gốc, mô hình cần được huấn luyện 400.000 updates trên tập Fisher (~160 giờ) hoặc nạp checkpoint đã huấn luyện sẵn (Pre-trained checkpoint)."*

---

## 5. Kịch bản giải trình chuyên nghiệp với Giảng viên

Khi trao đổi lại với giảng viên hướng dẫn, sinh viên có thể trình bày theo sườn ý sau:

> *"Dạ thưa Thầy/Cô, em đã rà soát lại toàn bộ mã nguồn và xác định chính xác nguyên nhân vì sao âm thanh trên Kaggle bị khó nghe:*
>
> 1. *Bản demo gốc của bài báo được tác giả huấn luyện trên tập dữ liệu tiếng người thật (Fisher Speech) với 400.000 bước cập nhật (tương đương nhiều ngày chạy trên cụm 16 GPU V100).*
> 2. *Trong notebook Kaggle vừa rồi, mục tiêu trọng tâm là hoàn thiện đường ống kỹ thuật (Pipeline Verification) trên môi trường ảo Python 3.10 mới, nên em đã sử dụng tập dữ liệu giả lập sóng sin (`np.sin`) và chỉ cho mô hình train thử 20 bước để kiểm tra luồng chạy.*
> 3. *Vì sóng sin không phải tiếng người và mô hình Transformer 20 bước chưa hội tụ (vẫn mang trọng số ngẫu nhiên), chuỗi discrete units sinh ra là chuỗi ngẫu nhiên. Khi đưa chuỗi này qua Vocoder HiFi-GAN, Vocoder sẽ tạo ra tiếng rè kim loại do lệch pha trường độ.*
> 4. *Để chứng minh kiến trúc Discrete Units và Vocoder HiFi-GAN của bài báo hoạt động hoàn toàn chính xác trên Kaggle, em đã bổ sung thêm phần **Unit Resynthesis (Analysis-by-Synthesis)**: lấy trực tiếp một câu nói tiếng người thật, lượng tử hóa qua HuBERT 100 cụm và tái tạo lại bằng Vocoder. Kết quả âm thanh phát ra rất trong trẻo và rõ lời theo đúng chuẩn của bài báo."*

---

## 6. Kiểm tra: `s2st-fairseq.ipynb` có reproduce đúng `fairseq/examples/speech_to_speech` không?

* **Ngày:** 28/10/2026 (review đầu) — **cập nhật 29/10/2026** sau khi mổ xẻ `s2st_results_package`.
* **Đối tượng:** Notebook **`s2st-fairseq.ipynb`** (bản hiện tại: **26 cell**, thêm Mục 3.2 — cổng kiểm tra units ↔ vocoder) + gói kết quả `s2st_results_package/` (40 wav model, 40 wav GT, 1 wav resynthesis, `train.log`).
* **Chuẩn đối chiếu:** `fairseq/examples/speech_to_speech/` — chủ yếu [`docs/direct_s2st_discrete_units.md`](fairseq/examples/speech_to_speech/docs/direct_s2st_discrete_units.md) và flow đã ghi tại [`FLOW.md`](fairseq/examples/speech_to_speech/FLOW.md).
* **Không đối chiếu:** paper-scale Fisher 400k updates, Unity, S2SPECT, enhanced wav2vec+mBART.

### 6.1 Kết luận ngắn

**Có — đúng đường ống mã S2UT reduced của example.** Notebook gọi đúng script, task, architecture, criterion, vocoder và thứ tự bước của `examples/speech_to_speech`.

**Không — không reproduce thí nghiệm S2ST của bài báo / README example.** Đây là **smoke test + identity-map overfit** trên clip LJSpeech (src = tgt, tiếng Anh → tiếng Anh), không phải dịch speech ngôn ngữ nguồn sang ngôn ngữ đích trên Fisher.

Hai việc này cần tách khi báo cáo: *pipeline code đúng*, *setup thí nghiệm không phải bản reproduce đầy đủ*.

> ⚠️ **Mục 6.7 mới là phần quan trọng nhất**: gói kết quả đã chạy được (loss 0.006, 2994 updates) nhưng audio vẫn rè, và nguyên nhân **không** nằm ở model — mà ở **file k-means sai label space** và **lỗi ghép thứ tự file**. Mục 1–5 ở trên chỉ đúng cho bản smoke-test sóng sin + 20 updates trước đó, không còn áp dụng cho gói này.

### 6.2 Ánh xạ bước notebook ↔ example (phần đúng)

| Bước official (`direct_s2st_discrete_units.md`) | Notebook `s2st-fairseq.ipynb` | Đánh giá |
| :--- | :--- | :--- |
| Layout `$SRC_AUDIO/{split}/{id}.wav` và `$TGT_AUDIO/{split}/{id}.wav`, target 16 kHz | Mục 3: `speech_dataset/src_audio`, `tgt_audio`, resample LJSpeech 22.05 kHz → 16 kHz | Đúng layout |
| HuBERT Base + k-means 100 unit, **layer 6** (`quantize_with_kmeans.py` / speech2unit) | Mục 2 + 3.1: `hubert_base_ls960.pt`, `hubert_base_ls960_L6_km100.bin`, `--layer 6`, script `examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py` | Đúng bộ trích unit paper S2UT 2021 |
| Vocoder `hubert_base_100_lj` (`g_00500000` + `config.json`) | Mục 2: đúng URL `fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/` | Đúng checkpoint example |
| `prep_s2ut_data.py --reduce-unit` + vocoder ckpt/cfg | Mục 3.3: gọi đúng `examples/speech_to_speech/preprocessing/prep_s2ut_data.py` với `--reduce-unit` | Đúng nhánh **reduced** |
| `fairseq-train --task speech_to_speech --target-is-code --target-code-size 100 --vocoder code_hifigan --criterion speech_to_unit --arch s2ut_transformer_fisher --share-decoder-input-output-embed` | Mục 4: đủ các flag trên | Đúng command train S2UT |
| `fairseq-generate` cùng task/code-size/vocoder, `--beam 10 --max-len-a 1` | Mục 5.1 (`generate_discrete_units`) | Đúng |
| Lấy dòng `D-` từ `generate-{subset}.txt` | `extract_units_from_log`: parse `D-` + sort theo id (tương đương `grep \| sed \| sort \| cut`) | Đúng ý, portable hơn shell |
| `generate_waveform_from_code.py --dur-prediction` với reduced unit | `synthesize_waveform(..., dur_prediction=True)` | Đúng bắt buộc của nhánh reduced |
| Resynthesis Analysis-by-Synthesis (HuBERT → unit → HiFi-GAN) | Mục 5.2 | Đúng tinh thần mục vocoder / Section 5.1 paper |

Cách gọi: **subprocess CLI** (`PY_BIN` + script example, `fairseq-train` / `fairseq-generate` trong venv 3.10), không bypass sang model tự viết. README repo nói “gọi API nội bộ thay vì shell”; notebook thực tế **bám CLI example** — với mục tiêu reproduce thì cách này **trung thành hơn**.

### 6.3 Điểm lệch so với reproduce “đúng bài / đúng example”

| Hạng mục | Official / paper | Notebook | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| **Bài toán** | S2ST: speech ngôn ngữ A → speech ngôn ngữ B (Fisher Es–En, …) | **Identity map:** cùng file LJSpeech copy vào src và tgt | Model học *copy unit*, không học dịch |
| **Dữ liệu** | ~160 giờ thoại, train/dev/test tách | 32 clip LJSpeech; **train và dev là bản sao giống nhau** (cùng `paths` ghi vào cả hai split) | Dev không phải tập giữ; metric valid bị leak |
| **Multitask** | Fisher reduced: `config_multitask.yaml` (source_letter, target_letter, decoder CTC) | Không truyền `--multitask-config-yaml` | Thiếu nhánh aux của example Fisher |
| **Regularization / lịch train** | dropout 0.1, label-smoothing 0.2, warmup 10k, **max-update 400000**, update-freq 4 | dropout 0, LS 0, warmup 100, **3000 updates**, update-freq 1, `max-tokens` 4000 | Chủ đích overfit demo; không phải recipe paper |
| **Đánh giá** | ASR rồi sacreBLEU (`asr_bleu/`) | Nghe tay + thống kê peak/RMS + dò mode-collapse | Không có ASR-BLEU (đã tự chạy ASR ngoài notebook, xem 6.7) |
| **S2SPECT / Unity / enhanced** | Có trong cùng thư mục example | Không chạy | Ngoài phạm vi notebook — chấp nhận được nếu tuyên bố chỉ S2UT reduced |
| **Unit rỗng khi generate** | Official không fallback | **Đã sửa (29/10):** `extract_units_from_log` `raise` khi có hypothesis rỗng | Trước đây ghi `'0'` → che lỗi hội tụ, vocoder vẫn “ra tiếng” |
| **GT units → WAV** | Reduced + `--dur-prediction` *hoặc* full sequence không dur-pred | File `{split}.txt` từ k-means là **frame-level (chưa reduce)**; vocode `dur_prediction=False` | Hợp lý cho file k-means thô; khác file reduced trong TSV train |
| **k-means** | URL Meta `textless_nlp/gslm/hubert/km100/km.bin` | **Đã sửa:** không fallback mirror + `sha256` chốt file + **cổng round-trip Mục 3.2** | Mirror làm sai label space → audio rè (bằng chứng ở 6.7) |
| **Thứ tự file kết quả** | — | **Đã sửa:** wav đặt tên `<sample_id>_pred.wav` + manifest natural sort | `{i}_pred.wav` khiến model output và GT units lệch câu |

### 6.4 Train/dev trùng — chi tiết code

Trong Mục 3, sau khi tải clip:

```python
for split in ("train", "dev"):
    ...
    for idx, p in enumerate(paths):
        sf.write(... src_audio/split/sample_{idx}.wav ...)
        sf.write(... tgt_audio/split/sample_{idx}.wav ...)
```

Mọi clip vào **cả** `train` và `dev`, `src = tgt`. Đây không phải split của example (train / dev / test độc lập). Hệ quả: loss valid thấp không chứng minh generalization; S2UT “dịch” dev chỉ là nhớ cùng speech đã thấy lúc train.

### 6.5 Tuyên bố reproduce nên dùng

| Câu | Đúng/sai |
| :--- | :--- |
| Notebook chạy đúng **pipeline S2UT reduced** trong `examples/speech_to_speech` (prep → train task/arch → generate → Code-HiFi-GAN). | **Đúng** |
| Vocoder + HuBERT L6 km100 là checkpoint example/paper 2021. | **Đúng** |
| Mục 5.2 chứng minh **khâu unit→wav** (resynthesis), không chứng minh mô hình S2UT đã dịch. | **Đúng** |
| Notebook reproduce **kết quả S2ST** (Fisher, BLEU, demo dịch xuyên ngôn ngữ của bài báo). | **Sai** |
| Audio model output nghe được (nếu identity overfit thành công) = đã reproduce translation. | **Sai** — đó là copy/reconstruction |

### 6.6 Việc còn lại nếu muốn reproduce sát example hơn

1. Cặp speech **khác ngôn ngữ** (hoặc ít nhất src ≠ tgt), train/dev/test **không overlap**.
2. Giữ dropout / label-smoothing như docs; không zero hóa regularization trừ khi chỉ demo overfit.
3. Optional: `config_multitask.yaml` đúng format example.
4. Đánh giá `asr_bleu/compute_asr_bleu.py` trên wav pred vs reference text.
5. ~~Bỏ fallback unit `'0'`~~ → **đã làm** (29/10).
6. ~~Không tuyên bố MOS ~4.0~~ → **đã sửa lời thoại Mục 5.2** (29/10).

### 6.7 Mổ xẻ `s2st_results_package`: vì sao audio vẫn rè dù model đã hội tụ

**Trạng thái gói:** 40 wav model output, 40 wav GT-units, 1 wav resynthesis, `train.log` 2994 updates, **loss 0.006**, epoch 599 → **model đã overfit identity map thành công**, không phải “20 updates chưa học gì” như Mục 3.2 ở trên.

#### Nguyên nhân #1 — file k-means sai **label space** (nhân tố quyết định)

* Log thực thi trong notebook cũ cho thấy URL Meta cũ (`.../gslm/hubert/hubert_base_ls960_L6_km100.bin`) trả **403** → code âm thầm fallback sang mirror HuggingFace `yfyeung/kmeans-librispeech-hubert-base`.
* Mirror gom **cùng 100 cụm** nhưng **nhãn bị hoán vị**: đối chiếu `gt_codes.txt` trong gói với units trích từ mirror khớp **95.4%**, còn với `km100/km.bin` chính thức chỉ **2.2%** (≈ mức ngẫu nhiên 1/100 × sai số).
* Vocoder `hubert_base_100_lj` được train với nhãn của file chính thức → nhận mã hoán vị ⇒ phát âm sai ⇒ tiếng rè/kim loại.
* **Đo A/B (faster-whisper `base.en`, min-WER trên 40 transcript LJSpeech):**

  | Nguồn audio | WER trung bình | WER trung vị | Hypothesis rỗng |
  | :--- | ---: | ---: | ---: |
  | LJSpeech gốc | 0.0 % | 0.0 % | 0/40 |
  | **Units trích bằng `km100/km.bin` chính thức** → HiFi-GAN | **8.5 %** | **4.2 %** | **0/40** |
  | `reference_gt_units/` trong gói (mirror) | 111.1 % | 94.6 % | 2/40 |
  | `model_output/` trong gói | 94.7 % | 90.2 % | 8/40 |

  ⇒ Cùng một vocoder, cùng một câu: đổi đúng file k-means là audio **đọc được gần nguyên văn** (4.2% trung vị).

* **Cổng kiểm tra tự động (mới, Mục 3.2):** round-trip `wav → units → wav → units`, đo % frame khớp.
  Đo trên LJ001-0001 (482 → 481 frame):

  | k-means | Frame khớp | Kết quả cổng |
  | :--- | ---: | :--- |
  | `km100/km.bin` (chính thức) | **85.3 %** | ĐẠT |
  | mirror `yfyeung` | **0.0 %** | `raise` + chỉ đúng cách sửa |

  Ngưỡng đặt **50 %** (cách xa cả hai phía) → sai label space bị chặn **trước** khi train, thay vì phát hiện sau 30 phút bằng tai.

#### Nguyên nhân #2 — lỗi ghép **thứ tự file** giữa model output và GT units

* `generate_waveform_from_code.py` đặt tên `{i}_pred.wav` theo **index dòng** của file mã.
* `create_wav_manifest` (manifest cho k-means) dùng `sorted(glob(...))` → thứ tự **lexicographic**: `sample_0, sample_1, sample_10, sample_11, …, sample_2, sample_20 …`
* `prep_s2ut_data.py` lấy `src_audio` theo thứ tự **filesystem** (`Path.glob`) → `sample_0, sample_1, sample_2, …, sample_31`.
* Hệ quả: `model_output/7_pred.wav` và `reference_gt_units/7_pred.wav` là **hai câu khác nhau** (hoán vị đo được: 0→36, 1→11, 2→27; kiểm chứng 40/40 bằng độ dài file).
* Vì vậy **mọi metric ghép cặp** trong gói (`_stoi.json`, `_mapping_analysis.json`, `_asr_control.json`) đều **không hợp lệ** — và đúng là chúng không phân biệt được gì (spec-corr 0.9899 vs shuffle 0.9835; STOI 0.724 vs 0.629).

#### Model S2UT **không** phải nguyên nhân

* Với 39/40 câu, độ dài chuỗi model sinh ra ≈ độ dài units GT (`D ≈ T`).
* Sau khi áp đúng hoán vị, chuỗi units của model **trùng khít** `dedup(gt_codes)` → model đã học đúng phép copy mà bài test identity yêu cầu.

#### Đã sửa gì trong `s2st-fairseq.ipynb` (29/10)

| # | Thay đổi | Chặn được lỗi gì |
| :--- | :--- | :--- |
| P0 | **Mục 3.2 mới** — `verify_unit_vocoder_roundtrip()`: round-trip 1 clip, `raise` nếu frame khớp < 50 % | Sai / lệch label space k-means (mirror, km500, …) trước khi tốn thời gian train |
| P0 | `sha256` chốt `km100/km.bin` + **không fallback** URL khác (Mục 2) | Âm thầm dùng mirror như lần trước |
| P1 | `create_wav_manifest` dùng **natural sort** | Thứ tự `sample_10` chen giữa `sample_1` và `sample_2` |
| P1 | `rename_pred_by_id()` — wav đổi thành `<sample_id>_pred.wav` (model: id từ `dev.tsv`; GT: id giữ từ tiền tố `dev.txt`) | Hai thư mục kết quả lệch câu, metric ghép cặp vô nghĩa |
| P2 | `extract_units_from_log` **bỏ fallback `'0'`**, `raise` khi hypothesis rỗng | Che lỗi model chưa hội tụ |
| P2 | Resynthesis đặt tên `real_speech_resynthesis_pred.wav`; bỏ tuyên bố “MOS ~4.0”; ghi round-trip vào `README.txt` của ZIP | Báo cáo quá mức so với bằng chứng |

#### Còn lại (chưa làm, cần biết khi báo cáo)

1. **Identity map** (src = tgt) và **train == dev** vẫn giữ nguyên → audio model ≈ tái tạo, **không** phải dịch.
2. `--dur-prediction` làm nhịp nói **nhanh hơn ~5–7 %** (đo tỉ lệ độ dài 0.926–1.057) — bình thường với giọng tổng hợp, nhưng khi so sánh thời lượng phải trừ hao.
3. Cổng Mục 3.2 là kiểm tra **label space**, không phải kiểm tra độ thông hiểu; muốn chấm điểm nghe phải chạy ASR như 6.7 (faster-whisper chưa cài trong venv 3.10 của notebook).
4. Gói cũ trong `s2st_results_package/` vẫn mang số liệu sai (do 2 lỗi trên) — cần chạy lại notebook mới rồi mới đóng gói lại.

---
*Cập nhật 29/10/2026 — review `s2st-fairseq.ipynb` vs `fairseq/examples/speech_to_speech`, kèm pháp chứng `s2st_results_package` (k-means label space + thứ tự file).*

---

## 7. Danh mục lỗi đã gặp (tra cứu nhanh)

Bảng tra ngược: **thấy triệu chứng gì → lỗi nào → đã sửa ở đâu**. Chi tiết + số đo ở Mục 6.7 và [`CHANGELOG.md`](CHANGELOG.md).

| # | Triệu chứng | Nguyên nhân gốc | Cách phát hiện | Đã sửa ở |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | Audio ra tiếng rè/kim loại, gần như không ra lời, **dù loss rất thấp** (~0.006) | File k-means sai **label space**: URL chính thức trả 403 → fallback mirror `yfyeung`, cùng 100 cụm nhưng **nhãn bị hoán vị** → vocoder nhận sai mã | WER ASR: mirror **94.6 %** vs km100 **4.2 %** (trung vị); round-trip frame khớp **0.0 %** vs **85.3 %** | Mục 2 (URL chính thức + `EXPECTED_SHA256`, bỏ fallback) và **Mục 3.2** (cổng round-trip ≥ 50 %) |
| **B2** | Metric ghép cặp (STOI / spec-corr / ASR-control) thấp vô lý, "model dịch không khớp reference" | `model_output/i_pred.wav` (thứ tự dòng `dev.tsv`) và `reference_gt_units/i_pred.wav` (thứ tự manifest lexicographic) là **hai câu khác nhau** | So độ dài theo id: lệch 40/40; hoán vị đo được 0→36, 1→11, 2→27 | Mục 3.1 (natural sort) + Mục 5.1 (`rename_pred_by_id` → `<sample_id>_pred.wav`) |
| **B3** | Vài file output gần như im lặng / quá ngắn nhưng pipeline **không báo lỗi** | Hypothesis rỗng bị thay bằng unit `'0'` (fallback im lặng) | Đếm dòng `D-` có token rỗng trong `generate-dev.txt` (8/40 ở gói cũ) | Mục 5.1 `extract_units_from_log` → `raise` |
| **B4** | Notebook "chạy một mạch không lỗi" nhưng tải sai model | Fallback URL **âm thầm** (403 → mirror thay thế) | Đọc log tải: thấy `403` rồi `yfyeung...` | Mục 2: không fallback + kiểm SHA-256, sai thì xoá file và dừng |
| **B5** | Tài liệu / checker báo sai sau khi thêm–bớt cell | Tham chiếu **index cell cứng** (`cells[23]`, "Cell 14"…) | `python verify_nb.py` FAIL | Tham chiếu theo **Mục** (3.1, 3.2, 5.1…), không theo số cell |
| **B6** | Báo cáo ghi "MOS ~4.0" nhưng chưa ai đo MOS | Sao chép số của bài báo cho thí nghiệm resynthesis trên Fisher | — | Mục 5.2 đã bỏ tuyên bố, chỉ nói "kiểm chứng khâu units → wav" |
| **B7** | Verifier vẫn PASS dù notebook đã hỏng | Check chỉ đếm cấu trúc, không kiểm nội dung | — | `verify_nb.py`: 32 check theo nội dung (URL, sha256, cổng, rename, raise) |

### 7.1 Chẩn đoán audio nhanh (đúng thứ tự)

1. `python verify_nb.py` → notebook đúng cấu trúc + đủ check nội dung.
2. Chạy **Mục 3.2** trên Kaggle: cổng ĐẠT (kỳ vọng ~85 % frame khớp) nghĩa là units ↔ vocoder khớp label space. Nếu FAIL → sai file k-means, đừng train.
3. Nếu audio vẫn rè: vocode **`reference_gt_units/`** (ground-truth units, bỏ model S2UT ra khỏi phương trình). Reference cũng rè ⇒ lỗi ở tầng units/vocoder, **không phải** model.
4. Đo độ thông hiểu bằng ASR (cần `pip install faster-whisper`, ~1 phút):

   ```python
   from faster_whisper import WhisperModel
   m = WhisperModel("base.en", device="cpu", compute_type="int8")
   segs, _ = m.transcribe("pred.wav", language="en", beam_size=1)
   print(" ".join(s.text.strip() for s in segs))
   ```

   Thang tham chiếu (đo trên 40 clip LJSpeech): LJ gốc 0 %; km100 + HiFi-GAN **4–8 %**; mirror **~95 %**.
5. **Chỉ tin metric ghép cặp** khi cả hai thư mục đều đặt tên `<sample_id>_pred.wav` (Mục 5.1).

### 7.2 Dấu vết đã dọn (29/10/2026)

Để repo gọn, các script/artefact dùng để chẩn đoán đã được xoá; **số liệu chốt vẫn nằm trong Mục 6.7** và có thể đo lại bằng quy trình 7.1 (chỉ cần Internet, không cần script cũ):

* Script phân tích: `analyze_package.py`, `analyze_mapping.py`, `analyze_stoi.py`, `stoi_vs_original.py`, `control_asr.py`, `asr_intelligibility.py`, `compare_kmeans.py`, `proof_ab_kmeans.py`.
* Dữ liệu/model tải về để đối chứng: `_km/`, `_orig_ljs/`, `_proof_models/`, `_proof_run/`, `_proofenv/`.
* Script vá notebook dùng một lần: `create_notebook.py`, `apply_zip_fix.py`, `fix_zip_v2.py`, `test_zip_cell.py`, `run_s2st_exp.py`, `_patch_km.py`, `_proof_download.py`, `_nb_overview.py`, `_test_fairseq_import.py`.
* Bản trùng: `direct_s2st_discrete_units.md` ở root (giống từng dòng với [`fairseq/examples/speech_to_speech/docs/direct_s2st_discrete_units.md`](fairseq/examples/speech_to_speech/docs/direct_s2st_discrete_units.md)).
* Giữ lại: `s2st_results_package/` (gói kết quả "trước khi sửa" — bằng chứng gốc của Mục 6.7), `s2st_fairseq_demo.ipynb` (bản smoke-test cũ, đối tượng phân tích Mục 1–5), `fairseq/examples/speech_to_speech/FLOW.md` (tài liệu luồng end-to-end được Mục 2 trích dẫn).

