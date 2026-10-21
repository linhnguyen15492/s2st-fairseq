# Direct Speech-to-Speech Translation (S2ST) with Fairseq

Dự án cung cấp pipeline hoàn chỉnh cho mô hình **Direct Speech-to-Speech Translation with Discrete Units (Lee et al., 2021)** dựa trên **Fairseq**.

---

## 🚀 Tính năng nổi bật
* **Kiến trúc Direct S2ST:** Âm thanh nguồn (Source Speech) ➡️ S2UT Transformer ➡️ Mã Discrete Units (HuBERT) ➡️ HiFi-GAN Vocoder ➡️ Âm thanh đích (Target Speech `.wav`).
* **Modular Python Functions:** Gọi trực tiếp các API nội bộ của Fairseq thay vì lạm dụng shell scripts (`!python examples/...`, `grep | sed | cut`).
* **Hỗ trợ đa nền tảng:** Tương thích Windows, Linux, Google Colab và Kaggle GPU T4.
* **Tự động xử lý lỗi:** Hỗ trợ mirror HuggingFace khi link Facebook CDN bị 403, tự tạo manifest TSV chuẩn và xử lý kích thước batch nhỏ với `--required-batch-size-multiple 1`.

---

## 🛠️ Hướng dẫn cài đặt môi trường Conda (Python 3.12)

> ⚠️ **Lưu ý về phiên bản Python:**
> Khuyến nghị sử dụng **Python 3.12** hoặc **3.10**. **Không dùng Python 3.13** vì Fairseq 0.12.2 và Omegaconf 2.0.6 chưa tương thích với dataclass của Python 3.13.

### Cách 1: Cài đặt nhanh bằng `environment.yml` (Khuyên dùng)

```bash
# 1. Clone repository (nếu chưa clone)
git clone https://github.com/linhnguyen15492/s2st-fairseq.git
cd s2st-fairseq

# 2. Tạo môi trường Conda từ file environment.yml
conda env create -f environment.yml

# 3. Kích hoạt môi trường
conda activate fairseq-py312

# 4. Đăng ký Kernel với Jupyter Notebook / VS Code
python -m ipykernel install --user --name fairseq-py312 --display-name "Python 3.12 (s2st-fairseq)"

# 5. Cài đặt Fairseq dưới dạng editable package
cd fairseq
pip install --editable ./ --no-build-isolation
```

### Cách 2: Thiết lập thủ công từng bước bằng CLI

```bash
# 1. Tạo và kích hoạt môi trường conda với Python 3.12
conda create -n fairseq-py312 python=3.12 -y
conda activate fairseq-py312

# 2. Cài đặt PyTorch và Torchaudio
# Dành cho máy có GPU NVIDIA (CUDA 12.1):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# Dành cho máy chỉ chạy CPU:
# pip install torch torchaudio

# 3. Cài đặt các thư viện phụ thuộc (BẮT BUỘC: numpy < 2.0)
pip install "numpy<2.0.0" setuptools wheel cython soundfile sacrebleu sentencepiece joblib editdistance ipykernel

# 4. Đăng ký Kernel với Jupyter
python -m ipykernel install --user --name fairseq-py312 --display-name "Python 3.12 (s2st-fairseq)"

# 5. Cài đặt Fairseq
cd fairseq
pip install --editable ./ --no-build-isolation
```

---

## 📓 Chạy Notebook Demo

1. Mở notebook [`s2st_fairseq_demo.ipynb`](s2st_fairseq_demo.ipynb) trong **VS Code** hoặc **Jupyter Lab**.
2. Ở góc trên bên phải, bấm vào nút chọn **Kernel** và chọn: **`Python 3.12 (s2st-fairseq)`**.
3. Chạy lần lượt các cell từ trên xuống dưới để trải nghiệm toàn bộ pipeline từ tải mô hình, chuẩn bị dữ liệu, huấn luyện đến sinh file âm thanh `.wav`.
