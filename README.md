# Natural Language Processing & Applications (NLP 2026)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Coursework-success)
![Institution](https://img.shields.io/badge/Institution-VNU--HUS-informational)

---

## Overview
This repository contains coursework, lab assignments, and experimental code for the **Natural Language Processing & Applications** course (Semester I - 2026) at **VNU University of Science (VNU-HUS)**.

The repository is organized progressively by lab topics and experimental modules throughout the course, transitioning from fundamental lexical representations to advanced deep learning architectures for NLP.

---

## Cấu trúc dự án

```text
NLP/
├── README.MD                   # Tài liệu tổng quan toàn bộ kho lưu trữ
├── requirements.txt            # Danh sách các thư viện phụ thuộc của dự án
├── Lab1/                       # Lab 01: From Text Processing to Search
│   ├── W1.pdf                  # Đề bài và hướng dẫn thực hành Lab 1
│   ├── c4-train...-30K.json    # Tập ngữ liệu 30.000 documents trích từ C4
│   ├── implementation.py       # Triển khai thuật toán cốt lõi TF-IDF & Unit Tests (Phần 8)
│   ├── experiments.ipynb       # Notebook thực nghiệm phân tích Sparse Matrix (Phần 7, 9-13)
│   ├── Calculations.pdf        # Lời giải chi tiết các bài toán tính tay (Part B)
│   ├── Prediction.pdf          # Dự đoán giả thuyết trước khi chạy thực nghiệm (Part C)
│   ├── results.csv             # Kết quả đo lường và đánh giá retrieval (Part H)
│   └── Reflection.pdf          # Báo cáo phản tư, phân tích lỗi & định hướng (Part 16)
└── Lab2/                       # Lab 02: Language Models (N-gram, Smoothing & Perplexity)
    ├── W2.pdf                  # Đề bài và hướng dẫn thực hành Lab 2
    ├── README.md               # Hướng dẫn chi tiết riêng cho Lab 2
    ├── calculations.pdf         # Tính toán xác suất bằng tay (Unigram, Bigram, PPL, Laplace)
    ├── prediction.pdf           # 5 dự đoán trước thực nghiệm (Prediction 1-5)
    ├── ngram_lm.py             # Cốt lõi: NGramLanguageModel, MLE, Laplace, Log-prob
    ├── experiments.ipynb       # Thực nghiệm C4: Thống kê Zipf, Perplexity, Next-Word, Ranking
    ├── results.csv             # Bảng tổng hợp Perplexity trên Train, Validation, Test
    ├── error_analysis.pdf       # Phân tích 2 ca dự đoán đúng & 2 ca dự đoán sai
    └── reflection.pdf           # Báo cáo tổng kết 7 câu hỏi kết nối sang Neural LM
```

---

## Nội dung các lab

### Lab 1 — Tách từ và biểu diễn văn bản bằng vector thưa
- **Chủ đề:** Text Processing, Sparse Vector Representations, TF-IDF và Document Search.
- **Nội dung chính:**
  - Thực hiện tokenization, normalization, loại bỏ stopwords và tiền xử lý đầu vào.
  - Tự cài đặt từ đầu (from scratch) các thành phần cốt lõi: Vocabulary Builder, Count Vector, Term Frequency (TF), Inverse Document Frequency (IDF), TF-IDF, và Cosine Similarity.
  - Thí nghiệm khảo sát trên kho ngữ liệu thực tế gồm **30.000 văn bản tiếng Anh** từ tập dữ liệu C4.
  - Xây dựng Document Search Engine, đánh giá Precision@5, Recall@5, MRR và phân tích hiện tượng **Vocabulary Mismatch Problem**.

### Lab 2 — Mô hình ngôn ngữ N-gram, Làm mịn và Perplexity
- **Chủ đề:** N-gram Language Models, Zero Probability, Laplace Smoothing, Perplexity và Ứng dụng.
- **Nội dung chính:**
  - Tự triển khai mô hình N-gram (Unigram, Bigram, Trigram) từ đầu theo chuẩn OOP với class `NGramLanguageModel`.
  - Giải quyết hiện tượng Zero-Frequency bằng **Laplace (Add-1) Smoothing** và xử lý chống Numerical Underflow bằng Log-probability.
  - Thực nghiệm khảo sát phân phối Zipf của N-gram trên tập dữ liệu C4 (hơn 50.000 câu văn).
  - Đánh giá định lượng mô hình qua chỉ số **Perplexity (PPL)** trên 3 tập tách biệt (Train, Validation, Test).
  - Triển khai 2 ứng dụng thực tế: **Dự đoán từ tiếp theo (Next-Word Prediction)** và **Xếp hạng câu (Sentence Ranking)**.
  - Phân tích lỗi và làm rõ các giới hạn cốt lõi của N-gram (ngữ cảnh ngắn, bùng nổ dữ liệu thưa, thiếu khái quát hóa ngữ nghĩa), tạo tiền đề cho Word Embeddings và Neural Language Models (RNN, Transformer).

---

## Các bước triển khai

### 1. Khởi tạo môi trường ảo (Khuyến nghị)
Sử dụng `venv` hoặc `conda` để tạo môi trường Python độc lập:
```bash
# Tạo môi trường ảo với Python 3.9+
python -m venv venv

# Kích hoạt môi trường trên Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Hoặc kích hoạt trên Linux/macOS:
source venv/bin/activate
```

### 2. Cài đặt các thư viện phụ thuộc
Cài đặt toàn bộ các gói thư viện cần thiết từ file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Thực thi và kiểm tra mã nguồn Lab 1

- **Chạy kiểm thử tự động thuật toán cốt lõi (Phần 8):**
  ```bash
  cd Lab1
  python implementation.py
  ```
  *Kết quả kỳ vọng:* Tất cả 6 hàm tính toán (`build_vocabulary`, `compute_counts`, `compute_tf`, `compute_idf`, `compute_tfidf`, `cosine_similarity`) đều vượt qua kiểm thử với sai số $< 10^{-9}$.

- **Mở và chạy Notebook thực nghiệm trên 30K documents (Phần 7):**
  Khởi động Jupyter Notebook / JupyterLab hoặc mở trực tiếp trong VS Code:
  ```bash
  jupyter notebook experiments.ipynb
  ```
  Các cell trong notebook sẽ nạp tập C4, tính toán ma trận thưa, đo độ thưa và trực quan hóa top từ vựng.

---

## Công nghệ & Thư viện sử dụng
- **Ngôn ngữ lập trình:** Python 3.9+
- **Khoa học dữ liệu & Tính toán số học:** NumPy, SciPy, Pandas
- **Học máy & NLP cổ điển:** scikit-learn, NLTK
- **Môi trường thực nghiệm & Trực quan:** Jupyter Notebook, Matplotlib
