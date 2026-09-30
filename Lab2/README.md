# LAB 02 — LANGUAGE MODELS: N-GRAM, SMOOTHING & PERPLEXITY

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP 2026)  
**Trường:** ĐH Khoa học Tự nhiên - ĐHQGHN (VNU-HUS)  
**Giảng viên / TA:** Phạm Ngọc Hải • **Hạn nộp:** 23:59 - 30/09/2026  

---

## 1. Cấu trúc thư mục nộp bài (Deliverables)
Tuân thủ đầy đủ và chính xác cấu trúc thư mục quy định tại Mục 27 của `W2.pdf`:
```
Lab2/
├── README.md              # Giới thiệu tổng quan, hướng dẫn chạy và tóm tắt kết quả (Rubric 100đ)
├── calculations.md        # Lời giải chi tiết tính toán lý thuyết bằng tay (Bài 1, 2, 3, 4, 9, 11, 18)
├── prediction.md          # 5 giả thuyết & dự đoán trước khi chạy thực nghiệm (Mục 12)
├── ngram_lm.py            # Module cốt lõi: NGramLanguageModel, MLE, Laplace, Log-prob, PPL
├── experiments.ipynb      # Toàn bộ thực nghiệm trên C4 Corpus, Zipf distribution, PPL, Ứng dụng
├── results.csv            # Bảng tổng hợp chỉ số Perplexity trên tập Train, Validation, Test
├── error_analysis.md      # Phân tích chi tiết 2 trường hợp đúng & 2 trường hợp thất bại (Mục 22)
└── reflection.md          # Báo cáo tổng kết 7 câu hỏi từ N-gram đến Neural LM (Mục 24)
```

---

## 2. Hướng dẫn chạy và Kiểm thử

### 2.1 Cài đặt môi trường
Sử dụng chung môi trường Python 3.9+ đã cấu hình từ Lab 1:
```bash
pip install numpy pandas matplotlib scipy nbformat nbclient ipykernel
```

### 2.2 Chạy Unit Tests thuật toán cốt lõi
Kiểm tra tính chính xác của các hàm tính toán xác suất, Laplace smoothing và Perplexity:
```bash
python ngram_lm.py
```
*Kết quả kỳ vọng:* Toàn bộ unit tests `build_vocabulary`, `Unigram counts`, `Bigram MLE`, `Zero Probability`, `Laplace Smoothing`, và `Perplexity` đều vượt qua (`ALL UNIT TESTS PASSED!`).

### 2.3 Chạy Notebook thực nghiệm
```bash
jupyter execute experiments.ipynb
```

---

## 3. Tóm tắt kết quả thực nghiệm chính

### 3.1 Thống kê ngữ liệu (Corpus Statistics — Mục 13)
* Ngữ liệu toàn bộ **30.000 văn bản C4**: gồm **691.345 câu** và **11.093.241 tokens**.
* Kích thước từ vựng ($|\mathcal{V}|$): **186.618 unique words**.
* Phân phối tần suất: Tuân theo chặt chẽ **Định luật Zipf (Zipf's Law)**, trong đó số lượng n-gram xuất hiện đúng 1 lần (Singletons) tăng vọt khi bậc $n$ tăng:
  - Unigram singletons: **88.482** (chiếm **47.41%**).
  - Bigram singletons: **1.953.031** (chiếm **71.30%**).
  - Trigram singletons: **5.521.625** (chiếm **86.21%**).

### 3.2 Bảng so sánh Perplexity (Train vs Validation vs Test — Mục 16 & 19)

| Mô hình (Model) | Train PPL | Valid PPL | Test PPL | Nhận xét bản chất |
| :--- | :---: | :---: | :---: | :--- |
| **Bigram MLE** | **209.81** | **$+\infty$** | **$+\infty$** | Zero Probability triệt tiêu toàn bộ xác suất trên tập mới. |
| **Bigram Laplace** | 5351.49 | 8152.02 | 6703.09 | Phân phối mượt mà, giải quyết triệt để Zero Probability. |
| **Trigram MLE** | **19.71** | **$+\infty$** | **$+\infty$** | Overfit cực nặng trên tập train; gãy hoàn toàn trên tập test. |
| **Trigram Laplace** | 27462.79 | 49395.52 | 43047.01 | Bị phạt nặng bởi Laplace do không gian trạng thái $V^2$ quá thưa. |
| **Unigram Laplace** | 1915.23 | 2279.18 | 2050.78 | Độc lập ngữ cảnh; tổng quát hóa ổn định nhất trên tập kiểm thử. |

### 3.3 Ứng dụng thực tế
1. **Dự đoán từ tiếp theo (Next-Word Prediction):** Mô hình Trigram nắm bắt chính xác các cụm từ ngữ cố định như `natural language` $\to$ `processing`, `one of` $\to$ `the`.
2. **Xếp hạng câu (Sentence Ranking):** Phát hiện hiện tượng lệch xếp hạng do hiệu ứng phạt theo độ dài (Length Penalty) khi so sánh $P(S)$, và cách Perplexity sửa sai thành công để đưa câu có nghĩa lên hạng 1.
