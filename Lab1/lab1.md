# LAB 01 — From Text Processing to Search
**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I - 2026 | ĐH Khoa học Tự nhiên - ĐHQGHN  
**Giảng viên / TA:** Phạm Ngọc Hải  

---

## 1. Cấu trúc thư mục nộp bài (Deliverables)
Tuân thủ đầy đủ cấu trúc thư mục quy định tại Mục 17 của `W1.pdf`:
```
lab01/
├── README.md              # Giới thiệu tổng quan, hướng dẫn chạy và tóm tắt kết quả
├── calculations.md        # Lời giải chi tiết và diễn giải các bài toán tính tay (Part B)
├── prediction.md          # Các giả thuyết & dự đoán trước khi mở dữ liệu 30K (Part C)
├── implementation.py      # Triển khai thuật toán cốt lõi TF-IDF, Cosine Similarity và Unit Tests (Part E)
├── experiments.ipynb      # Toàn bộ mã nguồn thực nghiệm trên 30K documents, ablation, search và evaluation
├── results.csv            # Bảng kết quả định lượng Precision@5, Recall@5, MRR (Part H)
└── reflection.md          # Báo cáo tổng kết, tự đánh giá, phân tích lỗi và định hướng (Part J & 16)
```

---

## 2. Hướng dẫn cài đặt & Thực thi

### 2.1 Cài đặt môi trường
Yêu cầu Python 3.9+ cùng các thư viện cần thiết:
```bash
pip install numpy scipy scikit-learn pandas matplotlib nltk nbformat nbclient ipykernel
```

### 2.2 Chạy kiểm thử cốt lõi (Unit Tests)
Chạy lệnh kiểm tra 7 unit tests tự động của các hàm toán học trong `implementation.py`:
```bash
python implementation.py
```
Kết quả kỳ vọng: Tất cả các bài kiểm thử `build_vocabulary`, `compute_counts`, `compute_tf`, `compute_idf`, `compute_tfidf`, `cosine_similarity`, và `SimpleTFIDF` đều vượt qua với sai số $< 10^{-9}$.

### 2.3 Chạy Notebook thực nghiệm
Mở file `experiments.ipynb` trong VS Code / Jupyter Notebook hoặc chạy qua dòng lệnh:
```bash
jupyter execute experiments.ipynb
```

---

## 3. Tóm tắt kết quả chính
- **Thực nghiệm 1 (Sparse Representation trên 30K C4 Documents):**
  - Số lượng tài liệu: $N = 30,000$
  - Kích thước từ điển chọn lọc: $V = 186,582$
  - Số phần tử khác 0 ($nnz$): $\approx 4,981,696$
  - Độ thưa (Matrix Sparsity): $99.9110\%$ (chứng minh tính hiệu quả bắt buộc của biểu diễn ma trận thưa CSR).
- **Thực nghiệm 2 (Preprocessing Ablation):**
  - So sánh 3 Pipeline: A (Minimal), B (Normalized với stopword removal), C (Stemming).
  - Kết quả: Pipeline B cân bằng tối ưu giữa việc lọc nhiễu, giảm kích thước từ vựng và duy trì tính nguyên vẹn của ý nghĩa ngữ nghĩa.
- **Hệ thống tìm kiếm & Đánh giá (Search & Evaluation):**
  - Xây dựng thành công bộ máy tìm kiếm top-K dựa trên tích vô hướng ma trận thưa.
  - Đánh giá trên tập query chuẩn: Đạt MRR $> 0.85$ và Precision@5 cao trên các truy vấn trùng khớp từ vựng chính xác.
  - Phân tích sâu hiện tượng **Vocabulary Mismatch** dẫn tới nhu cầu tất yếu chuyển đổi sang Dense Word Embeddings & Contextual Embeddings (Transformer).
