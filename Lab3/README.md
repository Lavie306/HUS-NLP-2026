# LAB 03 — Word Representations and Embeddings: From Sparse Representations to Dense Word Embeddings

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I, 2026  
**Đơn vị:** Trường Đại học Khoa học Tự nhiên, ĐHQGHN (VNU-HUS)  
**Sinh viên thực hiện:** Nguyễn Trọng Thành  

---

## 1. Cấu trúc thư mục nộp bài (Deliverables theo Mục 30 W3.pdf)

Thư mục đã được hoàn thiện đầy đủ 100% theo đúng đặc tả yêu cầu trong tài liệu `W3.pdf`:

```text
Lab3/
├── README.md              # Báo cáo tổng quan, hướng dẫn chạy và tường trình AI Policy
├── calculations.md        # Lời giải tính tay chi tiết Bài 1 - 4 & bài tập Analogy
├── prediction.md          # Dự đoán trước thực nghiệm theo Distributional Hypothesis
├── cooccurrence.py        # Mã nguồn cốt lõi (Vocabulary, Co-occurrence matrix, Cosine, Most-similar)
├── experiments.ipynb      # Notebook thực nghiệm hoàn chỉnh (Word2Vec sweep, Analogy, Semantic Search)
├── word_embedding.ipynb   # Notebook rút gọn tập trung vào bước chuyển Sparse -> Dense
├── results.csv            # Bảng tổng hợp toàn bộ số liệu đo đạc thực nghiệm
├── error_analysis.md      # Phân tích 3 trường hợp đúng & 3 trường hợp sai/bất ngờ kèm corpus evidence
└── reflection.md          # Ma trận so sánh 4 dạng biểu diễn, phân tích đa nghĩa 'bank' & oral check
```

---

## 2. Tóm tắt các kết quả chính

### A. Lý thuyết & Tính toán (`calculations.md`)
- **Bài 1:** Xây dựng ma trận co-occurrence ($k=1$). Minh chứng hai từ `cat` và `dog` có vector ngữ cảnh trùng khít (`[0, 0, 1, 1, 0, 0, 0]`) do chia sẻ cùng phân bố từ ngữ cảnh (`eats`, `likes`), minh họa trực quan giả thuyết Distributional Hypothesis.
- **Bài 2:** Hai vector $x = [1, 2, 1]$ và $y = [2, 4, 2]$ cùng phương cho $\cos(x, y) = 1.0$. Cosine similarity triệt tiêu ảnh hưởng của độ lớn/tần suất, chỉ đo hướng tương đối của ngữ cảnh.
- **Bài 3:** $\cos(\text{doctor}, \text{physician}) \approx 0.9871$ trong khi $\cos(\text{doctor}, \text{banana}) \approx -0.1414$, phân biệt chính xác từ đồng nghĩa y tế và từ không liên quan.
- **Bài 4:** Đánh giá trade-off Sparse ($10,000$ chiều) vs Dense ($300$ chiều). Khẳng định dense không tuyệt đối tốt hơn trong mọi bài toán (cần cân nhắc tính diễn giải, lượng dữ liệu huấn luyện, và bài toán exact matching).
- **Bài Analogy:** Thực hiện $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} = [8, 4, 7] \approx \vec{v}_{\text{queen}}$.

### B. Thực nghiệm lập trình (`cooccurrence.py` & `experiments.ipynb`)
- Đã cài đặt thư viện `gensim 4.4.0` vào môi trường Python.
- Chạy sweep đầy đủ tổ hợp siêu tham số Word2Vec: Context Window $\in \{2, 5, 10\}$ kết hợp Vector Dimension $\in \{50, 100, 300\}$.
- Triển khai thành công ứng dụng **Semantic Search** cho câu truy vấn `medical treatment`, xếp hạng chính xác các văn bản liên quan đến y tế/lâm sàng lên top đầu.

---

## 3. Hướng dẫn chạy kiểm thử

### Kiểm tra script độc lập:
```powershell
python -m py_compile cooccurrence.py
python cooccurrence.py
```
*(Kết quả: Tất cả self-tests đều vượt qua 100%).*

### Chạy Jupyter Notebooks:
Mở `experiments.ipynb` hoặc `word_embedding.ipynb` trong VS Code / Jupyter Lab và thực thi tuần tự các cell từ trên xuống dưới. Mọi ô lệnh đã được chạy sẵn và lưu đầy đủ biểu đồ, bảng biểu và kết quả đo đạc.

---

## 4. Tuyên bố sử dụng AI (AI Assistance Statement)

Tuân thủ nghiêm ngặt theo quy định mục 28 `W3.pdf`:
- **Công cụ hỗ trợ:** Antigravity / Gemini.
- **Mục đích được phép:** Hỗ trợ kiểm tra cấu trúc mã nguồn, giải thích cú pháp API Gensim Word2Vec, rà soát tính hợp lệ của môi trường chạy, và tối ưu hóa định dạng báo cáo Markdown / LaTeX.
- **Nội dung do sinh viên chịu trách nhiệm:** Toàn bộ các bước giải toán trung gian, lý giải hiện tượng Distributional Hypothesis, dự đoán trước thực nghiệm, số liệu kiểm chứng, phân tích lỗi ngữ liệu và chuẩn bị trả lời vấn đáp cá nhân.
