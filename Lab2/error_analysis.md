# LAB 02 — PHÂN TÍCH LỖI (ERROR ANALYSIS)

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Chủ đề:** N-gram Language Models, Smoothing and Perplexity  
**Quy tắc:** Phân tích chi tiết 2 trường hợp dự đoán đúng và 2 trường hợp dự đoán sai từ thực nghiệm Mục 20 và Mục 22 (`W2.pdf`).

---

## 1. Hai trường hợp dự đoán đúng (Correct Predictions)

### Trường hợp 1: Cụm thuật ngữ chuyên ngành cố định
* **Context:** `natural language`
* **Model Prediction (Trigram):** `processing`
* **Expected Word:** `processing`
* **Xác suất mô hình:** $P = 3.7042 \times 10^{-5}$ (xếp vị trí số 1).
* **Nguyên nhân thành công:**
  - `natural language processing` là một cụm danh từ cố định (collocation) có tính kết dính ngữ nghĩa cực kỳ cao.
  - Trong tập ngữ liệu huấn luyện C4 (nhiều tài liệu học thuật và khoa học máy tính), tần suất đi liền nhau của bộ 3 từ này xuất hiện với mật độ vượt trội so với các từ khác.

### Trường hợp 2: Cấu trúc ngữ pháp phổ biến
* **Context:** `one of`
* **Model Prediction (Trigram):** `the`
* **Expected Word:** `the`
* **Xác suất mô hình:** $P = 2.2247 \times 10^{-2}$ (chiếm xác suất áp đảo ~2.22%).
* **Nguyên nhân thành công:**
  - `one of the` là một mẫu cấu trúc ngữ pháp cơ bản và cực kỳ phổ biến trong tiếng Anh.
  - Sự lặp lại với số lần xuất hiện rất lớn trên 550.000 câu huấn luyện giúp mô hình N-gram ước lượng phân phối xác suất rất tự tin và chính xác.

---

## 2. Hai trường hợp dự đoán sai (Incorrect Predictions)

### Trường hợp 3: Sai do quy tắc tách từ và lệch miền ngữ liệu
* **Context:** `the cat`
* **Model Prediction (Trigram):** `queen` ($P = 3.7042 \times 10^{-5}$) hoặc `s`
* **Expected Word:** `had` hoặc `sat`
* **Xác suất mô hình:** Xác suất cho từ đúng bị tụt xuống dưới top-3.
* **Nguyên nhân thất bại:**
  1. **Quy tắc Tokenization (Biểu thức chính quy):** Biểu thức `[a-zA-Z0-9]+` tách sở hữu cách `cat's` thành hai token rời rạc `cat` và `s`, vô tình biến token `s` thành một trong những từ đứng sau phổ biến nhất của `cat`.
  2. **Lệch miền ngữ liệu (Domain Mismatch) & Sparsity:** Dữ liệu C4 thu thập từ web với nhiều nội dung kỹ thuật, tin tức, bằng sáng chế; các câu chuyện thiếu nhi hoặc ngữ cảnh mèo vận động (`the cat sat`) rất hiếm, khiến mô hình thiếu dữ liệu đại diện.

### Trường hợp 4: Sai do ngữ cảnh quá ngắn và ưu thế từ dừng (Stopwords)
* **Context:** `machine learning`
* **Model Prediction (Trigram):** `and` ($P = 6.7214 \times 10^{-5}$)
* **Expected Word:** `models`, `algorithms`, hoặc `techniques`
* **Xác suất mô hình:** Từ dừng `and` chiếm vị trí số 1.
* **Nguyên nhân thất bại:**
  - **Ngữ cảnh 2 từ quá ngắn:** Cụm từ `machine learning` chưa đủ chi tiết để xác định ý định diễn đạt ngữ nghĩa chuyên sâu.
  - **Tần suất từ dừng áp đảo:** Từ nối `and` xuất hiện ở khắp mọi nơi trong ngôn ngữ, nên ngay cả sau cụm từ chuyên ngành, xác suất kết hợp với liên từ vẫn lấn át các danh từ kỹ thuật cụ thể.

---

## 3. Tổng kết bài học rút ra
* **Hạn chế của N-gram:** Chỉ dựa vào tần suất bề mặt và phụ thuộc nặng nề vào chất lượng tiền xử lý; không thể phân biệt giữa từ dừng và từ nội dung.
* **Nhu cầu chuyển dịch:** Cần các mô hình biểu diễn liên tục (**Dense Embeddings**) và cơ chế chú ý (**Self-Attention**) để hiểu ngữ cảnh sâu và liên kết ngữ nghĩa giữa các từ tương đồng.
