# Prediction Before Experiments — LAB 03

Họ và tên sinh viên: Nguyễn Trọng Thành  
Mã sinh viên: (Điền MSV của bạn)  
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I, 2026  
Mục tiêu: Đưa ra các dự đoán lý thuyết, giải thích nguyên nhân dựa trên Distributional Hypothesis và ấn định độ tin cậy TRƯỚC KHI chạy mô hình trong notebook.

---

## Prediction 1 — Nearest words trong không gian vector

**Tập từ khảo sát:** `doctor`, `physician`, `hospital`, `banana`, `car`.

### Prediction:
- Cặp từ có độ tương đồng cao nhất: `doctor` và `physician` (đồng nghĩa, cùng trường vai trò).
- Tiếp theo là `doctor` và `hospital` (quan hệ chủ đề liên kết / topical relatedness, nhưng mức độ tương đồng cosine sẽ thấp hơn cặp `doctor - physician`).
- `car` và `banana` sẽ có độ tương đồng rất thấp hoặc xấp xỉ $0$ đối với `doctor`.

### Reason:
- Theo **Distributional Hypothesis** (*"You shall know a word by the company it keeps"* — J.R. Firth):
  - `doctor` và `physician` có tính thay thế cho nhau cao trong câu (substitutable), cùng xuất hiện xung quanh các từ như `patient`, `treatment`, `prescribe`, `clinic`. Do đó vector ngữ cảnh của chúng sẽ gần như trùng hướng.
  - `hospital` là địa điểm, xuất hiện trong cùng chủ đề nhưng có vai trò cú pháp khác biệt (danh từ chỉ nơi chốn thay vì chủ thể hành động).
  - `car` (phương tiện giao thông) và `banana` (thực phẩm) thuộc các trường từ vựng hoàn toàn độc lập, hầu như không chia sẻ ngữ cảnh chung với `doctor`.

### Confidence:
- **Rất cao (95%)**

---

## Prediction 2 — Ảnh hưởng khi tăng Context Window ($2 \to 5$)

### Prediction:
- Khi tăng context window từ $2$ lên $5$:
  - Độ tương đồng giữa các từ có quan hệ **chủ đề rộng / liên tưởng ngữ nghĩa** (topical / associative relatedness, ví dụ `doctor` - `hospital`, `clinic` - `treatment`, `football` - `stadium`) sẽ **TĂNG LÊN**.
  - Ngược lại, tính khu biệt về **vai trò cú pháp / chức năng từ loại** (syntactic / functional similarity, ví dụ phân biệt nghiêm ngặt giữa danh từ chỉ người và danh từ chỉ nơi chốn, hay động từ cùng loại) sẽ **GIẢM NHẸ**.

### Reason:
- Cửa sổ nhỏ ($k=2$) tập trung vào quan hệ cú pháp cục bộ trực tiếp (ngay trước và sau từ, chẳng hạn vị ngữ đi liền với tân ngữ, tính từ bổ nghĩa danh từ).
- Cửa sổ lớn ($k=5$) bao quát toàn bộ câu hoặc mệnh đề, thu thập các từ đồng xuất hiện thuộc cùng một chủ đề (topic), biến vector thành biểu diễn đặc trưng cho lĩnh vực/chủ đề thay vì cấu trúc ngữ pháp cục bộ.

### Confidence:
- **Cao (90%)**

---

## Prediction 3 — Ảnh hưởng của Embedding Dimension ($50 \to 100 \to 300$)

### Prediction:
- Khi tăng kích thước vector từ $50 \to 100$: Chất lượng biểu diễn và độ mịn của không gian vector sẽ tăng, khả năng phân tách các cụm ngữ nghĩa rõ ràng hơn.
- Khi tăng tiếp từ $100 \to 300$: Trên một tập ngữ liệu nhỏ/vừa, chất lượng **KHÔNG CHẮC CHẮN TĂNG**, thậm chí có thể bị suy giảm do hiện tượng overfitting và không gian bị phân tán ngẫu nhiên (under-determined representation), trong khi thời gian huấn luyện tăng gấp nhiều lần.

### Reason:
- Đây là bài toán đánh đổi kinh điển giữa **Mô hình dung lượng (Model Capacity)** và **Quy mô dữ liệu (Data Size)**:
  - Vector 50 chiều có thể hơi chật chội để mã hóa nhiều khía cạnh ngữ nghĩa độc lập.
  - Vector 100 chiều thường là điểm cân bằng lý tưởng cho các tập ngữ liệu thực nghiệm quy mô vừa và nhỏ.
  - Vector 300 chiều có số lượng tham số lớn ($|V| \times 300$), nếu không có hàng triệu từ ngữ cảnh để tối ưu, các chiều dư thừa sẽ chỉ học các nhiễu thống kê ngẫu nhiên (noise), làm loãng độ tương đồng cosine.

### Confidence:
- **Cao (85%)**

---

## Prediction 4 — Khả năng hội tụ khi Corpus chỉ có 100 câu

### Prediction:
- Hai từ `doctor` và `physician` **KHÔNG CHẮC CHẮN GẦN NHAU**; thậm chí chúng có thể có độ tương đồng bằng $0$ (trong ma trận co-occurrence) hoặc mang giá trị tương đồng ngẫu nhiên, sai lệch trong Word2Vec.

### Reason:
- **Hiện tượng cực kỳ thưa dữ liệu (Extreme Data Sparsity):**
  - Trong 100 câu ngắn, từ thông dụng `doctor` có thể xuất hiện 2–3 lần, nhưng từ trang trọng/ít gặp hơn như `physician` có thể chỉ xuất hiện 0 hoặc 1 lần.
  - Khả năng để cả hai từ cùng xuất hiện trong các câu có chung từ ngữ cảnh (như `patient`, `treat`) là rất thấp. Nếu không có từ ngữ cảnh chung, ma trận co-occurrence sẽ cho hai vector trực giao ($\cos = 0$).
  - Với Word2Vec, 100 câu không đủ số lượng bước gradient descent để kéo hai vector lại gần nhau, các vector hầu như vẫn giữ nguyên trọng số khởi tạo ngẫu nhiên ban đầu.

### Confidence:
- **Rất cao (90%)**

---

## Prediction bổ sung (Mục 16 W3.pdf) — So sánh CBOW và Skip-gram

### 1. Sự khác biệt cơ bản về Input / Target
- **CBOW (Continuous Bag of Words):** Input là các từ ngữ cảnh xung quanh $\to$ Mục tiêu (Target) là dự đoán từ trung tâm.
- **Skip-gram:** Input là từ trung tâm $\to$ Mục tiêu (Target) là dự đoán từng từ ngữ cảnh xung quanh.

### 2. Liệt kê Training Examples thủ công
Xét câu: `"the cat eats fish"` với cửa sổ ngữ cảnh $k = 1$:

| Vị trí target | Target word | Context words ($k=1$) | CBOW training example `(Context -> Target)` | Skip-gram training examples `(Target -> Context)` |
|---|---|---|---|---|
| $t = 1$ | `the` | `['cat']` | `(['cat'] -> 'the')` | `('the', 'cat')` |
| $t = 2$ | `cat` | `['the', 'eats']` | `(['the', 'eats'] -> 'cat')` | `('cat', 'the')`, `('cat', 'eats')` |
| $t = 3$ | `eats` | `['cat', 'fish']` | `(['cat', 'fish'] -> 'eats')` | `('eats', 'cat')`, `('eats', 'fish')` |
| $t = 4$ | `fish` | `['eats']` | `(['eats'] -> 'fish')` | `('fish', 'eats')` |

> **Nhận xét lý thuyết:** CBOW tính trung bình cộng (average) các vector ngữ cảnh nên làm mịn nhiễu tốt, huấn luyện nhanh hơn trên các từ phổ biến. Skip-gram tạo ra nhiều cặp huấn luyện độc lập cho mỗi từ trung tâm, do đó học biểu diễn cho các từ hiếm (rare words) hiệu quả hơn đáng kể.
