# Error Analysis — LAB 03: Word Representations and Embeddings

Họ và tên sinh viên: Nguyễn Trọng Thành  
Mã sinh viên: (Điền MSV của bạn)  
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I, 2026  
Mục tiêu: Phân tích định tính và định lượng 3 trường hợp dự đoán/đo đạc chính xác và 3 trường hợp sai lệch/bất ngờ, kèm minh chứng ngữ liệu (corpus evidence) cụ thể thay vì chỉ đưa ra nhận định chung chung.

---

## Bảng phân tích chi tiết các trường hợp thực nghiệm

| Case | Observed (Quan sát) | Expected (Kỳ vọng) | Possible Explanation (Nguyên nhân) | Evidence from Corpus (Minh chứng ngữ liệu) |
|---|---|---|---|---|
| **Correct 1** | `cat - dog` có Cosine = **1.0000** trong ma trận Co-occurrence ($k=1, 2$). | Cosine rất cao ($\approx 1.0$) do cùng là vật nuôi và cùng đóng vai trò cú pháp tương đương. | **Distributional Hypothesis hoàn hảo**: Hai từ có phân bố ngữ cảnh bậc 2 (second-order context) trùng khớp $100\%$, dù không cùng xuất hiện trong một câu. | Câu (1): `the cat eats fish`<br>Câu (2): `the dog eats fish`<br>Câu (3): `the cat likes milk`<br>Câu (4): `the dog likes meat`<br>$\implies$ Cả hai đều có ngữ cảnh $k=1$ là `eats` và `likes`. |
| **Correct 2** | `car - automobile` có Cosine = **0.8660** ($k=1$) và **0.8250** ($k=5$) trong Co-occurrence. | Cosine cao ($> 0.8$) vì đây là hai từ đồng nghĩa chỉ phương tiện giao thông. | **Đồng xuất hiện cục bộ nhất quán**: Cả hai từ đều chia sẻ từ ngữ cảnh cốt lõi `driver` và mạo từ `the`, đồng thời thuộc cùng trường từ vựng vận tải. | Câu 10: `the car driver travelled to the city by road`<br>Câu 11: `the automobile driver drove the car along the highway`<br>$\implies$ Cả hai đều đi liền trước `driver`. |
| **Correct 3** | **Semantic Search** với truy vấn `medical treatment` trả về Top 1 document có Cosine = **0.6113**. | Văn bản y tế chứa chẩn đoán, điều trị phải được xếp hạng cao nhất; các văn bản thể thao/hoa quả phải ở cuối. | **Mean Embedding gom cụm ngữ nghĩa chuẩn xác**: Phép lấy trung bình vector từ vựng bắt được đúng các tài liệu thuộc miền y tế/trị liệu lâm sàng. | Top 1 Doc: `the clinic provides medical treatment for disease`<br>Top 2 Doc: `the computer stores medical records and clinical data`<br>Top 3 Doc: `the doctor and physician discuss the therapy for clinical care`<br>Các câu về `football` và `banana` có score $< 0.1$. |
| **Surprising 1** | `doctor - banana` có Cosine = **0.8660** ở ma trận Co-occurrence khi $k=1$. | Cosine phải xấp xỉ $0$ hoặc rất thấp vì bác sĩ và quả chuối thuộc hai miền ngữ nghĩa hoàn toàn xa lạ. | **Nhiễu từ dừng (Stopword Noise & Small Corpus)**: Không loại bỏ stopwords khiến mạo từ `the` xuất hiện liền trước cả hai từ, tạo ra sự trùng khớp ngữ cảnh giả tạo khi cửa sổ quá hẹp. | Câu 1: `the doctor treated the patient...`<br>Câu 12: `the banana and apple were fresh...`<br>$\implies$ Cả `doctor` và `banana` đều có ngữ cảnh bên trái là `the`. Khi nới rộng window lên $k=5$, các từ nội dung khác xuất hiện kéo cosine giảm xuống **0.4330**. |
| **Surprising 2** | `doctor - physician` trong Word2Vec ($w=5, d=100$) cho Cosine = **-0.1351** (giá trị âm). | Phải có độ tương đồng dương rất cao ($> 0.7$) vì là hai từ đồng nghĩa cốt lõi trong y tế. | **Cực kỳ thưa dữ liệu & Huấn luyện chưa đủ (Extreme Sparsity & Insufficient Training)**: Không gian vector 100 chiều quá lớn so với ngữ liệu 16 câu. Số lần lặp gradient quá ít không đủ để vượt qua độ lệch ngẫu nhiên lúc khởi tạo (random initialization). | Cả `doctor` và `physician` chỉ xuất hiện 2 lần trong toàn bộ 16 câu. Mạng nơ-ron không đủ mẫu dữ liệu (pairs) để tối ưu hàm mục tiêu kéo hai vector lại gần nhau. |
| **Surprising 3** | Phép toán tương đồng vector `king - man + woman` trả về Top 1 là `software` (score 0.2177) thay vì `queen`. | Phải trả về `queen` theo tính chất đại số nổi tiếng $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$. | **Quy mô ngữ liệu hạn chế (Data Hunger of Neural Embeddings)**: Cấu trúc hình học tuyến tính (linear substructure) chỉ xuất hiện khi Word2Vec được huấn luyện trên hàng trăm triệu từ. Trên tập dữ liệu nhỏ, phép trừ/cộng vector chỉ là phép toán trên các tọa độ ngẫu nhiên mang tính nhiễu. | Từ `queen` chỉ xuất hiện đúng 1 lần trong câu: `the king ruled the royal kingdom with the wise queen`. Là một từ đơn độc (singleton), vector của nó chưa được định hình không gian rõ ràng so với các từ còn lại. |

---

## Đúc kết bài học thực nghiệm

1. **Về Co-occurrence Count-based:**
   - Hoạt động cực kỳ hiệu quả và trực quan trên các tập dữ liệu nhỏ khi các từ chia sẻ cùng cấu trúc ngữ pháp.
   - Rất dễ bị tổn thương bởi các từ dừng (stopwords) như `the`, `and`, `at`. Cần thiết phải áp dụng kỹ thuật lọc từ dừng (stopword removal) hoặc chuyển đổi trọng số PMI/PPMI (Pointwise Mutual Information) thay vì dùng tần suất đếm thô (raw counts).
2. **Về Word2Vec (Neural Embeddings):**
   - Word2Vec là mô hình **"đói dữ liệu" (data-hungry)**. Trên các tập dữ liệu cực nhỏ, Word2Vec không thể phát huy sức mạnh mà thậm chí còn kém hơn đếm tần suất truyền thống do hiện tượng under-trained.
   - Các tính chất nổi tiếng như Word Analogy ($\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$) là pattern nảy sinh từ dữ liệu khổng lồ (emergent properties from big data), hoàn toàn không phải mô hình "hiểu" ngữ nghĩa như tư duy con người.
