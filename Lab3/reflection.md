# Reflection — LAB 03: Word Representations and Embeddings

Họ và tên sinh viên: Nguyễn Trọng Thành  
Mã sinh viên: (Điền MSV của bạn)  
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I, 2026  
Mục tiêu: Đúc kết mạch phát triển từ biểu diễn rời rạc/tĩnh đến biểu diễn động theo ngữ cảnh (Contextual Embeddings / Transformers) và chuẩn bị kiểm tra vấn đáp cá nhân (Oral Check).

---

## 1. Representation Comparison Matrix

Bảng đối chiếu 4 thế hệ biểu diễn từ trong Xử lý ngôn ngữ tự nhiên:

| Representation | Context-dependent? (Phụ thuộc ngữ cảnh câu?) | Sparse / Dense? (Thưa hay Dày đặc?) | Can one word have multiple vectors? (Một từ có nhiều vector không?) |
|---|---|---|---|
| **TF-IDF** | **Không** *(Đo tầm quan trọng của từ trong văn bản, không xét ngữ cảnh cục bộ)* | **Sparse** *(Số chiều bằng kích thước từ vựng $|\mathcal{V}|$, hầu hết bằng 0)* | **Không** *(Mỗi từ chỉ là 1 chỉ số/trọng số trong không gian từ vựng)* |
| **Co-occurrence Matrix** | **Không** *(Tổng hợp tần suất đếm trên toàn bộ tập ngữ liệu)* | **Sparse** *(Ma trận $|\mathcal{V}| \times |\mathcal{V}|$, độ thưa thường $> 90\%$)* | **Không** *(Mỗi từ tương ứng với 1 hàng duy nhất trong ma trận)* |
| **Word2Vec (CBOW / Skip-gram)** | **Không** *(Biểu diễn tĩnh - Static Embedding)* | **Dense** *(Không gian vector liên tục số chiều thấp: $d \in [50, 300]$, các phần tử khác 0)* | **Không** *(Mỗi từ có đúng 1 vector duy nhất trong bảng tra cứu lookup table)* |
| **Contextual Embedding (BERT / Transformer)** | **CÓ** *(Vector được tính động dựa trên toàn bộ câu qua cơ chế Self-Attention)* | **Dense** *(Vector số chiều cố định: $d \in [768, 1024]$, dày đặc và liên tục)* | **CÓ** *(Cùng một từ nhưng ở các câu khác nhau sẽ có các vector hoàn toàn khác nhau)* |

---

## 2. Vì sao từ `bank` cần biểu diễn theo ngữ cảnh (Contextual Representation)?

### Phân tích hiện tượng đa nghĩa (Polysemy)
Xét hai câu ví dụ mẫu trong tài liệu `W3.pdf`:
1. *"I deposited money in the bank."* $\to$ `bank` mang nghĩa **tổ chức tài chính / ngân hàng** (financial institution).
2. *"We sat on the river bank."* $\to$ `bank` mang nghĩa **bờ sông / vùng đất ven nước** (geographical slope/riverbank).

### Giới hạn không thể vượt qua của Static Word Embeddings (Word2Vec)
- Trong các mô hình tĩnh như Word2Vec hay GloVe, kiến trúc mô hình duy trì một từ điển cố định và một ma trận trọng số $\mathbf{W} \in \mathbb{R}^{|\mathcal{V}| \times d}$. Mỗi từ chỉ được gán **duy nhất một vector điểm cố định trong không gian**.
- Khi huấn luyện trên một tập dữ liệu lớn chứa cả hai nghĩa trên:
  - Hàm mục tiêu sẽ cố gắng kéo vector của `bank` lại gần các từ tài chính (`money`, `deposit`, `interest`, `account`).
  - Đồng thời, nó cũng bị kéo lại gần các từ thiên nhiên/địa lý (`river`, `water`, `flow`, `mud`).
  - Kết quả là vector của `bank` sẽ bị rơi vào vị trí **trung bình cộng (superposition/compromise)** nằm lơ lửng ở giữa hai cụm ngữ nghĩa. Vector này không đại diện chính xác cho một nghĩa cụ thể nào cả. Tệ hơn, nếu một nghĩa xuất hiện áp đảo (ví dụ 90% nghĩa ngân hàng), vector sẽ bị thiên lệch hoàn toàn, làm triệt tiêu nghĩa thứ hai (dominant sense bias).

### Giải pháp đột phá từ Transformer / BERT
- Các kiến trúc Transformer hiện đại không dùng vector tĩnh để đại diện cho từ. Thay vào đó, chúng đưa từ qua các tầng **Multi-Head Self-Attention**.
- Cơ chế Attention cho phép token `bank` "chú ý" (attend) đến các token xung quanh nó trong từng câu cụ thể:
  - Khi gặp `money` và `deposit`, biểu diễn của `bank` được chiếu vào không gian tài chính.
  - Khi gặp `river` và `sat`, biểu diễn của `bank` được chiếu vào không gian địa lý.
- Đây chính là bước chuyển dịch mang tính cách mạng từ **Static Embeddings (Lab 03)** sang **Contextual Representations (Lab 04 - Transformers)**.

---

## 3. Chuẩn bị câu hỏi vấn đáp cá nhân (Individual Learning Check)

Dưới đây là câu trả lời ngắn gọn, chuẩn xác cho 6 câu hỏi ngẫu nhiên trong Section 29 của `W3.pdf` để sinh viên tự tin trả lời vấn đáp trong 3 phút:

### Câu 1: Distributional Hypothesis là gì?
> **Trả lời:** Distributional Hypothesis là giả thuyết ngôn ngữ học cốt lõi do J.R. Firth đề xuất (1957): *"Từ ngữ được hiểu thông qua các từ thường đi cùng với nó"* (*"You shall know a word by the company it keeps"*). Nói cách khác, các từ xuất hiện trong những phân bố ngữ cảnh tương tự nhau thì sẽ có ý nghĩa ngữ nghĩa tương đồng nhau. Đây là nền tảng toán học để chuyển ngữ cảnh thành vector trong NLP.

### Câu 2: Tại sao `doctor` và `physician` có thể gần nhau trong không gian vector?
> **Trả lời:** Dù hai từ này có thể không bao giờ xuất hiện chung trong cùng một câu, nhưng chúng là từ đồng nghĩa và có thể thay thế cho nhau (substitutable). Do đó, chúng cùng xuất hiện với các từ ngữ cảnh tương tự (như `patient`, `hospital`, `treatment`, `prescribe`). Nhờ hiện tượng đồng xuất hiện bậc hai (second-order co-occurrence), vector biểu diễn của chúng sẽ được kéo về cùng một hướng, dẫn đến độ tương đồng cosine rất cao.

### Câu 3: CBOW khác Skip-gram ở đâu?
> **Trả lời:** Khác biệt cốt lõi nằm ở chiều mũi tên bài toán dự đoán (Input vs Target):
> - **CBOW (Continuous Bag of Words):** Lấy tập hợp các từ ngữ cảnh xung quanh làm input để dự đoán một từ trung tâm duy nhất (`Context -> Target`). Huấn luyện nhanh, bắt tốt các từ thông dụng.
> - **Skip-gram:** Lấy từ trung tâm làm input để dự đoán lần lượt từng từ ngữ cảnh xung quanh (`Target -> Context`). Học biểu diễn rất tốt cho các từ hiếm (rare words) nhưng tốn thời gian tính toán hơn.

### Câu 4: Tại sao tăng context window có thể vừa tốt vừa xấu?
> **Trả lời:**
> - **Mặt tốt:** Cửa sổ lớn ($k=5, 10$) giúp thu thập ngữ cảnh chủ đề rộng (topical/associative context), giúp nhận diện các từ thuộc cùng lĩnh vực kiến thức ngay cả khi khoảng cách từ xa nhau.
> - **Mặt xấu:** Cửa sổ quá lớn sẽ làm loãng thông tin cú pháp cục bộ (syntactic/functional structure), gom cả những từ không liên quan ở các mệnh đề khác vào ngữ cảnh, làm giảm độ chuẩn xác khi phân biệt từ loại hoặc quan hệ ngữ pháp trực tiếp.

### Câu 5: Tại sao Word2Vec không phân biệt được hai nghĩa của từ `bank`?
> **Trả lời:** Vì Word2Vec là biểu diễn tĩnh (static embedding). Trong kiến trúc Word2Vec, bảng tra cứu chỉ cấp phát duy nhất 1 vector cố định cho mỗi chỉ số từ vựng. Khi gặp từ đa nghĩa như `bank` (ngân hàng vs bờ sông), mô hình buộc phải tối ưu vector này thành giá trị trung bình giữa hai ngữ cảnh hoàn toàn trái ngược, khiến vector bị mất đi tính chính xác cho từng ngữ cảnh cụ thể.

### Câu 6: Tại sao TF-IDF không phải là word embedding?
> **Trả lời:**
> - **TF-IDF** là biểu diễn **thưa (sparse)** và có số chiều rất lớn bằng toàn bộ kích thước từ vựng $|\mathcal{V}|$ (hoặc số lượng văn bản), trong đó mỗi chiều độc lập và trực giao với nhau, không nắm bắt được mối liên hệ ngữ nghĩa giữa các từ đồng nghĩa (ví dụ vector của `doctor` và `physician` trong TF-IDF có tích vô hướng bằng 0 nếu không chung từ khóa).
> - **Word embedding** là biểu diễn **dày đặc (dense)** trong không gian liên tục số chiều thấp ($d \ll |\mathcal{V}|$), trong đó các đặc trưng ngữ nghĩa được nén vào các chiều ẩn chung, cho phép đo đạc trực tiếp khoảng cách ngữ nghĩa liên tục giữa các từ.
