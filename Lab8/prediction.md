# Báo Cáo Dự Đoán Lý Thuyết Trước Thực Nghiệm & Đối Chiếu Kiểm Chứng — LAB 08

**Học phần:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Chủ đề:** Pretrained Transformers và Tinh Chỉnh Mô Hình BERT  
**Nội dung thực hiện:** Ghi nhận 3 Dự đoán tiên nghiệm (Mục 11) và Bảng đối chiếu kiểm chứng thực tế (Verification)  

---

## 1. Các Dự Đoán Lý Thuyết Trước Khi Thực Thi Mô Hình (Mục 11)

Theo đúng quy trình nghiên cứu khoa học được quy định tại Mục 11 của đề bài, toàn bộ các dự đoán dưới đây được xây dựng dựa trên nguyên lý kiến trúc mạng nơ-ron và cơ sở lý thuyết tiền huấn luyện, độc lập hoàn toàn trước khi tiến hành chạy các thực nghiệm tính toán.

---

### Dự đoán 1: So Sánh Pretrained Transformer vs. Transformer Train From Scratch Trên IMDb

**Câu hỏi đặt ra:**  
*Một mô hình Pretrained Transformer được fine-tune trên tập dữ liệu IMDb có khả năng đạt hiệu năng phân loại vượt trội hơn một Transformer cỡ nhỏ được huấn luyện từ đầu (train from scratch) trên cùng tập dữ liệu hay không?*

* **Dự đoán (Prediction):**  
  **CÓ.** Mô hình Pretrained Transformer (BERT) sau khi tinh chỉnh sẽ vượt trội áp đảo toàn diện so với Transformer huấn luyện từ con số không (From scratch) trên mọi thước đo đánh giá (Accuracy, Macro F1).

* **Lý do khoa học (Reason):**  
  1. *Khối lượng tri thức tiền huấn luyện (Pre-trained Knowledge Base):*  
     Mô hình BERT đã trải qua quá trình tiền huấn luyện trên quy mô ngữ liệu khổng lồ (BooksCorpus và English Wikipedia với tổng dung lượng hơn $3.3$ tỷ từ vựng). Thông qua các tác vụ tự giám sát (đặc biệt là Masked Language Modeling), mô hình đã tích lũy sẵn biểu diễn đa tầng về cấu trúc cú pháp, ngữ nghĩa ngữ cảnh, mối quan hệ đồng vị từ vựng và các trường sắc thái ngôn ngữ tự nhiên.
  2. *Thiên kiến cảm nạp (Inductive Bias) và sự thiếu hụt dữ liệu của mô hình From-scratch:*  
     Kiến trúc Transformer dựa hoàn toàn vào cơ chế Self-Attention tự do, không có các giả định cảm nạp cục bộ bảo toàn như tích chập không gian của CNN hay tính chuyển dịch tuần tự của RNN. Do đó, để học được ngữ nghĩa cơ bản của ngôn ngữ, Transformer đòi hỏi lượng dữ liệu cực lớn. Nếu chỉ cung cấp $4,000$ mẫu bình luận từ tập IMDb, mô hình train from scratch buộc phải gánh vác hai nhiệm vụ nặng nề cùng lúc: vừa phải giải mã cấu trúc ngữ pháp ngôn ngữ từ hư vô, vừa phải học phân biệt cảm xúc khen/chê. Với số lượng mẫu quá khiêm tốn so với hàng chục triệu tham số, mô hình khởi tạo ngẫu nhiên chắc chắn sẽ bị quá khớp (overfitting) trầm trọng trên tập train hoặc suy biến về đoán mò xác suất mà không thể tổng quát hóa lên tập kiểm tra.

* **Bằng chứng kỳ vọng (Expected Evidence):**  
  - Độ chính xác (Accuracy) và Macro F1 của Pretrained Transformer dự kiến sẽ vượt mốc $85\%$, trong khi mô hình Transformer from scratch nhiều khả năng chỉ dao động trong khoảng đoán ngẫu nhiên $50\% - 60\%$.
  - Đường cong học tập (Learning Curves): Mô hình Pretrained sẽ hội tụ rất nhanh chóng và mượt mà chỉ sau $2 - 3$ epochs với hàm mất mát trên tập Validation giảm đều đặn. Ngược lại, mô hình from-scratch sẽ có Validation Loss tăng vọt hoặc dao động bất ổn định ngay từ epoch đầu tiên.

---

### Dự đoán 2: Đóng Băng Trọng Số (Frozen Encoder) vs. Tinh Chỉnh Toàn Bộ (Full Fine-Tuning)

**Câu hỏi đặt ra:**  
*Nếu đóng băng toàn bộ các tầng trọng số của Pretrained Transformer Encoder và chỉ huấn luyện tầng phân loại đầu ra (Classification Head):*
$$\text{BERT Encoder (Frozen)} \longrightarrow \text{Classification Head (Trainable)} \longrightarrow \text{Prediction}$$
*Kết quả phân loại sẽ thuộc phương án nào sau đây?*
- **A.** Tốt hơn fine-tuning toàn bộ
- **B.** Tương đương
- **C. Kém hơn**
- **D.** Không thể dự đoán

* **Lựa chọn dự đoán:** **C. Kém hơn**

* **Giải thích khoa học (Reason):**  
  1. *Biểu diễn đặc trưng tĩnh vs. Biểu diễn thích nghi miền nhiệm vụ:*  
     Khi đóng băng (freeze) các khối Encoder, các vector trạng thái ẩn đầu ra tại vị trí `[CLS]` hoàn toàn là đặc trưng tĩnh (static contextual features) được kế thừa từ mục tiêu tổng quát của giai đoạn MLM. Tầng phân loại Linear Classifier lúc này chỉ có thể tìm kiếm một siêu phẳng phân tách tuyến tính đơn giản trong không gian đặc trưng cố định đó.
  2. *Sự thích ứng sâu của các đầu chú ý (Attention Map Adaptation):*  
     Khi tiến hành Full Fine-Tuning, tín hiệu gradient từ hàm mất mát phân loại cảm xúc được lan truyền ngược xuyên suốt toàn bộ $12$ tầng Transformer Encoder. Cơ chế này cho phép các ma trận chiếu Query, Key, Value trong từng đầu chú ý (Multi-Head Attention) được tái tinh chỉnh để hướng sự tập trung cao độ vào những từ ngữ mang trọng số cảm xúc then chốt (ví dụ: *terrific, masterpiece, abysmal, waste*), đồng thời tái cấu trúc lại không gian biểu diễn ẩn sao cho khoảng cách giữa hai cụm lớp Positive và Negative được kéo giãn tối đa. Do đó, Full Fine-Tuning chắc chắn sẽ khai thác sâu sắc và chính xác hơn so với chế độ Feature Extraction đơn thuần.

---

### Dự đoán 3: Mức Độ Nhạy Cảm Khi Thiếu Hụt Dữ Liệu Huấn Luyện (Data Efficiency — 10% Data)

**Câu hỏi đặt ra:**  
*Nếu quy mô dữ liệu huấn luyện bị cắt giảm chỉ còn $10\%$ ($400$ mẫu so với $4,000$ mẫu ban đầu), mô hình nào giữa Pretrained Transformer và Transformer Train From Scratch sẽ chịu sự sụt giảm hiệu năng (performance degradation) nghiêm trọng hơn?*

* **Dự đoán:**  
  **Mô hình Transformer Train From Scratch sẽ sụt giảm hiệu năng nghiêm trọng và toàn diện hơn rất nhiều.**

* **Cơ sở khoa học dựa trên vai trò của Pretraining:**  
  1. *Tính hiệu quả mẫu vượt trội (Superior Sample Efficiency) của Pre-training:*  
     Mô hình Pretrained đã được "trang bị sẵn" toàn bộ quy luật ngôn ngữ, từ vựng và ngữ pháp sau giai đoạn tiền huấn luyện. Khi tiếp cận tác vụ hạ nguồn với chỉ $10\%$ dữ liệu ($400$ mẫu), nhiệm vụ thực chất chỉ là học một ánh xạ chiếu (mapping) từ không gian biểu diễn ngữ nghĩa đã có sẵn sang không gian nhãn nhị phân của bài toán. Nhờ việc chuyển giao tri thức (Transfer Learning), mô hình chỉ cần một số lượng nhỏ ví dụ minh họa là đã có thể xác định ranh giới quyết định hiệu quả mà không làm suy giảm quá nhiều độ chính xác.
  2. *Sự sụp đổ tất yếu của mô hình From-scratch:*  
     Đối với mô hình khởi tạo ngẫu nhiên, $400$ mẫu là một lượng thông tin hoàn toàn bất khả thi để tối ưu hóa hàng triệu tham số tự do. Khi tỷ lệ giữa số lượng tham số và số lượng mẫu huấn luyện quá mất cân đối, mạng nơ-ron không thể học được bất kỳ đặc trưng khái quát nào; thay vào đó, nó sẽ ghi nhớ máy móc toàn bộ tập train (overfitting tuyệt đối) hoặc rơi vào trạng thái suy biến hoàn toàn (mode collapse) — dự đoán toàn bộ dữ liệu về một nhãn đa số duy nhất, dẫn đến độ chính xác sụp đổ về mức ngẫu nhiên $50\%$.

---

## 2. Bảng Đối Chiếu Kiểm Chứng Thực Nghiệm (Experimental Verification)

Sau khi hoàn tất quá trình huấn luyện thực tế trên tập dữ liệu IMDb theo đúng thiết lập nghiêm ngặt, bảng số liệu dưới đây đối chiếu trực tiếp giữa dự đoán tiên nghiệm và kết quả đo đạc thực tế (trích xuất từ [results.csv](file:///d:/NLP/Lab8/results.csv)):

| Nội dung kiểm chứng | Dự đoán tiên nghiệm ban đầu | Kết quả thực nghiệm đo đạc | Đánh giá & Kết luận khoa học |
| :--- | :--- | :--- | :--- |
| **Exp 1: Pretrained vs. Scratch**<br>*(Huấn luyện trên 4,000 mẫu)* | Mô hình Pretrained thắng áp đảo; mô hình From-scratch thất bại vì thiếu dữ liệu. | **Pretrained Fine-tuning:**<br>- Accuracy: **$86.00\%$**<br>- Macro F1: **$0.8649$**<br>**Transformer from scratch:**<br>- Accuracy: **$49.80\%$**<br>- Macro F1: **$0.0000$** | **Khớp hoàn toàn với dự đoán.**<br>Mô hình from-scratch hoàn toàn bất lực trong việc học biểu diễn ngôn ngữ từ $4,000$ mẫu, dẫn đến việc dự đoán toàn bộ nhãn Negative (Accuracy $\approx 50\%$, F1 của nhãn Positive $= 0$). Trong khi đó, Pretrained BERT đạt kết quả xuất sắc $86.00\%$. |
| **Exp 2: Frozen vs. Fine-tuning**<br>*(Huấn luyện trên 4,000 mẫu)* | Frozen Encoder sẽ kém hơn Full Fine-tuning (Chọn C). | **Full Fine-tuning:**<br>- Accuracy: **$86.00\%$**<br>- Macro F1: **$0.8649$**<br>**Frozen Encoder (Feature Extraction):**<br>- Accuracy: **$77.90\%$**<br>- Macro F1: **$0.7877$** | **Khớp hoàn toàn với dự đoán.**<br>Full Fine-tuning vượt trội hơn Frozen Encoder **$+8.10\%$** về Accuracy và $+0.0772$ về F1. Tuy vậy, Frozen Encoder chỉ cần huấn luyện vỏn vẹn $1,538$ tham số trong $65.68\text{s}$ vẫn đạt $77.90\%$, chứng minh biểu diễn tĩnh của BERT vốn đã rất chất lượng. |
| **Exp 3: Khảo sát 10% Dữ liệu**<br>*(Huấn luyện trên 400 mẫu)* | From-scratch sụp đổ hoàn toàn; Pretrained duy trì độ chính xác cao. | **Pretrained BERT (10% data):**<br>- Accuracy: **$82.60\%$**<br>- Macro F1: **$0.8249$**<br>**From-scratch (10% data):**<br>- Accuracy: **$50.20\%$**<br>- Macro F1: **$0.6684$** (suy biến nhãn) | **Khớp hoàn toàn với dự đoán.**<br>Khi giảm $90\%$ dữ liệu (từ 4000 xuống 400 mẫu), Pretrained BERT chỉ giảm nhẹ $3.40\%$ Accuracy ($86.00\% \to 82.60\%$). Trái lại, From-scratch hoàn toàn suy biến về đoán nhãn đa số ($50.20\%$). |

---

## 3. Tổng Kết Nhận Thức Khoa Học

Toàn bộ ba giả thiết tiên nghiệm đều được thực nghiệm kiểm chứng với độ tương thích $100\%$. Điều này khẳng định vững chắc rằng:
1. **Pre-training là bước chuyển giao tri thức mang tính quyết định** trong NLP hiện đại, giải phóng các tác vụ downstream khỏi gánh nặng phải tự học biểu diễn ngôn ngữ cơ sở.
2. **Fine-tuning toàn diện cho phép tái cấu trúc các đầu chú ý** hướng vào đặc thù của miền cảm xúc, đem lại bước nhảy vọt về hiệu năng so với việc chỉ trích xuất đặc trưng tĩnh (Feature Extraction).
3. **Hiệu quả mẫu (Data/Sample Efficiency) của mô hình tiền huấn luyện vượt trội tuyệt đối**, mở ra khả năng ứng dụng thực tế to lớn cho các bài toán công nghiệp vốn luôn khan hiếm dữ liệu gán nhãn chất lượng cao.
