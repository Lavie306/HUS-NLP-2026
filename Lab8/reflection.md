# Báo Cáo Phản Tư Học Thuật & Chuyển Tiếp Kiến Thức (Reflection) — LAB 08

**Học phần:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Chủ đề:** Pretrained Transformers and BERT Fine-Tuning  
**Nội dung phản tư:** Vì sao trong thực tiễn người ta không bắt đầu bằng việc huấn luyện Transformer từ đầu?  

---

## Bài Luận Phản Tư (Dung Lượng: 300 – 500 Từ)

Sau khi tự tay xây dựng Transformer Encoder từ các tầng nguyên thủy ở LAB 07, bước sang LAB 08, thực nghiệm đã mang lại bài học nền tảng của NLP hiện đại: **Trong thực tế ứng dụng, việc huấn luyện một Transformer từ đầu (train from scratch) hầu như không bao giờ là giải pháp tối ưu hay khả thi.**

Quy luật này được định hình bởi sáu khía cạnh cốt lõi sau:

1. **Khía cạnh Dữ liệu (Data):** Kiến trúc Transformer thiếu các thiên kiến cảm nạp cục bộ (inductive biases) như tính bất biến tịnh tiến của CNN hay tính tuần tự của RNN. Để các ma trận Self-Attention học được cấu trúc cú pháp và phân phối ngôn ngữ, mô hình đòi hỏi kho dữ liệu khổng lồ hàng tỷ từ vựng. Trong thực tế, các bộ dữ liệu gán nhãn chuyên biệt thường chỉ có vài nghìn mẫu. Kết quả từ Thí nghiệm 1 và 3 đã chứng minh: ép Transformer học từ đầu trên dữ liệu nhỏ dẫn tới sự sụp đổ hiệu năng (Accuracy chỉ đạt $49.8\%$, tương đương đoán ngẫu nhiên).

2. **Biểu diễn Tiền Huấn Luyện (Pretrained Representation):** Nhờ mục tiêu tự giám sát Masked Language Modeling (MLM), mô hình BERT đã tích lũy tri thức phong phú từ Wikipedia và BooksCorpus. Không gian biểu diễn này cung cấp điểm xuất phát tối ưu, nơi các khái niệm từ vựng đã được phân cụm rõ nét ngay tại vector ẩn `[CLS]`.

3. **Chi phí Tính toán (Computational Cost):** Giai đoạn Pre-training đòi hỏi cụm hàng chục GPU/TPU vận hành liên tục nhiều tuần, tiêu tốn hàng trăm nghìn USD cùng lượng điện năng lớn. Việc tái sử dụng các checkpoint mở giúp tiết kiệm tới hơn $99\%$ chi phí điện toán.

4. **Tác vụ Đích Hạ Nguồn (Downstream Task):** Thay vì phân tán tài nguyên để vừa học ngữ pháp cơ bản vừa phân loại, mô hình có thể tập trung toàn bộ tham số vào việc nhận diện quy luật đặc thù của tác vụ đích.

5. **Sức Mạnh của Fine-Tuning:** Quá trình tinh chỉnh toàn phần chỉ cần $2 - 3$ epochs trên GPU cá nhân trong vài phút để đạt độ chính xác trên $86\%$. Tín hiệu gradient giúp các đầu chú ý nhanh chóng tái định hình để bắt trọn đặc trưng cảm xúc.

6. **Giới Hạn & Dịch Chuyển Miền (Limitation & Domain Shift):** Pretrained Transformer không phải là vạn năng. Mô hình vẫn chịu ràng buộc độ phức tạp $\mathcal{O}(T^2)$ khi tăng độ dài chuỗi, hiện tượng tự tin thái quá vào dự đoán sai (overconfidence), nhầm lẫn trước thủ pháp trào phúng, và suy giảm hiệu năng khi gặp dịch chuyển miền dữ liệu (Domain Shift).

---

## Cầu Nối Chuyển Tiếp Sang LAB 09 (Bridge to LAB 09)

Mạch kiến thức của **LAB 08** khép lại với bài toán Phân loại chuỗi văn bản (Sequence Classification):
$$\text{Pretrained Transformer} \longrightarrow \text{Fine-Tuning} \longrightarrow \text{Text Classification}$$
Ở thiết lập này, chúng ta trích xuất duy nhất một vector đại diện tại vị trí `[CLS]` để gán **một nhãn cảm xúc duy nhất cho toàn bộ câu**.

Tuy nhiên, trong các bài toán thực tiễn phức tạp hơn như Nhận dạng thực thể tên riêng (Named Entity Recognition - NER) hay Trích xuất thông tin y sinh, ta cần xác định danh tính của **từng từ ngữ trong câu**:
$$\text{"Barack Obama visited Paris."} \implies \text{Barack Obama} \to \textbf{PERSON}, \; \text{Paris} \to \textbf{LOCATION}$$

Do đó, **LAB 09** sẽ kế thừa nền tảng Pretrained Transformer nhưng chuyển dịch mô thức phân loại:
$$\textbf{Sequence Classification} \longrightarrow \textbf{Token Classification} \longrightarrow \textbf{Named Entity Recognition (NER)}$$
tận dụng toàn bộ các vector trạng thái ẩn $h_1, h_2, \dots, h_T$ tại mọi vị trí token của Transformer.
