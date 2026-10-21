# DỰ ĐOÁN TRƯỚC THỰC NGHIỆM (PRE-EXPERIMENT PREDICTIONS) — LAB 05

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I, 2026  
**Căn cứ tài liệu:** Mục 10 trong `W5.pdf`

Theo quy định của bài thực hành, toàn bộ các dự đoán dưới đây được ghi lại **trước khi tiến hành huấn luyện** bất kỳ mô hình nào, kèm theo lập luận cơ chế kỹ thuật và mức độ tin cậy.

---

## Bảng tổng hợp dự đoán

| STT | Câu hỏi khảo sát | Dự đoán (Prediction) | Lý do dựa trên cơ chế mô hình (Reason) | Mức độ tin cậy (Confidence) |
|:---:|---|---|---|:---:|
| **1** | Giữa **TF-IDF + Logistic Regression** và **RNN + Embedding**, mô hình nào có khả năng xử lý word order (thứ tự từ) trực tiếp hơn? | **RNN + Embedding** | - TF-IDF biểu diễn văn bản dưới dạng túi từ (Bag-of-Words), đếm tần suất độc lập và mất hoàn toàn vị trí xuất hiện của các từ.<br>- RNN cập nhật trạng thái ẩn \(h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)\) tuần tự theo từng timestep, nên thông tin token trước trực tiếp ảnh hưởng tới biểu diễn của token sau. | **Rất cao (High)** |
| **2** | Giữa **RNN** và **LSTM**, mô hình nào có khả năng xử lý phụ thuộc xa (long-range dependency) tốt hơn? | **LSTM** | - Simple RNN dùng hàm kích hoạt phi tuyến liên tiếp \(\tanh\) và nhân dồn ma trận \(W_{hh}\), gây ra hiện tượng vanishing gradient khi chuỗi dài.<br>- LSTM bổ sung ô nhớ riêng biệt (cell state \(c_t\)) với cơ chế cộng tuyến tính \(c_t = f_t c_{t-1} + i_t \tilde{c}_t\). Cổng quên (\(f_t \approx 1\)) cho phép đạo hàm lan truyền ngược thẳng tắp mà không bị suy giảm theo hàm mũ. | **Rất cao (High)** |
| **3** | Nếu sequence length tăng từ **50 lên 200**, quá trình huấn luyện của RNN sẽ thay đổi như thế nào? | **Thời gian huấn luyện tăng mạnh, độ dốc gradient dễ bất ổn định; hiệu năng có thể tăng nhưng đối mặt rủi ro vanishing gradient** | - Mỗi mẫu phải unroll qua 200 bước thay vì 50 bước, làm tăng số phép nhân ma trận BPTT gấp 4 lần.<br>- Giữ được nhiều context hơn của văn bản (tỷ lệ token được giữ tăng từ ~21% lên ~69%), nhưng chuỗi 200 bước làm đường truyền đạo hàm dài gấp 4 lần, khiến gradient ở các bước đầu dễ bị triệt tiêu hơn. | **Cao (High) về thời gian, Trung bình (Medium) về độ tăng F1** |
| **4** | Nếu cắt cụt (truncate) toàn bộ review về **30 tokens**, thông tin phân loại có bị ảnh hưởng nghiêm trọng không? | **Bị ảnh hưởng rất nặng nề, độ chính xác (accuracy/F1) có xu hướng sụt giảm rõ rệt** | - Trong review phim IMDb, độ dài trung bình là ~231 tokens. 30 tokens chỉ giữ lại phần mở đầu câu chào hoặc vài từ giới thiệu phim.<br>- Những từ nhận định cảm xúc then chốt, lời kết luận tổng kết, hoặc mệnh đề đảo ngược ý (*"however, ultimately it was a terrible movie"*) thường nằm ở giữa hoặc cuối bài review và sẽ bị vứt bỏ hoàn toàn. | **Rất cao (High)** |
| **5** | LSTM có **chắc chắn** cho accuracy cao hơn Simple RNN trên **mọi dataset** không? | **KHÔNG CHẮC CHẮN** | - Hiệu năng thực nghiệm phụ thuộc vào: đặc điểm tập dữ liệu (chuỗi ngắn hay dài), kích thước tập train (LSTM có số tham số gấp 4 lần RNN nên dễ bị overfit nếu dữ liệu ít), ngân sách huấn luyện (số epoch, learning rate) và độ phức tạp của bài toán.<br>- Với các bài toán câu ngắn hoặc từ khóa cảm xúc xuất hiện ngay cuối câu, RNN hoàn toàn có thể đạt kết quả tương đương hoặc tốt hơn nhờ ít tham số hơn. | **Rất cao (High)** |

---
*Các dự đoán trên được lưu giữ làm đối chứng so sánh với bảng kết quả thực nghiệm định lượng trong file `results.csv` và báo cáo `reflection.md`.*
