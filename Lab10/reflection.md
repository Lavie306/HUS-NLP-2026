# Reflection — LAB 10: Từ Sequence/Token Classification đến Span Prediction

Chuỗi bài thực hành từ **LAB 08**, **LAB 09** đến **LAB 10** cho thấy một nguyên lý cốt lõi của NLP hiện đại: cùng một **pretrained Transformer backbone** có thể giải quyết những bài toán hoàn toàn khác nhau chỉ bằng cách thay đổi **biểu diễn đầu ra được trích xuất**, **prediction head** và **hàm mục tiêu (target/loss)**.

```text
                    Pretrained Transformer
                      /        |        \
                 [CLS]     Token reps   Token reps
                   |           |            |
            Classification    NER       Start / End
                   |           |            |
                LAB 08      LAB 09       LAB 10
```

Ở **LAB 08 (Sequence Classification)**, bài toán yêu cầu gán một nhãn duy nhất cho toàn bộ văn bản (ví dụ: phân loại cảm xúc tích cực/tiêu cực). Mô hình chỉ cần lấy vector biểu diễn tổng hợp tại token `[CLS]` (hoặc pooling toàn chuỗi) và đưa qua một lớp Linear để dự đoán phân phối xác suất trên tập nhãn cố định. Dù chỉ sử dụng một vector đầu ra, cơ chế Self-Attention vẫn cho phép `[CLS]` tổng hợp ngữ cảnh từ mọi token trong câu.

Sang **LAB 09 (Token Classification / NER)**, một nhãn cho cả câu không còn đủ; ta cần phân loại tại **từng vị trí token** (ví dụ nhãn `B-PER`, `I-PER`, `O`). Do đó, prediction head là một lớp Linear áp dụng độc lập lên hidden state của mỗi token $h_t$. Thách thức kỹ thuật nảy sinh là sự lệch pha giữa ranh giới từ (word) và ranh giới subword do tokenizer tạo ra, đòi hỏi căn chỉnh nhãn (label alignment) và bỏ qua các subword phụ (`-100`) khi tính loss.

Đến **LAB 10 (Extractive Question Answering)**, đầu vào không còn là một văn bản đơn lẻ mà là cặp **Question + Context**. Hidden states của Transformer lúc này mã hóa sự tương tác chéo (cross-attention bên trong self-attention) giữa câu hỏi và ngữ cảnh. Thay vì phân loại từng token vào tập nhãn cố định, QA head sử dụng hai bộ phân loại tuyến tính song song trên các token của context để dự đoán xác suất **vị trí bắt đầu ($P_{\text{start}}$)** và **vị trí kết thúc ($P_{\text{end}}$)** của đoạn trả lời. Cách thiết kế này cho phép trích xuất một span có độ dài tùy ý trực tiếp từ văn bản gốc mà không cần sinh từ mới.

Thách thức kỹ thuật lớn nhất ở LAB 10 là **offset mapping** và **sliding window**: dữ liệu SQuAD cung cấp vị trí đáp án theo chỉ số ký tự (`character offset`), trong khi Transformer làm việc theo chỉ số token (`token index`). Việc ánh xạ hai chiều, xử lý ranh giới subword, và chia context dài thành các cửa sổ trượt chồng lấp (`stride`) là bắt buộc để không làm mất câu trả lời. Đồng thời, thước đo đánh giá cũng chuyển từ Accuracy/Entity-F1 sang **Exact Match (EM)** và **Token-level F1** nhằm phản ánh cả độ chính xác tuyệt đối lẫn mức độ trùng lặp từng phần của span.

Cuối cùng, LAB 10 đặt nền móng trực tiếp cho **LAB 11 (Semantic Retrieval)** và **LAB 12 (RAG)**: Extractive QA giả định context đã được cung cấp sẵn, nhưng trong thực tế với hàng nghìn tài liệu, ta cần bước truy xuất (Retriever) để tìm ra Top-K tài liệu liên quan trước khi đưa vào mô hình đọc hiểu hoặc sinh câu trả lời.
