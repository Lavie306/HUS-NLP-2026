# Predictions Before Experiment — LAB 10

Ghi nhận các dự đoán lý thuyết **trước khi chạy thực nghiệm** (Mục 9).

---

## Prediction 1 — Lexical Overlap & Distractors

- **Câu hỏi:** Nếu question và context chứa nhiều từ giống nhau nhưng chỉ một vị trí thực sự trả lời câu hỏi, model có dễ bị nhầm không?
- **Dự đoán:** Có, đặc biệt với các mô hình dựa trên khớp từ khóa (như Baseline) và ở mức độ nhẹ hơn với Transformer khi các đoạn gây nhiễu (distractors) có cấu trúc ngữ nghĩa gần giống câu hỏi.
- **Lý do:** Trùng lặp từ vựng (lexical overlap) là tín hiệu gợi ý mạnh nhưng không đủ để phân biệt vai trò ngữ pháp hay quan hệ giữa các thực thể. Nếu hai câu trong context cùng nhắc đến các từ khóa trong câu hỏi nhưng nói về hai sự kiện/thời điểm khác nhau, mô hình có thể chọn nhầm đoạn chứa nhiều từ khóa hơn thay vì đoạn trả lời đúng trọng tâm câu hỏi.
- **Cách kiểm chứng:** So sánh hiệu năng (EM/F1) giữa nhóm câu hỏi có lexical overlap cao và thấp (Experiment 4), đồng thời kiểm tra các ca lỗi dạng `distractor` trong Error Analysis.

---

## Prediction 2 — Answer Length & Boundary Prediction

- **Câu hỏi:** Answer span càng dài thì boundary prediction có khó hơn không?
- **Dự đoán:** Có. Answer span càng dài thì xác suất đoán đúng đồng thời cả hai biên (`start` và `end`) càng giảm, dẫn đến EM giảm rõ rệt; F1 cũng có xu hướng giảm nhưng ít nghiêm trọng hơn EM.
- **Lý do:** Để đạt Exact Match, mô hình phải xác định chính xác tuyệt đối cả vị trí bắt đầu lẫn vị trí kết thúc. Với một span dài (ví dụ cụm danh từ dài, mệnh đề giải thích $\ge 7$ từ), ranh giới kết thúc thường mơ hồ hơn (dễ thiếu hoặc thừa mạo từ, giới từ, cụm bổ nghĩa). Khi thiếu/thừa vài từ, EM lập tức bằng 0 dù phần lớn nội dung vẫn trùng (F1 vẫn có thể cao).
- **Cách kiểm chứng:** Chia tập test theo độ dài đáp án (`1`, `2–3`, `4–6`, `7+` tokens) trong Experiment 5 và đối chiếu EM/F1 giữa các nhóm.

---

## Prediction 3 — Irrelevant Information & Distractors in Context

- **Câu hỏi:** Nếu context chứa nhiều thông tin không liên quan và distractor, hiệu năng sẽ thay đổi như thế nào?
- **Dự đoán:** Hiệu năng (EM và F1) sẽ **giảm** khi thêm nhiều thông tin không liên quan hoặc distractor vào context.
- **Lý do:**
  1. Không gian tìm kiếm `(start, end)` mở rộng, làm tăng khả năng xuất hiện các span nhiễu có xác suất cạnh tranh.
  2. Nếu thông tin thừa đứng trước câu trả lời và đẩy tổng độ dài vượt quá `max_length`, context phải chia thành nhiều sliding windows (có window không chứa đáp án), làm tăng rủi ro chọn nhầm window.
  3. Các câu distractor có từ vựng tương đồng với câu hỏi dễ đánh lừa cơ chế attention nếu câu hỏi không đủ chặt chẽ.
- **Cách kiểm chứng:** Chèn đoạn văn bản không liên quan vào trước và sau context gốc trên tập test (giữ nguyên câu hỏi và đáp án, cập nhật lại offset) và so sánh EM/F1 với bản gốc.

---

## Prediction 4 — Sentence Order Perturbation

- **Câu hỏi:** Nếu chỉ thay đổi thứ tự một số câu trong context nhưng giữ nguyên answer, model có bị ảnh hưởng không?
- **Dự đoán:** Có thể bị ảnh hưởng nhẹ đến vừa; một số prediction sẽ thay đổi (cả tích cực lẫn tiêu cực) chứ không bất biến hoàn toàn.
- **Lý do:** Transformer sử dụng Positional Embeddings và cơ chế Self-Attention trên toàn chuỗi. Việc đảo thứ tự câu làm thay đổi khoảng cách vị trí tương đối giữa câu hỏi và đoạn chứa đáp án, đồng thời phá vỡ mạch tham chiếu đại từ/ngữ cảnh giữa các câu lân cận. Ngoài ra, nếu câu trả lời bị đẩy ra gần ranh giới window hoặc sang window khác, kết quả dự đoán cũng có thể thay đổi.
- **Cách kiểm chứng:** Hoán vị thứ tự các câu trong context (cập nhật lại `answer_start` tương ứng với vị trí mới của đáp án để đảm bảo chấm đúng) và đo lại EM/F1 cũng như tỷ lệ câu trả lời bị thay đổi.
