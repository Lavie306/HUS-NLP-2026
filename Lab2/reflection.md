# LAB 02 — BÁO CÁO TỔNG KẾT & SUY NGẪM (REFLECTION)

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Chủ đề:** N-gram Language Models, Smoothing and Perplexity  

---

### Câu 1: Nếu tăng $n$, mô hình nhận thêm thông tin gì?
* **Trả lời:** Mô hình nhận thêm **ngữ cảnh lịch sử liền kề (local context)** và **thứ tự từ (word order)**. Nhờ đó, mô hình nắm bắt được các kết hợp từ cố định (collocations) và cấu trúc ngữ pháp ngắn thay vì chỉ xem các từ xuất hiện độc lập như unigram.

---

### Câu 2: Tại sao tăng $n$ lại làm sparsity tăng?
* **Trả lời:** Không gian tổ hợp chuỗi $n$ từ bùng nổ theo cấp số mũ $|\mathcal{V}|^n$, trong khi tập dữ liệu huấn luyện luôn hữu hạn. Đa số các cụm từ trong thực tế sẽ không xuất hiện trong tập train ($C=0$), khiến ma trận thống kê bị thưa cực hạn.

---

### Câu 3: Tại sao smoothing cần thiết?
* **Trả lời:** Để tránh **Zero Probability**. Nếu không làm mịn, chỉ cần một $n$-gram lạ xuất hiện trong câu, xác suất MLE bằng 0 sẽ làm triệt tiêu xác suất cả câu về 0 ($P(S)=0$) và kéo Perplexity lên vô cùng ($+\infty$). Smoothing chiết khấu xác suất từ các từ phổ biến để bù cho các $n$-gram chưa từng thấy.

---

### Câu 4: Perplexity đo điều gì?
* **Trả lời:** Đo mức độ "bối rối" (uncertainty) của mô hình trước dữ liệu. Về trực giác, nó tương đương số lượng từ trung bình mà mô hình phải phân vân ở mỗi bước dự đoán. **Perplexity càng thấp, mô hình dự đoán càng chính xác.**

---

### Câu 5: Model có perplexity thấp hơn có luôn tạo ra văn bản tốt hơn đối với con người không?
* **Trả lời:** **Không nhất thiết.**
* **Lý do:** Mô hình có thể đạt PPL thấp nhờ đoán đúng các từ nối, mạo từ hay hư từ phổ biến, nhưng khi sinh văn bản dài vẫn có thể bị lặp từ, thiếu mạch lạc hoặc phi logic. Ngoài ra, PPL còn phụ thuộc mạnh vào cách xử lý từ vựng và OOV.

---

### Câu 6: N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?
* **Trả lời:** 
  1. **Không hiểu ngữ nghĩa:** Xem từ là ký hiệu rời rạc, không biết các từ đồng nghĩa (như *bác sĩ* và *y tá*) có thể chia sẻ ngữ cảnh.
  2. **Tầm nhìn quá ngắn:** Giả định Markov ép mô hình chỉ nhớ $n-1$ từ trước.
  3. **Không nắm bắt được cú pháp sâu và logic thế giới thực.**

---

### Câu 7: Nếu context dài 100 từ, trigram có sử dụng được thông tin của 97 từ đầu không?
* **Trả lời:** **Không hề.**
* Theo giả định Markov bậc 2: $P(w_{101} | w_1 \dots w_{100}) \approx P(w_{101} | w_{99}, w_{100})$. Mô hình bỏ qua hoàn toàn 97 từ trước đó, làm mất các quan hệ phụ thuộc xa (long-range dependencies). 
* **Cầu nối sang Neural LM:** Hạn chế này dẫn đến sự ra đời của **RNN/LSTM** (lưu trạng thái ẩn) và **Transformer** (cơ chế Self-Attention chú ý toàn bộ ngữ cảnh).
