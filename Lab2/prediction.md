# Prediction Before Experiment — Lab 02

---

### 1. Prediction 1: Kích thước từ vựng ($|\mathcal{V}|$)
* **Dự đoán:** Không đổi khi tăng từ Unigram $\to$ Bigram $\to$ Trigram.
* **Lý do:** Từ vựng chỉ tính số lượng từ đơn phân biệt trong ngữ liệu; việc ghép cặp hay bộ ba từ không làm thay đổi tập từ vựng nền tảng này.
* **Độ tự tin:** 95%.
* **Kiểm chứng:** **ĐÚNG**. Cả 3 mô hình đều dùng chung tập từ vựng cố định $|\mathcal{V}| = 186.618$ từ.

---

### 2. Prediction 2: Số lượng $n$-gram phân biệt
* **Dự đoán:** Tăng vọt theo bậc $n$ ($N_{\text{unigram}} \ll N_{\text{bigram}} \ll N_{\text{trigram}}$).
* **Lý do:** Không gian tổ hợp mở rộng theo $|\mathcal{V}|^n$, cách ghép từ trong câu rất đa dạng nên số cụm từ mới tăng cực nhanh theo độ dài văn bản.
* **Độ tự tin:** 99%.
* **Kiểm chứng:** **ĐÚNG**.
  - Unigram: **186.618**
  - Bigram: **2.739.018** (gấp ~14.7 lần)
  - Trigram: **6.404.589** (gấp ~34.3 lần)

---

### 3. Prediction 3: Nguy cơ gặp Zero Probability
* **Dự đoán:** Trigram dễ gặp nhất, kế đến là Bigram, thấp nhất là Unigram.
* **Lý do:** Không gian trigram quá rộng ($|\mathcal{V}|^3 \approx 10^{15}$) nên dữ liệu huấn luyện chỉ bao phủ một phần rất nhỏ; câu mới rất dễ chứa bộ 3 từ chưa từng thấy trong tập train.
* **Độ tự tin:** 95%.
* **Kiểm chứng:** **ĐÚNG**.
  - Tỷ lệ n-gram chỉ xuất hiện 1 lần tăng mạnh: Unigram (47.41%) $\to$ Bigram (71.30%) $\to$ Trigram (86.21%).
  - Trên tập Test, cả Bigram MLE và Trigram MLE đều bị $\text{PPL} = \infty$.

---

### 4. Prediction 4: Perplexity trên Training Set
* **Dự đoán:** Trigram MLE đạt Perplexity thấp nhất trên tập huấn luyện.
* **Lý do:** Có ngữ cảnh dài hơn (2 từ trước) và nhiều tham số hơn nên mô hình dễ dàng "học vẹt" chính xác dữ liệu đã thấy.
* **Độ tự tin:** 90%.
* **Kiểm chứng:** **ĐÚNG**.
  - Training PPL của Trigram MLE chỉ là **19.71** (so với Bigram MLE: 209.81 và Unigram: 1915.23).
  - Tuy nhiên, khi dùng Laplace smoothing với $|\mathcal{V}|$ quá lớn, Trigram Laplace bị phạt nặng lên **27.462,79**.

---

### 5. Prediction 5: Trigram vs Bigram trên Corpus nhỏ
* **Dự đoán:** Trigram không chắc tốt hơn Bigram; Bigram/Unigram thậm chí có thể vượt trội khi dữ liệu ít.
* **Lý do:** Dữ liệu nhỏ khiến hầu hết trigram bị thiếu (sparsity), mô hình phụ thuộc hoàn toàn vào smoothing nhân tạo làm loãng phân phối xác suất.
* **Độ tự tin:** 85%.
* **Kiểm chứng:** **ĐÚNG**.
  - Trên tập Test, Unigram đạt PPL tốt nhất (**2050.78**), kế đến là Bigram (**6703.09**), còn Trigram tệ nhất (**43047.01**) do mẫu số làm mịn cộng thêm $|\mathcal{V}| = 186.618$ quá lớn.
