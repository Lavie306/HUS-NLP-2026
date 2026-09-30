# LAB 02 — BÀI TẬP TÍNH TOÁN BẰNG TAY (CALCULATIONS)

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Chủ đề:** N-gram Language Models, Smoothing and Perplexity  
**Quy tắc:** Thực hiện tính toán thủ công trước khi kiểm chứng bằng code.

---

## 1. Bài 1 — Unigram (Mục 7)
### Dữ liệu ngữ liệu (Corpus):
* $D_1$: `the cat eats fish`
* $D_2$: `the cat likes fish`
* $D_3$: `the dog eats meat`

### 1.1 Xác định Vocabulary ($V$)
Tập hợp các từ phân biệt (unique tokens):
$$\mathcal{V} = \{\text{the}, \text{cat}, \text{eats}, \text{fish}, \text{likes}, \text{dog}, \text{meat}\}$$
* Kích thước từ vựng: $|\mathcal{V}| = V = 7$.

### 1.2 Tổng số token ($N$)
Đếm tổng số lần xuất hiện của tất cả các từ trong cả 3 câu:
* $D_1$: 4 tokens
* $D_2$: 4 tokens
* $D_3$: 4 tokens
$$\Rightarrow N = 4 + 4 + 4 = 12\text{ tokens}$$

### 1.3 Tính $P(w) = \frac{C(w)}{N}$
Tần số xuất hiện (Count):
* $C(\text{the}) = 3 \implies P(\text{the}) = \frac{3}{12} = 0.25$
* $C(\text{cat}) = 2 \implies P(\text{cat}) = \frac{2}{12} = \frac{1}{6} \approx 0.1667$
* $C(\text{fish}) = 2 \implies P(\text{fish}) = \frac{2}{12} = \frac{1}{6} \approx 0.1667$
* $C(\text{dog}) = 1 \implies P(\text{dog}) = \frac{1}{12} \approx 0.0833$
* $C(\text{eats}) = 2 \implies P(\text{eats}) = \frac{2}{12} = \frac{1}{6} \approx 0.1667$
* $C(\text{likes}) = 1 \implies P(\text{likes}) = \frac{1}{12} \approx 0.0833$
* $C(\text{meat}) = 1 \implies P(\text{meat}) = \frac{1}{12} \approx 0.0833$

### 1.4 Kiểm tra phân phối hợp lệ: $\sum_{w \in \mathcal{V}} P(w) = 1$
$$\sum_{w \in \mathcal{V}} P(w) = \frac{3 + 2 + 2 + 1 + 2 + 1 + 1}{12} = \frac{12}{12} = 1.0 \quad \text{(Chính xác)}$$

---

## 2. Bài 2 — Bigram (Mục 7)
Công thức Maximum Likelihood Estimation (MLE):
$$P(w_t | w_{t-1}) = \frac{C(w_{t-1}, w_t)}{C(w_{t-1})}$$

Đếm số lần xuất hiện các context và bigram trong corpus:
* Context $w_{t-1} = \text{the}$: $C(\text{the}) = 3$.
  - $C(\text{the}, \text{cat}) = 2$
  - $C(\text{the}, \text{dog}) = 1$
  $$\implies P(\text{cat} | \text{the}) = \frac{C(\text{the}, \text{cat})}{C(\text{the})} = \frac{2}{3} \approx 0.6667$$
  $$\implies P(\text{dog} | \text{the}) = \frac{C(\text{the}, \text{dog})}{C(\text{the})} = \frac{1}{3} \approx 0.3333$$

* Context $w_{t-1} = \text{cat}$: $C(\text{cat}) = 2$.
  - $C(\text{cat}, \text{eats}) = 1$
  - $C(\text{cat}, \text{likes}) = 1$
  $$\implies P(\text{eats} | \text{cat}) = \frac{C(\text{cat}, \text{eats})}{C(\text{cat})} = \frac{1}{2} = 0.5$$
  $$\implies P(\text{likes} | \text{cat}) = \frac{C(\text{cat}, \text{likes})}{C(\text{cat})} = \frac{1}{2} = 0.5$$

### Trả lời câu hỏi:
**Tại sao tổng xác suất của các từ đứng sau "the" phải bằng 1 nếu vocabulary và context được xử lý đầy đủ?**
* **Trả lời:** Theo tiên đề xác suất Kolmogorov, với một biến cố điều kiện $A = \{\text{từ trước là 'the'}\}$, không gian mẫu của từ tiếp theo là $\Omega = \mathcal{V}$. Vì mỗi lần từ "the" xuất hiện trong văn bản, nó bắt buộc phải được theo sau bởi chính xác một từ duy nhất thuộc từ vựng $\mathcal{V}$, nên:
  $$\sum_{w \in \mathcal{V}} P(w | \text{the}) = \sum_{w \in \mathcal{V}} \frac{C(\text{the}, w)}{C(\text{the})} = \frac{\sum_{w \in \mathcal{V}} C(\text{the}, w)}{C(\text{the})} = \frac{C(\text{the})}{C(\text{the})} = 1$$

---

## 3. Bài 3 — Xác suất câu & Độ dài (Mục 7)
Xét câu: $S = \text{the cat eats fish}$

Theo mô hình Bigram:
$$P(S) = P(\text{the}) \cdot P(\text{cat} | \text{the}) \cdot P(\text{eats} | \text{cat}) \cdot P(\text{fish} | \text{eats})$$

Tra cứu các xác suất từ corpus:
* $P(\text{the}) = \frac{3}{12} = 0.25$
* $P(\text{cat} | \text{the}) = \frac{2}{3} \approx 0.6667$
* $P(\text{eats} | \text{cat}) = \frac{1}{2} = 0.5$
* Context $\text{eats}$: $C(\text{eats}) = 2$, trong đó có $C(\text{eats}, \text{fish}) = 1$, $C(\text{eats}, \text{meat}) = 1$.
  $$P(\text{fish} | \text{eats}) = \frac{1}{2} = 0.5$$

Tính xác suất toàn bộ câu:
$$P(S) = 0.25 \times \frac{2}{3} \times 0.5 \times 0.5 = \frac{1}{4} \times \frac{2}{3} \times \frac{1}{4} = \frac{2}{48} = \frac{1}{24} \approx 0.04167$$

### Trả lời câu hỏi:
**Nếu thêm một từ vào câu, xác suất của cả câu có thể tăng không?**
* **Trả lời:** **Không thể tăng.**
* **Giải thích:** Giả sử ta thêm từ $w_{T+1}$ vào cuối câu, xác suất mới sẽ là:
  $$P(w_1, \dots, w_T, w_{T+1}) = P(w_1, \dots, w_T) \times P(w_{T+1} | w_T)$$
  Vì $P(w_{T+1} | w_T) \in [0, 1]$, nhân thêm một thừa số $\le 1$ chỉ có thể làm xác suất giữ nguyên (nếu xác suất bằng 1) hoặc giảm đi. 
* **Hệ quả quan trọng:** Xác suất chuỗi $P(S)$ luôn bị phạt theo độ dài câu (câu càng dài thì xác suất càng tiến gần về 0). Do đó, $P(S)$ **không thể dùng trực tiếp để so sánh chất lượng giữa các câu có độ dài khác nhau**. Đây chính là lý do ra đời của thước đo chuẩn hóa theo độ dài: **Perplexity**.

---

## 4. Bài 4 — Sentence Ranking (Mục 7)
So sánh hai câu:
* $S_1 = \text{the cat eats fish}$
* $S_2 = \text{the dog eats fish}$

### Phân rã xác suất:
* $P(S_1) = P(\text{the}) \cdot P(\text{cat} | \text{the}) \cdot P(\text{eats} | \text{cat}) \cdot P(\text{fish} | \text{eats})$
* $P(S_2) = P(\text{the}) \cdot P(\text{dog} | \text{the}) \cdot P(\text{eats} | \text{dog}) \cdot P(\text{fish} | \text{eats})$

### Quan sát trên dữ liệu:
* Trong corpus, sau `dog` xuất hiện từ gì?
  - Dòng 3: `the dog eats meat` $\implies C(\text{dog}) = 1, C(\text{dog}, \text{eats}) = 1 \implies P(\text{eats} | \text{dog}) = 1.0$.
* So sánh các thành phần:
  - $P(\text{the}) = 0.25$ (giống nhau)
  - $P(\text{fish} | \text{eats}) = 0.5$ (giống nhau)
  - Với $S_1$: $P(\text{cat} | \text{the}) \cdot P(\text{eats} | \text{cat}) = \frac{2}{3} \times \frac{1}{2} = \frac{1}{3} \approx 0.3333$
  - Với $S_2$: $P(\text{dog} | \text{the}) \cdot P(\text{eats} | \text{dog}) = \frac{1}{3} \times 1.0 = \frac{1}{3} \approx 0.3333$
* **Kết luận:**
  $$P(S_1) = 0.25 \times \frac{1}{3} \times 0.5 = \frac{1}{24} \approx 0.04167$$
  $$P(S_2) = 0.25 \times \frac{1}{3} \times 0.5 = \frac{1}{24} \approx 0.04167$$
  $\implies P(S_1) = P(S_2)$. Dưới mô hình bigram không có smoothing trên tập dữ liệu nhỏ này, cả 2 câu có xác suất bằng nhau.

---

## 5. Bài 9 — Suy luận Zero Probability trước khi Smoothing (Mục 9)
### Dữ liệu ngữ liệu:
* `I like NLP`
* `I like AI`
* `I study NLP`

Cần tính: $P(\text{AI} | \text{study})$
1. **Count của bigram `study AI`:** $C(\text{study}, \text{AI}) = 0$.
2. **Xác suất MLE:**
   $$P_{\text{MLE}}(\text{AI} | \text{study}) = \frac{C(\text{study}, \text{AI})}{C(\text{study})} = \frac{0}{1} = 0.0$$
3. **Điều gì xảy ra khi tính xác suất câu chứa bigram này?**
   - Khi tính xác suất của bất kỳ câu nào có chứa cụm `study AI` (ví dụ câu `"I study AI"`), do tích chuỗi chứa một thừa số bằng 0 nên $P(S) = \prod P = 0.0$.
4. **Điều này có nghĩa mô hình "biết" rằng câu đó chắc chắn không thể xảy ra trong thực tế không?**
   - **Không.** Hiện tượng xác suất bằng 0 này đơn thuần là do dữ liệu huấn luyện có kích thước hữu hạn (sampling bias / sparsity), không thể bao quát toàn bộ thế giới thực.
   - **Bản chất:** *Không quan sát thấy trong tập huấn luyện $\ne$ Không thể xảy ra trong thực tế.*

---

## 6. Bài 11 — Laplace Smoothing (Mục 11)
Công thức Laplace (Add-1) Smoothing:
$$P_{\text{Laplace}}(w | h) = \frac{C(h, w) + 1}{C(h) + V}$$

Cho dữ liệu: $C(\text{cat}) = 10$, $C(\text{cat}, \text{eats}) = 0$, $V = 5$.

### Trường hợp 1: $C(\text{cat}, \text{eats}) = 0$
$$P_{\text{Laplace}}(\text{eats} | \text{cat}) = \frac{0 + 1}{10 + 5} = \frac{1}{15} \approx 0.0667$$

### Trường hợp 2: $C(\text{cat}, \text{eats}) = 3$
$$P_{\text{Laplace}}(\text{eats} | \text{cat}) = \frac{3 + 1}{10 + 5} = \frac{4}{15} \approx 0.2667$$
*(So với MLE thuần: $P_{\text{MLE}} = \frac{3}{10} = 0.30$)*

### Trả lời câu hỏi:
**Smoothing đã thay đổi xác suất của những bigram khác như thế nào?**
* **Trả lời:** Smoothing thực hiện cơ chế **"chiết khấu xác suất" (discounting / probability mass redistribution)**.
* Nó lấy đi một phần xác suất tích lũy từ các từ/bigram đã xuất hiện thường xuyên (như từ có count = 3 bị giảm từ $0.30$ xuống $0.2667$) để phân phối đồng đều lại cho các bigram chưa từng xuất hiện (nâng từ $0$ lên $0.0667$). Tổng xác suất của toàn bộ phân phối điều kiện vẫn được bảo toàn đúng bằng 1.

---

## 7. Bài 18 — Tính toán Perplexity (Mục 18)
Cho chuỗi $W = (w_1, w_2, w_3)$ có độ dài $N = 3$:
* $P(w_1) = 0.5$
* $P(w_2 | w_1) = 0.25$
* $P(w_3 | w_2) = 0.5$

### 7.1 Tính xác suất chuỗi $P(W)$
$$P(W) = 0.5 \times 0.25 \times 0.5 = 0.0625 = \frac{1}{16}$$

### 7.2 Tính Perplexity $PP(W)$
Công thức:
$$PP(W) = P(W)^{-\frac{1}{N}} = \left(\frac{1}{16}\right)^{-\frac{1}{3}} = (16)^{\frac{1}{3}} = \sqrt[3]{16} \approx 2.5198$$

### 7.3 Tính lại khi $P(w_2 | w_1)$ giảm xuống $0.1$
* Xác suất chuỗi mới:
  $$P'(W) = 0.5 \times 0.1 \times 0.5 = 0.025 = \frac{1}{40}$$
* Perplexity mới:
  $$PP'(W) = \left(\frac{1}{40}\right)^{-\frac{1}{3}} = (40)^{\frac{1}{3}} = \sqrt[3]{40} \approx 3.4199$$

### Trả lời câu hỏi:
**Vì sao chỉ một xác suất nhỏ cũng có thể làm perplexity thay đổi đáng kể?**
* **Trả lời:** Vì Perplexity tỷ lệ nghịch lũy thừa với xác suất của từng bước:
  $$PP(W) = \exp\left( - \frac{1}{N} \sum_{i=1}^N \ln P(w_i | \text{context}_i) \right)$$
* Hàm số $-\ln(p)$ tăng vọt rất nhanh về vô cùng khi xác suất $p \to 0$. Khi một xác suất đơn lẻ giảm mạnh, log-loss tăng đột biến, kéo theo trung bình hình học bị kéo xuống và làm Perplexity tăng vọt. Đặc biệt, nếu có bất kỳ $p = 0$, $PP(W) \to +\infty$.
