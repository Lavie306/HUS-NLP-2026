# LAB 07 — Transformer Architecture: Báo cáo Lý thuyết & Bài tập Tính toán

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — Năm học 2026  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Chủ đề:** From Attention to Transformer  

---

## 1. Bối cảnh Kiến trúc: Từ Self-Attention đến Transformer Block

### 1.1. Hành trình phát triển qua các Lab
Trong tiến trình tiếp cận các phương pháp biểu diễn ngôn ngữ tự nhiên:
- **LAB 04 (Túi từ & Phân loại văn bản):** Dùng biểu diễn thống kê thưa (TF-IDF) hoặc nhúng tĩnh trung bình (Avg Word2Vec). Hạn chế lớn là triệt tiêu thứ tự từ và không phụ thuộc ngữ cảnh động.
- **LAB 05 (Mô hình chuỗi hồi quy - RNN/LSTM/GRU):** Xử lý văn bản tuần tự theo trục thời gian $t = 1, \dots, T$. Mặc dù mô hình hóa được thứ tự từ, RNN vướng phải **điểm nghẽn thông tin** (information bottleneck khi nén toàn bộ văn bản vào một vector ẩn duy nhất $h_T$) và **điểm nghẽn tính toán tuần tự** (sequential bottleneck, không thể tính toán song song trên GPU).
- **LAB 06 (Cơ chế Chú ý - Attention & Self-Attention):** Cho phép các từ tương tác trực tiếp với nhau thông qua cơ chế so khớp Query-Key và tổng trọng số Value:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
  Self-Attention phá vỡ giới hạn cự ly xa, rút ngắn đường truyền thông tin giữa hai vị trí bất kỳ xuống $O(1)$.

### 1.2. Đặt vấn đề của LAB 07
Nếu Self-Attention có thể cho phép mọi vị trí trong câu tương tác trực tiếp và tính toán song song, **làm thế nào để ghép Self-Attention thành một kiến trúc mạng nơ-ron sâu hoàn chỉnh có thể huấn luyện ổn định và hiệu quả?**

Kiến trúc **Transformer** (Vaswani et al., 2017) giải quyết câu hỏi này bằng cách kết hợp một hệ thống gồm 4 trụ cột chặt chẽ:
1. **Positional Encoding (PE):** Bổ sung tọa độ vị trí vào token embeddings để bù đắp đặc tính bất biến hoán vị (permutation invariance) của Self-Attention.
2. **Multi-Head Self-Attention (MHA):** Tách không gian biểu diễn thành nhiều không gian con (subspaces) độc lập, cho phép mô hình đồng thời chú ý tới nhiều khía cạnh cú pháp và ngữ nghĩa khác nhau.
3. **Position-wise Feed-Forward Network (FFN):** Mạng nơ-ron truyền thẳng biến đổi phi tuyến cục bộ tại từng token, gia tăng sức chứa dung lượng thông tin.
4. **Residual Connections & Layer Normalization (Add & Norm):** Đường truyền tắt bảo toàn thông tin gốc và chuẩn hóa phương sai đặc trưng, tạo ra "cao tốc gradient" giúp huấn luyện ổn định các khối xếp chồng sâu.

---

## 2. Sơ đồ Luồng dữ liệu (Dataflow) Khối Transformer Encoder

```
            Đầu vào chuỗi token x_1, x_2, ..., x_T
                             │
                             ▼
                  Embedding + Positional Encoding
                             │
            ┌────────────────┴────────────────┐
            │                                 │ (Residual Highway 1)
            ▼                                 │
     Multi-Head Attention                     │
            │                                 │
            ▼                                 │
            └──────────────► (+) ◄────────────┘
                             │
                             ▼
                         LayerNorm
                             │
            ┌────────────────┴────────────────┐
            │                                 │ (Residual Highway 2)
            ▼                                 │
     Feed-Forward Network                     │
            │                                 │
            ▼                                 │
            └──────────────► (+) ◄────────────┘
                             │
                             ▼
                         LayerNorm
                             │
                             ▼
             Output biểu diễn ngữ cảnh (Contextual Rep)
```

---

## 3. Bài tập Tính toán 1: Tính toán chi tiết một Attention Head (Mục 6)

### 3.1. Đề bài
Cho ma trận đầu vào biểu diễn chuỗi gồm 3 tokens, mỗi token có số chiều $d_{model} = 4$:
$$X \in \mathbb{R}^{3 \times 4}$$

Giả sử một attention head có số chiều chiếu là $d_k = 2$.  
Các ma trận trọng số chiếu tương ứng:
$$W^Q \in \mathbb{R}^{4 \times 2}, \quad W^K \in \mathbb{R}^{4 \times 2}, \quad W^V \in \mathbb{R}^{4 \times 2}$$

### 3.2. Thiết lập giá trị số học cụ thể
Để kiểm chứng chính xác từng phép tính, ta xét các ma trận số học cụ thể:
$$X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 1 \\ 1 & 1 & 0 & 2 \end{bmatrix}$$

$$W^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W^K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad W^V = \begin{bmatrix} 1 & 1 \\ 0 & 2 \\ 2 & 0 \\ 1 & 1 \end{bmatrix}$$

### 3.3. Các bước tính toán từng bước

#### Bước 1: Chiếu tuyến tính sinh Query ($Q$), Key ($K$), Value ($V$)
- **Tính $Q = X W^Q \in \mathbb{R}^{3 \times 2}$:**
  $$Q = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 1 \\ 1 & 1 & 0 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1\cdot 1 + 1\cdot 1 & 0\cdot 1 + 0\cdot 1 \\ 0\cdot 1 + 0\cdot 1 & 2\cdot 1 + 1\cdot 1 \\ 1\cdot 1 + 0\cdot 1 & 1\cdot 1 + 2\cdot 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \\ 1 & 3 \end{bmatrix}$$

- **Tính $K = X W^K \in \mathbb{R}^{3 \times 2}$:**
  $$K = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 1 \\ 1 & 1 & 0 & 2 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0\cdot 1 + 0\cdot 1 & 1\cdot 1 + 1\cdot 1 \\ 2\cdot 1 + 1\cdot 1 & 0\cdot 1 + 0\cdot 1 \\ 1\cdot 1 + 2\cdot 1 & 1\cdot 1 + 0\cdot 1 \end{bmatrix} = \begin{bmatrix} 0 & 2 \\ 3 & 0 \\ 3 & 1 \end{bmatrix}$$

- **Tính $V = X W^V \in \mathbb{R}^{3 \times 2}$:**
  $$V = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 1 \\ 1 & 1 & 0 & 2 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 2 \\ 2 & 0 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 1\cdot 1 + 1\cdot 2 & 1\cdot 1 + 0\cdot 2 \\ 0\cdot 1 + 1\cdot 1 & 2\cdot 2 + 1\cdot 1 \\ 1\cdot 1 + 2\cdot 1 & 1\cdot 1 + 1\cdot 2 + 2\cdot 1 \end{bmatrix} = \begin{bmatrix} 3 & 1 \\ 1 & 5 \\ 3 & 5 \end{bmatrix}$$

---

#### Bước 2: Tính tích vô hướng và hệ số co dãn $\sqrt{d_k}$
Chuyển vị ma trận Key:
$$K^T = \begin{bmatrix} 0 & 3 & 3 \\ 2 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

Tích vô hướng $Q K^T \in \mathbb{R}^{3 \times 3}$:
$$Q K^T = \begin{bmatrix} 2 & 0 \\ 0 & 3 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 0 & 3 & 3 \\ 2 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 6 & 6 \\ 6 & 0 & 3 \\ 6 & 3 & 6 \end{bmatrix}$$

Với $d_k = 2$, hệ số co dãn là $\sqrt{d_k} = \sqrt{2} \approx 1.4142$.  
Ma trận điểm số tương tác $S = \frac{Q K^T}{\sqrt{2}}$:
$$S = \begin{bmatrix} \frac{0}{\sqrt{2}} & \frac{6}{\sqrt{2}} & \frac{6}{\sqrt{2}} \\ \frac{6}{\sqrt{2}} & \frac{0}{\sqrt{2}} & \frac{3}{\sqrt{2}} \\ \frac{6}{\sqrt{2}} & \frac{3}{\sqrt{2}} & \frac{6}{\sqrt{2}} \end{bmatrix} \approx \begin{bmatrix} 0.0000 & 4.2426 & 4.2426 \\ 4.2426 & 0.0000 & 2.1213 \\ 4.2426 & 2.1213 & 4.2426 \end{bmatrix}$$

---

#### Bước 3: Chuẩn hóa phân phối Attention qua Softmax
Áp dụng Softmax trên từng hàng của ma trận điểm $S$:
$$A_{i, j} = \frac{\exp(S_{i, j})}{\sum_{k=1}^3 \exp(S_{i, k})}$$

- **Hàng 1:** $S_1 = [0.0000, 4.2426, 4.2426]$
  - $\exp(S_1) = [\exp(0), \exp(4.2426), \exp(4.2426)] \approx [1.0, 69.589, 69.589]$
  - Tổng hàng $= 1.0 + 69.589 + 69.589 = 140.178$
  - Trọng số $A_1 = [0.0071, 0.4964, 0.4964]$

- **Hàng 2:** $S_2 = [4.2426, 0.0000, 2.1213]$
  - $\exp(S_2) \approx [69.589, 1.0, 8.342]$
  - Tổng hàng $= 69.589 + 1.0 + 8.342 = 78.931$
  - Trọng số $A_2 = [0.8816, 0.0127, 0.1057]$

- **Hàng 3:** $S_3 = [4.2426, 2.1213, 4.2426]$
  - $\exp(S_3) \approx [69.589, 8.342, 69.589]$
  - Tổng hàng $= 69.589 + 8.342 + 69.589 = 147.520$
  - Trọng số $A_3 = [0.4717, 0.0565, 0.4717]$

Ma trận trọng số chú ý $A \in \mathbb{R}^{3 \times 3}$:
$$A \approx \begin{bmatrix} 0.0071 & 0.4964 & 0.4964 \\ 0.8816 & 0.0127 & 0.1057 \\ 0.4717 & 0.0565 & 0.4717 \end{bmatrix}$$

---

#### Bước 4: Tính biểu diễn ngữ cảnh đầu ra $Z = A V$
$$Z = A V = \begin{bmatrix} 0.0071 & 0.4964 & 0.4964 \\ 0.8816 & 0.0127 & 0.1057 \\ 0.4717 & 0.0565 & 0.4717 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 5 \\ 3 & 5 \end{bmatrix}$$

- **Hàng 1 của $Z$:**
  $$Z_{1, 1} = 0.0071(3) + 0.4964(1) + 0.4964(3) \approx 0.0213 + 0.4964 + 1.4892 = 2.0069 \approx 2.0071$$
  $$Z_{1, 2} = 0.0071(1) + 0.4964(5) + 0.4964(5) \approx 0.0071 + 2.4820 + 2.4820 = 4.9711 \approx 4.9715$$

- **Hàng 2 của $Z$:**
  $$Z_{2, 1} = 0.8816(3) + 0.0127(1) + 0.1057(3) \approx 2.6448 + 0.0127 + 0.3171 = 2.9746 \approx 2.9747$$
  $$Z_{2, 2} = 0.8816(1) + 0.0127(5) + 0.1057(5) \approx 0.8816 + 0.0635 + 0.5285 = 1.4736 \approx 1.4734$$

- **Hàng 3 của $Z$:**
  $$Z_{3, 1} = 0.4717(3) + 0.0565(1) + 0.4717(3) \approx 1.4151 + 0.0565 + 1.4151 = 2.8867 \approx 2.8869$$
  $$Z_{3, 2} = 0.4717(1) + 0.0565(5) + 0.4717(5) \approx 0.4717 + 0.2825 + 2.3585 = 3.1127 \approx 3.1131$$

Kết quả ma trận biểu diễn ngữ cảnh đầu ra của head:
$$Z \approx \begin{bmatrix} 2.0071 & 4.9715 \\ 2.9747 & 1.4734 \\ 2.8869 & 3.1131 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

### 3.4. Nhận xét cốt lõi (Core Insight)
Phép toán trên chứng minh một cách tường minh rằng: **Một attention head đơn lẻ hoàn toàn tương đồng về mặt toán học với cơ chế Scaled Dot-Product Attention đã học ở LAB 06, với điểm khác biệt duy nhất là các vector đầu vào $Q, K, V$ được tạo ra thông qua các phép chiếu ma trận $W^Q, W^K, W^V$.** Nhờ có các ma trận tham số có thể học được này, mô hình có thể linh hoạt dịch chuyển và trích xuất thông tin đặc thù thay vì bị bó hẹp trong không gian biểu diễn cố định ban đầu.

---

## 4. Bài tập Tính toán 2: Cơ chế Multi-Head Attention (Mục 7)

### 4.1. Đề bài & Thiết lập
Giả sử có $h = 2$ attention heads song song, mỗi head sinh ra vector ngữ cảnh có số chiều $d_k = 2$:
$$head_1 \in \mathbb{R}^{T \times 2}, \quad head_2 \in \mathbb{R}^{T \times 2}$$

Khi ghép nối các head dọc theo trục số chiều đặc trưng (feature dimension):
$$\text{Concat}(head_1, head_2) \in \mathbb{R}^{T \times (2 + 2)} = \mathbb{R}^{T \times 4}$$

Sau đó, một ma trận chiếu tuyến tính $W^O \in \mathbb{R}^{4 \times d_{model}}$ (với $d_{model} = 4$) được áp dụng:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, head_2) W^O \in \mathbb{R}^{T \times 4}$$

### 4.2. Trả lời câu hỏi: Vì sao cần Concatenation trước khi áp dụng $W^O$?

1. **Tổng hợp đa góc nhìn độc lập (Multi-Perspective Fusion):**  
   Mỗi head $i$ hoạt động trong một không gian con (subspace) có số chiều $d_k$ độc lập và có xu hướng chú ý vào những đặc trưng riêng (ví dụ: head 1 theo dõi quan hệ cú pháp chủ-vị, head 2 theo dõi từ phủ định và từ mang cảm xúc). Phép `Concat` đóng vai trò tập hợp toàn bộ các phát hiện độc lập này vào một vector duy nhất mà không làm triệt tiêu hay làm mờ bất kỳ kênh thông tin nào.
   
2. **Tạo điều kiện cho tương tác chéo giữa các Head (Cross-Head Mixing):**  
   Nếu không ghép nối mà giữ các head riêng rẽ, mô hình sẽ không có cơ chế giao tiếp giữa các head. Khi ghép nối thành vector kích thước $h \cdot d_k = d_{model}$ và nhân với ma trận học được $W^O$, mỗi hàng của $W^O$ sẽ tạo ra một tổ hợp tuyến tính kết hợp thông tin từ tất cả các head. Điều này cho phép thông tin ở Head 1 tác động và bổ trợ cho thông tin ở Head 2.

3. **Tương thích kích thước với đường tắt Residual Connection:**  
   Khối Transformer Encoder sử dụng cơ chế cộng đường tắt $X + \text{MHA}(X)$. Do đầu vào $X$ có kích thước $\mathbb{R}^{T \times d_{model}}$, đầu ra của khối Attention bắt buộc phải có cùng kích thước $\mathbb{R}^{T \times d_{model}}$. Phép chiếu $W^O$ chuẩn hóa kích thước không gian để phép cộng số học với residual connection diễn ra hợp lệ.

---

## 5. Bài tập Tính toán 3: Mã hóa Vị trí Sinusoidal Positional Encoding (Mục 10)

### 5.1. Công thức Toán học
Kiến trúc Transformer sử dụng các hàm lượng giác hình sin và cosin với tần số giảm dần theo số chiều (Vaswani et al., 2017):
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i / d}}\right)$$
Trong đó:
- $pos$ là chỉ số vị trí của token trong chuỗi ($pos = 0, 1, 2, \dots$).
- $i$ là chỉ số chiều đặc trưng ($0 \le 2i < d$).
- $d$ là số chiều vector biểu diễn ($d_{model} = 4$).

### 5.2. Xác định các hệ số phân kỳ (Divisors)
Với $d = 4$, ta có $i \in \{0, 1\}$:
- Với $i = 0$ (chiều 0 và 1):
  $$\text{Divisor}_0 = 10000^{2(0)/4} = 10000^0 = 1$$
- Với $i = 1$ (chiều 2 và 3):
  $$\text{Divisor}_1 = 10000^{2(1)/4} = 10000^{0.5} = \sqrt{10000} = 100$$

### 5.3. Tính toán chi tiết cho các vị trí $pos = 0, 1, 2$

#### Vị trí $pos = 0$:
- Chiều 0 ($2i = 0$): $PE(0, 0) = \sin\left(\frac{0}{1}\right) = \sin(0) = 0.0000$
- Chiều 1 ($2i+1 = 1$): $PE(0, 1) = \cos\left(\frac{0}{1}\right) = \cos(0) = 1.0000$
- Chiều 2 ($2i = 2$): $PE(0, 2) = \sin\left(\frac{0}{100}\right) = \sin(0) = 0.0000$
- Chiều 3 ($2i+1 = 3$): $PE(0, 3) = \cos\left(\frac{0}{100}\right) = \cos(0) = 1.0000$

$$\implies PE(0) = [0.0000, 1.0000, 0.0000, 1.0000]$$

#### Vị trí $pos = 1$:
- Chiều 0: $PE(1, 0) = \sin\left(\frac{1}{1}\right) = \sin(1\text{ rad}) \approx 0.8415$
- Chiều 1: $PE(1, 1) = \cos\left(\frac{1}{1}\right) = \cos(1\text{ rad}) \approx 0.5403$
- Chiều 2: $PE(1, 2) = \sin\left(\frac{1}{100}\right) = \sin(0.01\text{ rad}) \approx 0.0100$
- Chiều 3: $PE(1, 3) = \cos\left(\frac{1}{100}\right) = \cos(0.01\text{ rad}) \approx 0.9999$

$$\implies PE(1) \approx [0.8415, 0.5403, 0.0100, 0.9999]$$

#### Vị trí $pos = 2$:
- Chiều 0: $PE(2, 0) = \sin\left(\frac{2}{1}\right) = \sin(2\text{ rad}) \approx 0.9093$
- Chiều 1: $PE(2, 1) = \cos\left(\frac{2}{1}\right) = \cos(2\text{ rad}) \approx -0.4161$
- Chiều 2: $PE(2, 2) = \sin\left(\frac{2}{100}\right) = \sin(0.02\text{ rad}) \approx 0.0200$
- Chiều 3: $PE(2, 3) = \cos\left(\frac{2}{100}\right) = \cos(0.02\text{ rad}) \approx 0.9998$

$$\implies PE(2) \approx [0.9093, -0.4161, 0.0200, 0.9998]$$

### 5.4. Bảng tổng hợp Ma trận Positional Encoding

| Vị trí ($pos$) | Chiều 0 ($\sin$, $\omega=1$) | Chiều 1 ($\cos$, $\omega=1$) | Chiều 2 ($\sin$, $\omega=0.01$) | Chiều 3 ($\cos$, $\omega=0.01$) |
|:---:|:---:|:---:|:---:|:---:|
| **$pos = 0$** | $0.0000$ | $1.0000$ | $0.0000$ | $1.0000$ |
| **$pos = 1$** | $0.8415$ | $0.5403$ | $0.0100$ | $0.9999$ |
| **$pos = 2$** | $0.9093$ | $-0.4161$ | $0.0200$ | $0.9998$ |

### 5.5. Nhận xét định lượng
1. **Tính độc nhất (Uniqueness):** Mỗi vị trí $pos$ sở hữu một vector vị trí hoàn toàn phân biệt. Khoảng cách Euclid giữa các vị trí liền kề:
   $$\|PE(0) - PE(1)\|_2 \approx \sqrt{0.8415^2 + (-0.4597)^2 + 0.0100^2 + (-0.0001)^2} \approx 0.959$$
   $$\|PE(1) - PE(2)\|_2 \approx \sqrt{(0.9093-0.8415)^2 + (-0.4161-0.5403)^2 + 0.0100^2} \approx 0.959$$
2. **Đặc trưng đa tần số (Multi-scale Frequencies):** Các chiều thấp có bước sóng ngắn ($\lambda = 2\pi$), biến thiên nhanh theo từng bước nhảy của $pos$, giúp phân biệt các token lân cận. Các chiều cao có bước sóng cực dài ($\lambda = 200\pi$), biến thiên rất chậm, giúp duy trì đặc trưng vị trí tương đối trên toàn bộ văn bản dài.

---

## 6. Bài tập Tính toán 4: Mạng Feed-Forward từng vị trí (Mục 14)

### 6.1. Đề bài
Cho vector biểu diễn tại một vị trí:
$$x = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \in \mathbb{R}^{2 \times 1}$$

Các tham số của tầng tuyến tính thứ nhất:
$$W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad b_1 = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$$

### 6.2. Tính toán từng bước
1. **Biến đổi tuyến tính tầng 1:**
   $$z_1 = W_1 x + b_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

2. **Kích hoạt phi tuyến qua hàm ReLU:**
   $$h = \text{ReLU}(z_1) = \text{ReLU}\left(\begin{bmatrix} 1 \\ 1 \end{bmatrix}\right) = \begin{bmatrix} \max(0, 1) \\ \max(0, 1) \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

3. **Tầng tuyến tính thứ hai $y = W_2 h + b_2$:**
   Giả sử các tham số của tầng 2 là:
   $$W_2 = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}, \quad b_2 = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$$
   Khi đó:
   $$y = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} + \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 2(1) - 1(1) \\ 1(1) + 1(1) \end{bmatrix} + \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix}$$

### 6.3. Mục tiêu Nhận thức: Bản chất của Position-wise FFN
FFN trong Transformer **không phải là một cơ chế attention khác**. Trong khi Self-Attention cho phép các token khác nhau trong câu trao đổi và tổng hợp thông tin chéo lẫn nhau (trộn theo trục không gian thời gian $T$), thì FFN tác động hoàn toàn độc lập lên từng token riêng lẻ (position-wise) với cùng một bộ trọng số $(W_1, b_1, W_2, b_2)$.  
FFN đóng vai trò mở rộng số chiều lên $d_{ff}$ (thường bằng $4 \times d_{model}$), áp dụng biến đổi phi tuyến sâu, và hoạt động tương tự một bộ nhớ kết hợp (key-value associative memory) giúp lưu trữ và kích hoạt tri thức mẫu của mô hình.

---

## 7. Phân tích Residual Connection & Layer Normalization (Mục 15 & 16)

### 7.1. Residual Connection: $y = x + F(x)$
#### Câu hỏi: Nếu chỉ dùng $F(x)$, điều gì khác biệt so với việc dùng $x + F(x)$?

| Tiêu chí | Mô hình không có Residual: $y = F(x)$ | Mô hình có Residual: $y = x + F(x)$ |
|:---|:---|:---|
| **Mục tiêu học của hàm $F$** | $F$ phải gánh toàn bộ trách nhiệm tái tạo biểu diễn đầu ra đầy đủ. Nếu $F$ học kém, thông tin gốc bị xóa sổ. | $F$ chỉ cần học **phần sai lệch cần bổ sung** (residual $\Delta x = y - x$). Nếu khối không cần thiết, trọng số có thể học tiệm cận 0 để $y \approx x$. |
| **Lan truyền Gradient** | $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial F(x)}{\partial x}$. Khi mạng xếp chồng nhiều lớp, chuỗi tích Jacobian $\prod \frac{\partial F}{\partial x}$ dễ dẫn tới **triệt tiêu gradient (vanishing gradient)** hoặc bùng nổ gradient. | $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left( I + \frac{\partial F(x)}{\partial x} \right) = \frac{\partial \mathcal{L}}{\partial y} + \frac{\partial \mathcal{L}}{\partial y}\frac{\partial F(x)}{\partial x}$. Nhờ có ma trận đơn vị $I$, gradient luôn có một "đường cao tốc" truyền thẳng về các tầng đầu mà không bị suy hao. |
| **Độ sâu mô hình** | Rất khó tối ưu khi xếp chồng quá 4-6 lớp. | Cho phép huấn luyện ổn định các mạng cực sâu (12, 24, thậm chí hàng trăm lớp như BERT, GPT). |

---

### 7.2. Layer Normalization (LN)
Khối Transformer sử dụng Layer Normalization thay vì Batch Normalization:
$$\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$
$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

#### Điểm khác biệt mấu chốt so với Batch Normalization (BN):
1. **Trục tính toán:**
   - **BatchNorm:** Tính thống kê $\mu, \sigma^2$ xuyên qua toàn bộ các mẫu trong một mini-batch theo từng kênh/đặc trưng riêng lẻ.
   - **LayerNorm:** Tính thống kê $\mu, \sigma^2$ độc lập trên chính bản thân các đặc trưng $d_{model}$ của **từng token đơn lẻ**.
2. **Khắc phục vấn đề biến thiên độ dài chuỗi (Variable Lengths):**  
   Trong NLP, các câu có độ dài $T$ khác nhau và chứa nhiều padding `<PAD>`. BatchNorm phụ thuộc rất lớn vào batch size và gặp nhiễu nghiêm trọng khi batch nhỏ hoặc chuỗi có độ dài thay đổi. LayerNorm hoàn toàn độc lập với kích thước batch và độ dài chuỗi, đảm bảo biểu diễn tại mỗi token luôn duy trì phân phối chuẩn hóa có trung bình $\approx 0$ và phương sai $\approx 1$, giúp quá trình tối ưu diễn ra nhịp nhàng và mượt mà.

---

## 8. Bảng Kiểm tra Shape Bắt buộc (Mục 18 — Shape Verification)

Giả sử cấu hình thử nghiệm chuẩn theo quy định của LAB 07:
- Kích thước batch: $B = 8$
- Độ dài chuỗi: $T = 100$
- Chiều không gian mô hình: $d_{model} = 128$
- Số lượng Attention Heads: $n_{heads} = 4$
- Chiều của mỗi head: $d_k = \frac{d_{model}}{n_{heads}} = \frac{128}{4} = 32$
- Chiều mở rộng của khối Feed-Forward: $d_{ff} = 256$

### Bảng theo dõi Shape chi tiết qua từng tầng biến đổi:

| STT | Bước thực hiện / Tầng mạng | Ký hiệu Toán học | Kích thước Tensor (Shape) | Giải thích cấu trúc không gian |
|:---:|:---|:---:|:---:|:---|
| **1** | Đầu vào chuỗi | $X$ | `[8, 100, 128]` | $B$ câu, mỗi câu $T=100$ tokens, biểu diễn vector $d_{model}=128$. |
| **2** | Chiếu tuyến tính Q, K, V | $X W^Q, X W^K, X W^V$ | `[8, 100, 128]` | Nhân với $W \in \mathbb{R}^{128 \times 128}$. |
| **3** | Tách và chuyển vị Multi-Head | $Q, K, V$ | `[8, 4, 100, 32]` | Tách $128 \to 4 \times 32$, đưa $n_{heads}$ lên chiều thứ hai để tính song song. |
| **4** | Chuyển vị Key để nhân vô hướng | $K^T$ | `[8, 4, 32, 100]` | Chuyển vị 2 chiều cuối cùng của ma trận $K$. |
| **5** | Điểm số tương tác chú ý | $S = \frac{Q K^T}{\sqrt{d_k}}$ | `[8, 4, 100, 100]` | Tích vô hướng cặp token: $[100, 32] \times [32, 100] = [100, 100]$. |
| **6** | Trọng số chú ý sau Softmax | $A = \text{softmax}(S, -1)$ | `[8, 4, 100, 100]` | Chuẩn hóa xác suất phân phối trên trục token đích (tổng mỗi hàng = 1). |
| **7** | Nhân với Value trích xuất ngữ cảnh | $Z_{heads} = A V$ | `[8, 4, 100, 32]` | $[100, 100] \times [100, 32] = [100, 32]$ cho từng head. |
| **8** | Ghép nối các Heads (Concat) | $Z_{concat}$ | `[8, 100, 128]` | Chuyển vị `[8, 100, 4, 32]` và gộp chiều thành $[8, 100, 128]$. |
| **9** | Chiếu đầu ra MHA qua $W^O$ | $Z_{out} = Z_{concat} W^O$ | `[8, 100, 128]` | Nhân với $W^O \in \mathbb{R}^{128 \times 128}$, hòa trộn thông tin giữa các heads. |
| **10** | Add & Norm 1 (Sub-layer 1) | $X_1 = \text{LN}(X + Z_{out})$ | `[8, 100, 128]` | Cộng đường tắt Residual và chuẩn hóa LayerNorm. |
| **11** | Tầng FFN 1 (Mở rộng & ReLU) | $H_{ffn} = \text{ReLU}(X_1 W_1 + b_1)$ | `[8, 100, 256]` | Mở rộng số chiều lên $d_{ff} = 256$. |
| **12** | Tầng FFN 2 (Thu hẹp) | $Z_{ffn} = H_{ffn} W_2 + b_2$ | `[8, 100, 128]` | Thu hẹp số chiều trở lại $d_{model} = 128$. |
| **13** | Add & Norm 2 (Sub-layer 2) | $\text{Out} = \text{LN}(X_1 + Z_{ffn})$ | `[8, 100, 128]` | Cộng đường tắt Residual thứ hai và chuẩn hóa LayerNorm. |

> **Khẳng định kiểm chứng:** Kết quả kiểm tra bằng mã nguồn `transformer_block.py` đã xác nhận 100% các shape trên đều khớp hoàn toàn với thực thi tensor trong PyTorch, không có bất kỳ xung đột kích thước nào xảy ra.
