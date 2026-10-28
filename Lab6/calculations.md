# LAB 06 — Attention Mechanism: Lý thuyết & Bài tập tính toán

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — Năm học 2026  
**Trường:** ĐH Khoa học Tự nhiên — ĐHQGHN (VNU-HUS)  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Sinh viên thực hiện:** Nguyễn Trọng Thành  

---

## 1. Bối cảnh lý thuyết: Từ Recurrent Hidden State sang Context-Aware Representation

Ở **LAB 05**, việc phân loại chuỗi (sequence classification) dựa vào mạng hồi quy (RNN / LSTM / GRU):
$$x_1 \longrightarrow x_2 \longrightarrow \dots \longrightarrow x_T \implies h_T \implies \text{Classifier} \implies \hat{y}$$

Mô hình hồi quy cập nhật trạng thái ẩn tuần tự qua từng bước thời gian:
$$h_t = f(h_{t-1}, x_t)$$

### Điểm nghẽn thông tin (Information Bottleneck)
Khi độ dài chuỗi $T$ lớn, toàn bộ ngữ cảnh quá khứ buộc phải bị nén vào một vector cố định duy nhất $h_T$. Xét câu ví dụ:
> *"The movie was extremely long and the acting was sometimes boring, but the final scene was absolutely fantastic."*

Nếu tín hiệu quyết định sentiment nằm ở xa (cuối câu hoặc đầu câu), thông tin phải truyền qua rất nhiều bước trung gian. Mặc dù LSTM/GRU giảm thiểu vanishing gradient bằng các cổng (gates), trạng thái cuối $h_T$ vẫn chịu áp lực nén quá tải và thường bị thiên lệch về các token xuất hiện sau cùng (recency bias).

### Động lực của Cơ chế Attention (Attention Mechanism)
Thay vì ép toàn bộ chuỗi vào một vector đơn lẻ $h_T$, **Attention** cho phép mô hình nhìn lại toàn bộ các biểu diễn $h_1, h_2, \dots, h_T$ và tự động gán trọng số tập trung vào những phần quan trọng nhất đối với mục tiêu đang cần truy vấn:

$$\begin{matrix}
\text{Query } (q) \\
\downarrow \\
\text{So khớp với Keys } (k_1, \dots, k_T) \longrightarrow \text{Attention Scores } (s_1, \dots, s_T) \\
\downarrow \\
\text{Chuẩn hóa bằng Softmax} \longrightarrow \text{Attention Weights } (\alpha_1, \dots, \alpha_T) \\
\downarrow \\
\text{Tổng có trọng số Values } (v_1, \dots, v_T) \longrightarrow \text{Context Vector } (c)
\end{matrix}$$

### Vai trò của bộ ba Query, Key, Value
1. **Query ($Q$):** Đại diện cho *thông tin đang cần tìm kiếm* hoặc trạng thái truy vấn hiện tại.
2. **Key ($K$):** Đại diện cho *đặc trưng dùng để so khớp / đánh chỉ mục* của từng vị trí trong chuỗi.
3. **Value ($V$):** Chứa *nội dung thông tin thực tế* được trích xuất và kết hợp để tạo ra vector ngữ cảnh (context vector).

---

## 2. Bài tập tính toán chi tiết bằng tay

### Bài 1 — Attention scores (Mục 8)

**Đề bài:** Cho vector Query:
$$q = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$
và ma trận các Key:
$$K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \implies k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad k_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad k_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

**Lời giải:**
Điểm tương đồng Dot-product score được tính bằng tích vô hướng giữa Query và từng Key:
$$s_i = q^T k_i$$

- Với $k_1$:
  $$s_1 = q^T k_1 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 1 \cdot 1 + 2 \cdot 0 = 1$$
- Với $k_2$:
  $$s_2 = q^T k_2 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 1 \cdot 0 + 2 \cdot 1 = 2$$
- Với $k_3$:
  $$s_3 = q^T k_3 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 1 \cdot 1 + 2 \cdot 1 = 3$$

**Kết luận:** Vector attention scores đầy đủ là:
$$s = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$$

---

### Bài 2 — Softmax attention weights (Mục 9)

**Đề bài:** Cho vector score $s = [1, 2, 3]$. Tính $\alpha = \text{softmax}(s)$. Chứng minh $\sum_i \alpha_i = 1$ và giải thích tại sao phần tử có score lớn nhất nhận weight lớn nhất.

**Lời giải:**
Công thức hàm Softmax:
$$\alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^3 \exp(s_j)}$$

1. Tính hàm mũ cho từng thành phần:
   - $\exp(s_1) = \exp(1) = e^1 \approx 2.7182818$
   - $\exp(s_2) = \exp(2) = e^2 \approx 7.3890561$
   - $\exp(s_3) = \exp(3) = e^3 \approx 20.0855369$

2. Tổng các hàm mũ (mẫu số chuẩn hóa):
   $$\sum_{j=1}^3 \exp(s_j) = 2.7182818 + 7.3890561 + 20.0855369 = 30.1928748$$

3. Tính trọng số attention $\alpha_i$:
   - $\alpha_1 = \frac{2.7182818}{30.1928748} \approx 0.090031 \quad (9.00\%)$
   - $\alpha_2 = \frac{7.3890561}{30.1928748} \approx 0.244728 \quad (24.47\%)$
   - $\alpha_3 = \frac{20.0855369}{30.1928748} \approx 0.665241 \quad (66.52\%)$

Vector trọng số attention:
$$\alpha = \begin{bmatrix} 0.090031 & 0.244728 & 0.665241 \end{bmatrix}$$

**Kiểm tra tổng:**
$$\sum_{i=1}^3 \alpha_i = 0.090031 + 0.244728 + 0.665241 = 1.000000 \quad \text{(đúng)}$$

**Giải thích:**
Hàm số $f(x) = \exp(x)$ là hàm đồng biến nghiêm ngặt ($\frac{d}{dx} e^x = e^x > 0$ với mọi $x \in \mathbb{R}$). Do đó:
$$s_3 > s_2 > s_1 \implies \exp(s_3) > \exp(s_2) > \exp(s_1)$$
Khi chia cho cùng một mẫu số dương $\sum_j \exp(s_j)$, thứ tự được bảo toàn:
$$\alpha_3 > \alpha_2 > \alpha_1$$
Đặc biệt, tính chất lồi (convexity) của hàm mũ còn khuếch đại khoảng cách: khoảng chênh lệch giữa các score được nhân theo cấp số nhân, khiến phần tử lớn nhất $s_3 = 3$ chiếm ưu thế vượt trội ($66.52\%$).

---

### Bài 3 — Context vector (Mục 10)

**Đề bài:** Cho $\alpha = [0.1, 0.2, 0.7]$ và các Value vectors:
$$v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$
Tính vector ngữ cảnh $c = 0.1 v_1 + 0.2 v_2 + 0.7 v_3$. Cho biết Value nào đóng góp nhiều nhất?

**Lời giải:**
$$c = \sum_{i=1}^3 \alpha_i v_i = 0.1 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.2 \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0.7 \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$
Tính theo từng tọa độ:
- Chiều thứ nhất:
  $$c_{(1)} = 0.1(1) + 0.2(0) + 0.7(2) = 0.1 + 0.0 + 1.4 = 1.5$$
- Chiều thứ hai:
  $$c_{(2)} = 0.1(0) + 0.2(1) + 0.7(2) = 0.0 + 0.2 + 1.4 = 1.6$$

Vector ngữ cảnh:
$$c = \begin{bmatrix} 1.5 \\ 1.6 \end{bmatrix}$$

**Đóng góp của từng Value:**
- $v_1$ đóng góp: $0.1 \times [1, 0]^T = [0.1, 0.0]^T$
- $v_2$ đóng góp: $0.2 \times [0, 1]^T = [0.0, 0.2]^T$
- $v_3$ đóng góp: $0.7 \times [2, 2]^T = [1.4, 1.4]^T$

**Kết luận:** Value $v_3$ đóng góp nhiều nhất vào context vector vì nó vừa có trọng số attention cao nhất ($\alpha_3 = 0.7$, chiếm 70% tổng trọng số), vừa có độ lớn vector Value cao ($\|v_3\|_2 = \sqrt{8} \approx 2.83$).

---

### Bài 4 — Toàn bộ quy trình tính Attention (Mục 11)

**Đề bài:** 
- Query: $q = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
- Keys: $k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad k_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad k_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$
- Values: $v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 2 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

Thực hiện toàn bộ luồng: $q \to \text{scores} \to \text{softmax} \to \text{weights} \to \text{weighted sum} \to \text{context vector}$.

**Lời giải chi tiết:**

#### Bước 1: Tính Dot-product scores ($s_i = q^T k_i$)
- $s_1 = q^T k_1 = 1(1) + 0(0) = 1$
- $s_2 = q^T k_2 = 1(0) + 0(1) = 0$
- $s_3 = q^T k_3 = 1(1) + 0(1) = 1$

Vector scores thô:
$$s = \begin{bmatrix} 1 & 0 & 1 \end{bmatrix}$$

#### Bước 2: Tính Attention weights qua Softmax ($\alpha = \text{softmax}(s)$)
- $\exp(s_1) = e^1 \approx 2.718282$
- $\exp(s_2) = e^0 = 1.000000$
- $\exp(s_3) = e^1 \approx 2.718282$
- Tổng mẫu số: $\sum_j \exp(s_j) = 2e + 1 \approx 2(2.718282) + 1 = 6.436564$

Các trọng số attention:
- $\alpha_1 = \frac{e}{2e+1} = \frac{2.718282}{6.436564} \approx 0.422319 \quad (42.23\%)$
- $\alpha_2 = \frac{1}{2e+1} = \frac{1.000000}{6.436564} \approx 0.155362 \quad (15.54\%)$
- $\alpha_3 = \frac{e}{2e+1} = \frac{2.718282}{6.436564} \approx 0.422319 \quad (42.23\%)$

Kiểm tra: $\sum_i \alpha_i = 0.422319 + 0.155362 + 0.422319 = 1.000000$.

#### Bước 3: Tính Context vector qua Weighted sum ($c = \sum_i \alpha_i v_i$)
$$c = 0.422319 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.155362 \begin{bmatrix} 0 \\ 2 \end{bmatrix} + 0.422319 \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$
- Chiều thứ nhất:
  $$c_{(1)} = 0.422319(1) + 0.155362(0) + 0.422319(2) = 0.422319 + 0.844638 = 1.266957$$
- Chiều thứ hai:
  $$c_{(2)} = 0.422319(0) + 0.155362(2) + 0.422319(1) = 0.310724 + 0.422319 = 0.733043$$

Vector ngữ cảnh cuối cùng:
$$c \approx \begin{bmatrix} 1.266957 \\ 0.733043 \end{bmatrix}$$

*(Ghi chú: Nếu áp dụng scaled dot-product với $d_k = 2$, scores sẽ là $s / \sqrt{2} = [1/\sqrt{2}, 0, 1/\sqrt{2}] \approx [0.7071, 0, 0.7071]$; khi đó $\alpha \approx [0.4011, 0.1978, 0.4011]$ và $c \approx [1.2033, 0.7967]^T$).*

---

## 3. Scaling, Temperature và Tính ổn định số học (Numerical Stability)

### 3.1. Bài tập Scaling (Mục 14)

**Đề bài:** Cho $d_k = 4$ và $Q K^T = \begin{bmatrix} 4 & 2 & 0 \end{bmatrix}$. 
Tính scaled scores $s_{\text{scaled}} = \frac{Q K^T}{\sqrt{d_k}} = \frac{Q K^T}{\sqrt{4}}$. Tính softmax và so sánh với trường hợp không scaling.

**Lời giải:**
1. Scaled scores:
   $$s_{\text{scaled}} = \frac{\begin{bmatrix} 4 & 2 & 0 \end{bmatrix}}{2} = \begin{bmatrix} 2 & 1 & 0 \end{bmatrix}$$

2. So sánh Softmax:
   - **Trường hợp không scaling ($s = [4, 2, 0]$):**
     $$\exp(4) \approx 54.59815, \quad \exp(2) \approx 7.38906, \quad \exp(0) = 1.00000 \implies \sum \approx 62.98721$$
     $$\alpha_{\text{unscaled}} = \begin{bmatrix} \frac{54.59815}{62.98721} & \frac{7.38906}{62.98721} & \frac{1.00000}{62.98721} \end{bmatrix} \approx \begin{bmatrix} 0.866813 & 0.117310 & 0.015876 \end{bmatrix}$$
   - **Trường hợp có scaling ($s_{\text{scaled}} = [2, 1, 0]$):**
     $$\exp(2) \approx 7.38906, \quad \exp(1) \approx 2.71828, \quad \exp(0) = 1.00000 \implies \sum \approx 11.10734$$
     $$\alpha_{\text{scaled}} = \begin{bmatrix} \frac{7.38906}{11.10734} & \frac{2.71828}{11.10734} & \frac{1.00000}{11.10734} \end{bmatrix} \approx \begin{bmatrix} 0.665241 & 0.244728 & 0.090031 \end{bmatrix}$$

**Nhận xét:**
Scaling chia cho $\sqrt{d_k}$ làm **giảm độ chênh lệch biên độ** giữa các scores, từ đó làm phân phối softmax trở nên **phẳng hơn (softer/less peaked)**. Trọng số của phần tử lớn nhất giảm từ $86.68\%$ xuống $66.52\%$, trong khi các phần tử nhỏ hơn nhận được nhiều sự chú ý hơn.
Trong thực tế, khi $d_k$ lớn (ví dụ $d_k = 64$ hoặc $512$), tích vô hướng $q^T k = \sum_{i=1}^{d_k} q_i k_i$ có phương sai tăng tuyến tính theo $d_k$ ($\text{Var}(q^T k) = d_k$). Nếu không chia cho $\sqrt{d_k}$, giá trị score sẽ rất lớn, đẩy softmax vào vùng bão hòa cực đoan (gần giống hàm argmax/one-hot) khiến đạo hàm (gradient) triệt tiêu về gần 0 (vanishing gradient during backprop).

---

### 3.2. Nhiệt độ (Temperature) trong Softmax (Mục 15)

Công thức Softmax có tham số nhiệt độ $T > 0$:
$$\text{softmax}(s / T)_i = \frac{\exp(s_i / T)}{\sum_j \exp(s_j / T)}$$

Với cùng vector score $s = [4.0, 2.0, 0.0]$, ta khảo sát 3 mức nhiệt độ:
- $T = 0.5 \implies s / 0.5 = [8.0, 4.0, 0.0]$
- $T = 1.0 \implies s / 1.0 = [4.0, 2.0, 0.0]$
- $T = 2.0 \implies s / 2.0 = [2.0, 1.0, 0.0]$

**Bảng kết quả thực nghiệm:**

| Temperature ($T$) | Attention Weights ($\alpha$) | Max Weight | Entropy $H(p)$ (nats) | Nhận xét hành vi phân phối |
|:---:|:---|:---:|:---:|:---|
| **0.5** (Lạnh / Thấp) | $[0.981690, 0.017980, 0.000329]$ | **0.9817** | **0.0930** | Rất tập trung (peaked/sharp), gần như chọn lọc tuyệt đối phần tử max (argmax). |
| **1.0** (Chuẩn) | $[0.866813, 0.117310, 0.015876]$ | **0.8668** | **0.4411** | Phân phối gốc ban đầu, thiên về phần tử có score lớn nhất. |
| **2.0** (Nóng / Cao) | $[0.665241, 0.244728, 0.090031]$ | **0.6652** | **0.8324** | Phân phối mềm hơn, dàn đều (flatter), entropy tăng mạnh. |

*Ghi chú:* Khi $T \to 0^+$, softmax trở thành phép lấy argmax cứng. Khi $T \to +\infty$, softmax tiến tới phân phối đều (uniform distribution, entropy đạt cực đại $\ln 3 \approx 1.0986$). Phép chia $\sqrt{d_k}$ trong Scaled Dot-Product Attention có vai trò tương đương như việc đặt một mức nhiệt độ cố định $T = \sqrt{d_k}$.

---

### 3.3. Tính ổn định số học (Numerical Stability - Mục 17)

Khi tính trực tiếp $\exp(x)$ trong máy tính với kiểu số thực dấu phẩy động 64-bit (`float64`), ngưỡng tràn số cực đại là $\exp(709.78) \approx 1.79 \times 10^{308}$.
Nếu $x = [1000, 1001, 1002]$, tính trực tiếp $\exp(1002)$ sẽ dẫn đến `overflow -> inf`, và phép chia $\frac{\text{inf}}{\text{inf}}$ sẽ cho kết quả hỏng `[nan, nan, nan]`.

**Thuật toán Stable Softmax:**
$$\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}$$

**Chứng minh toán học:**
Với mọi hằng số $c \in \mathbb{R}$:
$$\frac{\exp(x_i - c)}{\sum_{j} \exp(x_j - c)} = \frac{\exp(x_i) \cdot \exp(-c)}{\sum_{j} (\exp(x_j) \cdot \exp(-c))} = \frac{\exp(x_i) \cdot e^{-c}}{e^{-c} \cdot \sum_j \exp(x_j)} = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
Vì thừa số chung $e^{-c}$ ở cả tử số và mẫu số triệt tiêu nhau hoàn toàn, phép tịnh tiến trừ đi $\max(x)$ **không làm thay đổi kết quả toán học** của Softmax.

**Kiểm nghiệm với $x = [1000, 1001, 1002]$:**
Đặt $c = \max(x) = 1002$:
$$x - 1002 = [-2, -1, 0]$$
- $\exp(-2) \approx 0.135335$
- $\exp(-1) \approx 0.367879$
- $\exp(0) = 1.000000$
- Tổng mẫu số: $0.135335 + 0.367879 + 1.000000 = 1.503215$

Trọng số thu được:
$$\alpha = \begin{bmatrix} \frac{0.135335}{1.503215} & \frac{0.367879}{1.503215} & \frac{1.000000}{1.503215} \end{bmatrix} = \begin{bmatrix} 0.090031 & 0.244728 & 0.665241 \end{bmatrix}$$
Kết quả hoàn toàn xác thực, triệt tiêu nguy cơ overflow mà không làm lệch xác suất.

---

## 4. Bài tập Self-Attention bằng tay (Mục 21)

**Đề bài:** Cho chuỗi gồm 3 tokens: `["I", "like", "cats"]`.
Giả sử ma trận embedding $X \in \mathbb{R}^{3 \times 2}$:
$$X = \begin{bmatrix} x_1^T \\ x_2^T \\ x_3^T \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \quad \begin{matrix} \text{ứng với "I"} \\ \text{ứng với "like"} \\ \text{ứng với "cats"} \end{matrix}$$
Chọn ma trận chiếu trực giao đơn vị $W_Q = W_K = W_V = I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ ($d = d_k = 2$).

**Lời giải chi tiết:**

1. **Chiếu ma trận Query, Key, Value:**
   $$Q = X W_Q = X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad K = X W_K = X, \quad V = X W_V = X$$

2. **Tính ma trận tích vô hướng $Q K^T$:**
   $$Q K^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 1(1)+0(0) & 1(0)+0(1) & 1(1)+0(1) \\ 0(1)+1(0) & 0(0)+1(1) & 0(1)+1(1) \\ 1(1)+1(0) & 1(0)+1(1) & 1(1)+1(1) \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$$

3. **Chia cho $\sqrt{d_k} = \sqrt{2} \approx 1.414214$:**
   $$S = \frac{Q K^T}{\sqrt{2}} \approx \begin{bmatrix} 0.7071 & 0.0000 & 0.7071 \\ 0.0000 & 0.7071 & 0.7071 \\ 0.7071 & 0.7071 & 1.4142 \end{bmatrix}$$

4. **Tính ma trận Attention Weights $A = \text{softmax}(S)$ theo từng hàng:**
   - **Hàng 1 (Token "I"):** $s_1 = [0.7071, 0, 0.7071]$
     $$\exp(s_1) \approx [2.0281, 1.0000, 2.0281], \quad \sum = 5.0562$$
     $$A_{1,:} \approx \begin{bmatrix} 0.4011 & 0.1978 & 0.4011 \end{bmatrix}$$
   - **Hàng 2 (Token "like"):** $s_2 = [0, 0.7071, 0.7071]$
     $$\exp(s_2) \approx [1.0000, 2.0281, 2.0281], \quad \sum = 5.0562$$
     $$A_{2,:} \approx \begin{bmatrix} 0.1978 & 0.4011 & 0.4011 \end{bmatrix}$$
   - **Hàng 3 (Token "cats"):** $s_3 = [0.7071, 0.7071, 1.4142]$
     $$\exp(s_3) \approx [2.0281, 2.0281, 4.1132], \quad \sum = 8.1694$$
     $$A_{3,:} \approx \begin{bmatrix} 0.2483 & 0.2483 & 0.5035 \end{bmatrix}$$

   Ma trận trọng số Attention đầy đủ:
   $$A \approx \begin{bmatrix} 0.4011 & 0.1978 & 0.4011 \\ 0.1978 & 0.4011 & 0.4011 \\ 0.2483 & 0.2483 & 0.5035 \end{bmatrix}$$
   *(Lưu ý: Tổng mỗi hàng bằng đúng 1.0)*

5. **Tính ma trận Context Representation $C = A V$:**
   $$C = A V = \begin{bmatrix} 0.4011 & 0.1978 & 0.4011 \\ 0.1978 & 0.4011 & 0.4011 \\ 0.2483 & 0.2483 & 0.5035 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$
   - Hàng 1:
     $$c_1 = 0.4011[1, 0]^T + 0.1978[0, 1]^T + 0.4011[1, 1]^T = [0.8022, 0.5989]^T$$
   - Hàng 2:
     $$c_2 = 0.1978[1, 0]^T + 0.4011[0, 1]^T + 0.4011[1, 1]^T = [0.5989, 0.8022]^T$$
   - Hàng 3:
     $$c_3 = 0.2483[1, 0]^T + 0.2483[0, 1]^T + 0.5035[1, 1]^T = [0.7518, 0.7518]^T$$

   $$C \approx \begin{bmatrix} 0.8022 & 0.5989 \\ 0.5989 & 0.8022 \\ 0.7518 & 0.7518 \end{bmatrix}$$

**Ý nghĩa:** Mỗi token sau phép self-attention không còn là một vector embedding cô lập, mà trở thành một **Contextual Representation** chứa đựng thông tin kết hợp từ toàn bộ chuỗi theo mức độ tương đồng giữa Query của nó với Key của các token khác.

---

## 5. Phân tích chi phí tính toán và Giới hạn $O(T^2)$ của Attention (Mục 25 & 29)

Ma trận Attention giữa một chuỗi có độ dài $T$ với chính nó có kích thước:
$$A \in \mathbb{R}^{T \times T} \implies \text{Số phần tử} = T^2$$

### Bảng định lượng kích thước và dung lượng bộ nhớ:

| Chiều dài chuỗi ($T$) | Kích thước ma trận $A$ | Số phần tử ($T^2$) | Bộ nhớ Float32 (1 head, 1 sample) | Bộ nhớ Batch=64, Heads=8 (Forward + Backward) |
|:---:|:---:|---:|---:|---:|
| **10** | $10 \times 10$ | 100 | 400 B | ~400 KB |
| **100** | $100 \times 100$ | 10,000 | 40 KB | ~40 MB |
| **500** | $500 \times 500$ | 250,000 | 1 MB | ~1 GB |
| **1,000** | $1,000 \times 1,000$ | 1,000,000 | 4 MB | ~4 GB |
| **10,000** | $10,000 \times 10,000$ | 100,000,000 | **400 MB** | **~400 GB (OOM - Tràn bộ nhớ)** |

### Điểm mấu chốt: "Attention không miễn phí" (Attention is not free)
- Khi độ dài chuỗi $T$ tăng gấp 10 lần (từ 1,000 lên 10,000), số lượng phần tử và chi phí tính toán nhân ma trận tăng **100 lần** ($10^2$).
- Trong khi mạng hồi quy RNN có độ phức tạp thời gian tuyến tính $O(T \cdot d^2)$ và bộ nhớ $O(T \cdot d)$, thì Full Self-Attention đòi hỏi chi phí không gian và thời gian bậc hai $O(T^2 \cdot d)$.
- Đây chính là rào cản tính toán lớn nhất khi mở rộng Attention lên các tài liệu dài hàng chục ngàn tokens, dẫn tới sự ra đời của các kiến trúc chuyên biệt sau này như FlashAttention, Sparse Attention hay Linear Attention.
