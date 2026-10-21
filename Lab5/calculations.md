# BÀI TẬP TÍNH TOÁN VÀ LÝ THUYẾT (CALCULATIONS & THEORY) — LAB 05

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I, 2026  
**Chủ đề:** Sequence Models: RNN, LSTM and GRU — From Static Representations to Sequential Modeling  
**Căn cứ tài liệu:** Yêu cầu tại các Mục 5, 6, 7, 8, 9, 16, 17, 18 trong `W5.pdf`

---

## 1. Bài 1 — Một bước Forward Computation của Simple RNN (Section 6)

### 1.1. Đề bài
Cho đầu vào và trạng thái ẩn trước đó tại bước thời gian \(t\):
\[
x_t = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad h_{t-1} = \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix}
\]
Ma trận trọng số đầu vào \(W_{xh}\) và ma trận trọng số hồi quy \(W_{hh}\):
\[
W_{xh} = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix}, \quad W_{hh} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix}
\]
Vector bias:
\[
b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\]

Yêu cầu:
1. Tính giá trị tiền kích hoạt (pre-activation vector): \(a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h\).
2. Tính trạng thái ẩn kế tiếp: \(h_t = \tanh(a_t)\).

---

### 1.2. Trình bày chi tiết từng bước tính toán

#### Bước 1: Tính thành phần tuyến tính từ đầu vào hiện tại \(W_{xh} x_t\)
Phép nhân ma trận \(W_{xh} \in \mathbb{R}^{2 \times 2}\) với vector cột \(x_t \in \mathbb{R}^{2 \times 1}\):
\[
W_{xh} x_t = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} (0.5 \times 1) + (0.2 \times 0) \\ (0.1 \times 1) + (0.4 \times 0) \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix}
\]

#### Bước 2: Tính thành phần tuyến tính từ trạng thái ẩn trước đó \(W_{hh} h_{t-1}\)
Phép nhân ma trận \(W_{hh} \in \mathbb{R}^{2 \times 2}\) với vector cột \(h_{t-1} \in \mathbb{R}^{2 \times 1}\):
\[
W_{hh} h_{t-1} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix} \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix} = \begin{bmatrix} (0.3 \times 0.5) + (0.1 \times 0.2) \\ (0.2 \times 0.5) + (0.5 \times 0.2) \end{bmatrix} = \begin{bmatrix} 0.15 + 0.02 \\ 0.10 + 0.10 \end{bmatrix} = \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix}
\]

#### Bước 3: Tính pre-activation vector \(a_t\)
Cộng các thành phần với vector bias \(b_h\):
\[
a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix} + \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.5 + 0.17 \\ 0.1 + 0.20 \end{bmatrix} = \begin{bmatrix} 0.67 \\ 0.30 \end{bmatrix}
\]

#### Bước 4: Áp dụng hàm kích hoạt phi tuyến \(\tanh\)
Hàm \(\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}\):
- Với tọa độ thứ nhất \(a_{t,1} = 0.67\):
  \[
  \tanh(0.67) = \frac{e^{0.67} - e^{-0.67}}{e^{0.67} + e^{-0.67}} \approx \frac{1.954237 - 0.511709}{1.954237 + 0.511709} = \frac{1.442528}{2.465946} \approx 0.584980
  \]
- Với tọa độ thứ hai \(a_{t,2} = 0.30\):
  \[
  \tanh(0.30) = \frac{e^{0.30} - e^{-0.30}}{e^{0.30} + e^{-0.30}} \approx \frac{1.349859 - 0.740818}{1.349859 + 0.740818} = \frac{0.609041}{2.090677} \approx 0.291313
  \]

Do đó, vector trạng thái ẩn \(h_t\) là:
\[
h_t = \begin{bmatrix} 0.584980 \\ 0.291313 \end{bmatrix}
\]

> [!NOTE]
> Kết quả tính toán giải tích này đã được kiểm chứng trùng khớp tuyệt đối (\(\text{allclose}\) với sai số \(< 10^{-7}\)) với cài đặt `SimpleRNNCell` tự xây dựng và module chuẩn `torch.nn.RNNCell` của PyTorch trong script `rnn_cell.py`.

---

## 2. Bài 2 — Luồng thông tin (Information Flow qua Sequence) (Section 7)

### 2.1. Quá trình biến đổi trạng thái qua chuỗi: `"I did not like this movie"`
Giả sử mô hình đọc tuần tự từ trái sang phải qua 6 bước thời gian:

1. **Bước 1 — Đọc `"I"`:**
   \[
   h_1 = \tanh(W_{xh} x_{\text{"I"}} + W_{hh} h_0 + b_h)
   \]
   \(h_1\) mang thông tin về chủ ngữ (người phát biểu / đối tượng trải nghiệm).
2. **Bước 2 — Đọc `"did"`:**
   \[
   h_2 = \tanh(W_{xh} x_{\text{"did"}} + W_{hh} h_1 + b_h)
   \]
   \(h_2\) tổng hợp cấu trúc trợ động từ trong thì quá khứ.
3. **Bước 3 — Đọc `"not"`:**
   \[
   h_3 = \tanh(W_{xh} x_{\text{"not"}} + W_{hh} h_2 + b_h)
   \]
   \(h_3\) tích hợp toán tử phủ định vào biểu diễn nội tại.
4. **Bước 4 — Đọc `"like"`:**
   \[
   h_4 = \tanh(W_{xh} x_{\text{"like"}} + W_{hh} h_3 + b_h)
   \]
   Đây là bước then chốt. Động từ mang cảm xúc tích cực `"like"` đi vào thông qua \(W_{xh} x_{\text{"like"}}\), nhưng nó được cộng trực tiếp với trạng thái trước đó \(W_{hh} h_3\) (vốn đã nén thông tin của từ `"not"`). Do đó, phép biến đổi phi tuyến \(\tanh\) có khả năng tạo ra một vector \(h_4\) biểu diễn ý nghĩa phủ định cảm xúc (*dislike* / *không thích*).
5. **Bước 5 — Đọc `"this"`:**
   \[
   h_5 = \tanh(W_{xh} x_{\text{"this"}} + W_{hh} h_4 + b_h)
   \]
   \(h_5\) định hướng bổ ngữ xác định cho cảm xúc phủ định vừa hình thành.
6. **Bước 6 — Đọc `"movie"`:**
   \[
   h_6 = \tanh(W_{xh} x_{\text{"movie"}} + W_{hh} h_5 + b_h)
   \]
   \(h_6\) là hidden state cuối cùng (\(h_T\)), đại diện cho toàn bộ câu, mang ngữ nghĩa hoàn chỉnh: người xem không thích bộ phim này.

### 2.2. Câu hỏi: Tại thời điểm model đọc `"like"`, thông tin `"not"` ảnh hưởng tới hidden state như thế nào?
- **Cơ chế:** Thông tin của token `"not"` được mã hóa vào vector trạng thái ẩn \(h_3\). Khi sang bước tiếp theo, \(h_3\) được nhân với ma trận trọng số hồi quy \(W_{hh}\) để tạo thành đầu vào context cho bước 4. Sau đó, \(W_{hh} h_3\) được cộng tuyến tính với vector biểu diễn của từ `"like"` (\(W_{xh} x_{\text{"like"}}\)) trước khi đưa qua hàm phi tuyến \(\tanh\).
- **Hệ quả ngữ nghĩa:** Nhờ phép biến đổi này, hidden state \(h_4\) không chỉ là biểu diễn riêng rẽ của từ `"like"`, mà là sự kết hợp có ngữ cảnh giữa toán tử phủ định và vị ngữ, cho phép mạng nơ-ron đảo chiều cực tính cảm xúc (từ tích cực thành tiêu cực).

### 2.3. So sánh với Average Word2Vec (LAB 04)
Trong phương pháp Average Word2Vec của LAB 04:
\[
\vec{v}_{\text{sentence}} = \frac{1}{T} \sum_{t=1}^T \vec{w}_t
\]
- Phép cộng vector có **tính chất giao hoán** (\(A + B = B + A\)).
- Vector trung bình gom tất cả các từ trong câu mà không quan tâm `"not"` đứng trước `"like"` hay sau `"like"`. Thậm chí, vector của `"not"` (thường mang tính chức năng tổng quát) khi cộng trung bình với `"like"` (mang tính tích cực mạnh) dễ bị lấn át bởi các từ vựng khác, khiến mô hình phân loại (Logistic Regression) dễ nhầm lẫn câu thành tích cực.
- Ngược lại, RNN lưu giữ thông tin vị trí thông qua chuỗi trạng thái ẩn cập nhật theo thời gian, giải quyết được khiếm khuyết căn bản của mô hình túi từ (Bag-of-Words).

---

## 3. Bài 3 — Phân biệt thứ tự từ (Sequence Order) (Section 8)

Cho hai câu:
- **S1:** `"dog bites man"` (chó cắn người — tin bình thường)
- **S2:** `"man bites dog"` (người cắn chó — tin giật gân, bất thường)

### 3.1. Average Word2Vec có thể phân biệt hai câu này không?
- **Trả lời:** **KHÔNG THỂ.**
- **Giải thích:** Cả hai câu S1 và S2 đều có tập hợp đa tập từ vựng (multiset of tokens) hoàn toàn giống nhau: \(\{\text{"dog"}, \text{"bites"}, \text{"man"}\}\).
  Vì phép cộng vector có tính giao hoán:
  \[
  \vec{v}_{\text{avg}}(S1) = \frac{1}{3} (\vec{w}_{\text{dog}} + \vec{w}_{\text{bites}} + \vec{w}_{\text{man}}) = \frac{1}{3} (\vec{w}_{\text{man}} + \vec{w}_{\text{bites}} + \vec{w}_{\text{dog}}) = \vec{v}_{\text{avg}}(S2)
  \]
  Hai câu có biểu diễn vector đồng nhất tuyệt đối trong không gian đặc trưng, nên bất kỳ bộ phân loại tuyến tính nào (như Logistic Regression hay SVM) cũng đều đưa ra cùng một đầu ra dự đoán.

### 3.2. RNN phân biệt hai câu này dựa trên cơ chế mô hình như thế nào?
- **Trả lời:** RNN **HOÀN TOÀN CÓ KHẢ NĂNG** phân biệt hai câu dựa trên cơ chế cập nhật trạng thái có thứ tự.
- **Giải thích theo cơ chế nội tại:**
  - **Với câu S1 (`"dog bites man"`):**
    \[
    \begin{aligned}
    h_1^{(1)} &= \tanh(W_{xh} x_{\text{dog}} + W_{hh} h_0 + b_h) \\
    h_2^{(1)} &= \tanh(W_{xh} x_{\text{bites}} + W_{hh} h_1^{(1)} + b_h) \\
    h_3^{(1)} &= \tanh(W_{xh} x_{\text{man}} + W_{hh} h_2^{(1)} + b_h)
    \end{aligned}
    \]
  - **Với câu S2 (`"man bites dog"`):**
    \[
    \begin{aligned}
    h_1^{(2)} &= \tanh(W_{xh} x_{\text{man}} + W_{hh} h_0 + b_h) \\
    h_2^{(2)} &= \tanh(W_{xh} x_{\text{bites}} + W_{hh} h_1^{(2)} + b_h) \\
    h_3^{(2)} &= \tanh(W_{xh} x_{\text{dog}} + W_{hh} h_2^{(2)} + b_h)
    \end{aligned}
    \]
  - Vì \(x_{\text{dog}} \neq x_{\text{man}}\), ngay từ bước 1 ta đã có \(h_1^{(1)} \neq h_1^{(2)}\).
  - Khi sang bước 2 (từ `"bites"`), dù đầu vào \(x_{\text{bites}}\) giống nhau, nhưng trạng thái ẩn trước đó khác nhau, dẫn tới \(h_2^{(1)} \neq h_2^{(2)}\).
  - Ở bước 3, vừa có đầu vào khác nhau (\(x_{\text{man}}\) vs \(x_{\text{dog}}\)), vừa có context trước đó khác nhau, dẫn tới hai vector trạng thái cuối cùng khác biệt rõ rệt:
    \[
    h_3^{(1)} \neq h_3^{(2)}
    \]
  - Khi đưa qua lớp phân loại \(z = W_{hy} h_T + b_y\), mô hình có thể gán hai nhãn khác nhau cho S1 và S2 (ví dụ: phân biệt chủ ngữ và tân ngữ, hoặc phát hiện câu dị thường).

---

## 4. Hiện tượng Vanishing Gradient và Giới hạn Phụ thuộc Xa (Section 9)

### 4.1. Bản chất toán học của Backpropagation Through Time (BPTT)
Trong RNN đơn giản, hàm mất mát \(L\) được tính ở cuối chuỗi hoặc tổng qua các bước. Khi tính đạo hàm của hàm mất mát đối với trạng thái ẩn \(h_t\) tại một thời điểm quá khứ \(t < T\), theo quy tắc đạo hàm chuỗi (chain rule):
\[
\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \frac{\partial h_T}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}}
\]
Trong đó ma trận Jacobian giữa hai bước liên tiếp là:
\[
\frac{\partial h_k}{\partial h_{k-1}} = \operatorname{diag}\left(1 - \tanh^2(a_k)\right) W_{hh}^T
\]

### 4.2. Giải thích: Vì sao sequence càng dài càng khiến RNN khó học dependency ở xa?
1. **Vanishing Gradient (Biến mất đạo hàm):**
   - Đạo hàm của hàm kích hoạt \(\tanh\) thỏa mãn \(0 < 1 - \tanh^2(z) \le 1\), giá trị này đạt cực đại là 1 tại \(z = 0\) và nhanh chóng tiệm cận 0 khi \(|z|\) lớn.
   - Nếu các trị riêng (eigenvalues) lớn nhất của ma trận hồi quy \(W_{hh}\) nhỏ hơn 1, thì chuẩn của ma trận Jacobian:
     \[
     \left\| \frac{\partial h_k}{\partial h_{k-1}} \right\| < 1
     \]
   - Tích của \((T - t)\) ma trận có chuẩn nhỏ hơn 1 sẽ suy giảm theo hàm mũ về 0 khi khoảng cách \((T - t)\) tăng lên:
     \[
     \left\| \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}} \right\| \to 0 \quad \text{khi } (T - t) \gg 1
     \]
   - Kết quả là gradient truyền ngược về các token ở đầu chuỗi bị tiêu biến gần như hoàn toàn. Trọng số mô hình không nhận được tín hiệu cập nhật tương ứng với các từ xuất hiện sớm trong văn bản.
2. **Exploding Gradient (Bùng nổ đạo hàm):**
   - Ngược lại, nếu trị riêng lớn nhất của \(W_{hh} > 1\), tích lũy Jacobian qua nhiều bước có thể tăng theo hàm mũ dẫn tới tràn số (\(\text{NaN}\)). Hiện tượng này thường được giải quyết bằng kỹ thuật **Gradient Clipping** (\(\|\nabla\| \le \text{threshold}\)).
3. **Ý nghĩa thực tế trong NLP:**
   - Khi phân loại một bài đánh giá phim dài 200–500 từ, nếu lời kết luận then chốt nằm ở đầu bài ("A masterpiece...") nhưng phần sau là tóm tắt dài dòng, Simple RNN sẽ "quên" mất tín hiệu đầu câu do vanishing gradient, dẫn đến phân loại sai.

---

## 5. Bài tập tính toán — LSTM Gates (Section 16 & 17)

### 5.1. Đề bài
Cho các giá trị tại một timestep \(t\) của tế bào LSTM:
- Cổng quên (Forget gate): \(f_t = 0.9\)
- Cổng vào (Input gate): \(i_t = 0.2\)
- Trạng thái ứng viên (Candidate state): \(\tilde{c}_t = 0.5\)
- Trạng thái ô nhớ trước đó (Previous cell state): \(c_{t-1} = 0.8\)

Công thức cập nhật cell state của LSTM:
\[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
\]

---

### 5.2. Các bước tính toán cụ thể
1. **Tính lượng thông tin giữ lại từ quá khứ:**
   \[
   f_t \cdot c_{t-1} = 0.9 \times 0.8 = 0.72
   \]
2. **Tính lượng thông tin mới được nạp vào:**
   \[
   i_t \cdot \tilde{c}_t = 0.2 \times 0.5 = 0.10
   \]
3. **Cập nhật cell state hiện tại \(c_t\):**
   \[
   c_t = 0.72 + 0.10 = 0.82
   \]

---

### 5.3. Ý nghĩa điều khiển của Forget Gate
- **Khi \(f_t \approx 1\):**
  - Ta có \(f_t c_{t-1} \approx c_{t-1}\). Mô hình có xu hướng **bảo toàn gần như nguyên vẹn thông tin cũ** từ các bước thời gian trước.
  - Về mặt toán học, phép cộng tuyến tính \(c_t = c_{t-1} + \dots\) tạo thành một "đường cao tốc" (gradient highway), giúp đạo hàm \(\frac{\partial c_t}{\partial c_{t-1}} \approx 1\). Nhờ vậy, gradient có thể lan truyền ngược qua hàng trăm timestep mà không bị tiêu biến, giải quyết bài toán phụ thuộc xa (long-range dependencies).
- **Khi \(f_t \approx 0\):**
  - Ta có \(f_t c_{t-1} \approx 0\). Mô hình chủ động **xóa bỏ / loại bỏ hoàn toàn thông tin cũ** đã lưu trong ô nhớ.
  - Điều này cực kỳ hữu ích khi chuyển đổi ngữ cảnh ngôn ngữ (ví dụ: kết thúc một đoạn văn, gặp dấu chấm câu, hoặc đổi chủ đề phản biện), cho phép cell state được "làm mới" để tiếp nhận thông tin từ câu tiếp theo mà không bị nhiễu bởi quá khứ.

---

## 6. So sánh cơ chế kiến trúc: RNN vs LSTM vs GRU (Section 18 & 25)

| Tiêu chí | Simple RNN | LSTM (Long Short-Term Memory) | GRU (Gated Recurrent Unit) |
|---|---|---|---|
| **Cơ chế hồi quy** | Có (\(h_t\)) | Có (\(h_t\)) | Có (\(h_t\)) |
| **Hidden state** | Duy nhất vector \(h_t\) | Duy nhất vector \(h_t\) | Duy nhất vector \(h_t\) |
| **Cell state riêng biệt** | **Không** | **Có** (\(c_t\) hoạt động như băng chuyền bộ nhớ) | **Không** (chỉ có \(h_t\)) |
| **Hệ thống cổng (Gates)** | **Không có cổng** | **3 cổng:** Forget (\(f_t\)), Input (\(i_t\)), Output (\(o_t\)) | **2 cổng:** Reset gate (\(r_t\)), Update gate (\(z_t\)) |
| **Số bộ tham số** | \(1 \times (d_h \times d_x + d_h^2 + d_h)\) | \(4 \times (d_h \times d_x + d_h^2 + d_h)\) | \(3 \times (d_h \times d_x + d_h^2 + d_h)\) |
| **Khả năng nhớ xa** | Kém (bị vanishing gradient nghiêm trọng) | Rất tốt (nhờ cơ chế cộng cell state) | Tốt tương đương LSTM (ít tham số hơn) |
| **Độ phức tạp tính toán** | Thấp nhất, train nhanh | Cao nhất, tốn bộ nhớ và thời gian | Trung bình (nhanh hơn LSTM ~25–30%) |
| **Động lực dẫn tới Attention** | Nén thông tin vào 1 vector cố định gây nghẽn cổ chai (bottleneck) | Vẫn bị giới hạn khi chuỗi quá dài (>500 từ) và phải tính toán tuần tự | Vẫn tuần tự không thể song song hóa như Transformer |

---
*Bản tính toán này được biên soạn đầy đủ, chính xác phục vụ nộp bài tập và bảo vệ kiểm tra cá nhân môn học NLP 2026.*
