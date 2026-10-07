# Calculations — LAB 03: Word Representations and Embeddings

Họ và tên sinh viên: Nguyễn Trọng Thành  
Mã sinh viên: (Điền MSV của bạn)  
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I, 2026  
Mục tiêu: Hoàn thành các bài tập tính toán bằng tay theo yêu cầu tài liệu `W3.pdf`.

---

## Bài 1 — Co-occurrence Matrix & Context Vectors

### 1. Ngữ liệu (Corpus)
```text
(1) the cat eats fish
(2) the dog eats fish
(3) the cat likes milk
(4) the dog likes meat
```

### 2. Thiết lập tham số
- **Cửa sổ ngữ cảnh (Context window):** $k = 1$ (xét 1 từ liền trước và 1 từ liền sau).
- **Tập từ vựng (Vocabulary):** Xét tập từ vựng chính (content words) theo thứ tự chuẩn trong đề bài `W3.pdf` (Mục 5):
  $$\mathcal{V} = [\text{cat, dog, eats, likes, fish, milk, meat}]$$
  *(Kích thước từ vựng $|\mathcal{V}| = 7$. Từ dừng `the` đóng vai trò mạo từ bổ ngữ ở đầu câu, nếu tính cả `the` ta có $|\mathcal{V}_{\text{full}}| = 8$, sẽ được phân tích đối chiếu bên dưới).*

### 3. Phân tích ngữ cảnh lân cận ($k=1$) cho từng Target Word
- **`cat`**:
  - Câu (1): trước `cat` là `the` (bỏ qua nếu không tính stopwords), sau `cat` là `eats`.
  - Câu (3): trước `cat` là `the`, sau `cat` là `likes`.
  - Từ lân cận thuộc $\mathcal{V}$: `eats` (1 lần), `likes` (1 lần).
- **`dog`**:
  - Câu (2): trước `dog` là `the`, sau `dog` là `eats`.
  - Câu (4): trước `dog` là `the`, sau `dog` là `likes`.
  - Từ lân cận thuộc $\mathcal{V}$: `eats` (1 lần), `likes` (1 lần).
- **`eats`**:
  - Câu (1): trước `eats` là `cat` (1 lần), sau `eats` là `fish` (1 lần).
  - Câu (2): trước `eats` là `dog` (1 lần), sau `eats` là `fish` (1 lần).
  - Từ lân cận thuộc $\mathcal{V}$: `cat` (1 lần), `dog` (1 lần), `fish` (2 lần).
- **`likes`**:
  - Câu (3): trước `likes` là `cat` (1 lần), sau `likes` là `milk` (1 lần).
  - Câu (4): trước `likes` là `dog` (1 lần), sau `likes` là `meat` (1 lần).
  - Từ lân cận thuộc $\mathcal{V}$: `cat` (1 lần), `dog` (1 lần), `milk` (1 lần), `meat` (1 lần).

### 4. Bảng vector ngữ cảnh hoàn chỉnh

Thứ tự các chiều: `[cat, dog, eats, likes, fish, milk, meat]`

| Target | Context occurrences | Context Vector $\vec{w}$ |
|---|---|---|
| **cat** | eats (1), likes (1) | `[0, 0, 1, 1, 0, 0, 0]` |
| **dog** | eats (1), likes (1) | `[0, 0, 1, 1, 0, 0, 0]` |
| **eats** | cat (1), dog (1), fish (2) | `[1, 1, 0, 0, 2, 0, 0]` |
| **likes** | cat (1), dog (1), milk (1), meat (1) | `[1, 1, 0, 0, 0, 1, 1]` |

*(Ghi chú mở rộng: Nếu tính thêm từ `the` vào vị trí đầu tiên của từ điển: `[the, cat, dog, eats, likes, fish, milk, meat]`, ta có $\vec{v}_{\text{cat}} = [2, 0, 0, 1, 1, 0, 0, 0]$ và $\vec{v}_{\text{dog}} = [2, 0, 0, 1, 1, 0, 0, 0]$. Cả 2 vector vẫn hoàn toàn trùng khớp nhau).*

> **Ý nghĩa phân tích:** Mặc dù `cat` và `dog` không bao giờ đồng xuất hiện trong cùng một câu, nhưng vector ngữ cảnh của chúng hoàn toàn trùng khớp ($\vec{v}_{\text{cat}} = \vec{v}_{\text{dog}} \implies \cos(\text{cat}, \text{dog}) = 1.0$). Đây là minh chứng mẫu mực cho **Distributional Hypothesis**: hai từ xuất hiện trong cùng phân bố ngữ cảnh sẽ có biểu diễn ngữ nghĩa tương đồng.

---

## Bài 2 — Cosine Similarity & Vector Magnitude

### 1. Đề bài
Cho hai vector:
$$x = [1, 2, 1]$$
$$y = [2, 4, 2]$$

### 2. Các bước tính toán chi tiết
- **Tích vô hướng (Dot Product):**
  $$\text{dot}(x, y) = x \cdot y = (1 \times 2) + (2 \times 4) + (1 \times 2) = 2 + 8 + 2 = 12$$

- **Độ dài Euclid của vector $x$ ($\|x\|$):**
  $$\|x\| = \sqrt{1^2 + 2^2 + 1^2} = \sqrt{1 + 4 + 1} = \sqrt{6} \approx 2.4494897$$

- **Độ dài Euclid của vector $y$ ($\|y\|$):**
  $$\|y\| = \sqrt{2^2 + 4^2 + 2^2} = \sqrt{4 + 16 + 4} = \sqrt{24} = \sqrt{4 \times 6} = 2\sqrt{6} \approx 4.8989795$$

- **Cosine Similarity:**
  $$\cos(x, y) = \frac{\text{dot}(x, y)}{\|x\| \cdot \|y\|} = \frac{12}{\sqrt{6} \cdot 2\sqrt{6}} = \frac{12}{2 \times 6} = \frac{12}{12} = 1.0$$

### 3. Ý nghĩa của kết quả
- **Về mặt hình học:** Hai vector $x$ và $y$ cùng phương, cùng hướng trong không gian 3 chiều ($y = 2x$, góc giữa hai vector $\theta = 0^\circ \implies \cos(0^\circ) = 1.0$).
- **Về mặt ngữ nghĩa NLP:** Độ lớn (magnitude $\|x\|$) của vector thường phản ánh **tần suất xuất hiện tuyệt đối** (frequency) của từ trong văn bản (ví dụ từ $y$ xuất hiện gấp đôi từ $x$). Trong khi đó, Cosine Similarity triệt tiêu yếu tố độ lớn thông qua phép chuẩn hóa độ dài ($\ell_2$-normalization), chỉ đo lường **hướng tương đối của phân bố ngữ cảnh**.
- **Kết luận:** Hai từ có tần suất xuất hiện lệch nhau nhưng tỷ lệ kết hợp với các từ ngữ cảnh tương đồng thì vẫn có độ tương đồng ngữ nghĩa bằng $1.0$.

---

## Bài 3 — So sánh Semantic Similarity

### 1. Đề bài
Cho 3 vector embedding giả định:
- $v_{\text{doctor}} = [0.8, 0.1, 0.7]$
- $v_{\text{physician}} = [0.7, 0.2, 0.8]$
- $v_{\text{banana}} = [-0.2, 0.9, -0.1]$

### 2. Dự đoán trước khi tính (Prediction)
- **Dự đoán:** Từ `physician` sẽ gần `doctor` hơn rất nhiều so với `banana` ($\cos(\text{doctor}, \text{physician}) \gg \cos(\text{doctor}, \text{banana})$).
- **Giải thích:** `doctor` và `physician` là hai từ đồng nghĩa (synonyms) cùng thuộc miền y tế, chia sẻ phân bố ngữ cảnh chặt chẽ (`patient`, `hospital`, `treatment`). Ngược lại, `banana` là danh từ chỉ loại quả nhiệt đới, hoàn toàn trực giao/xa lạ với miền y tế.

### 3. Các bước tính toán chi tiết

#### a. Chuẩn độ dài các vector:
- $\|v_{\text{doctor}}\| = \sqrt{0.8^2 + 0.1^2 + 0.7^2} = \sqrt{0.64 + 0.01 + 0.49} = \sqrt{1.14} \approx 1.0677078$
- $\|v_{\text{physician}}\| = \sqrt{0.7^2 + 0.2^2 + 0.8^2} = \sqrt{0.49 + 0.04 + 0.64} = \sqrt{1.17} \approx 1.0816654$
- $\|v_{\text{banana}}\| = \sqrt{(-0.2)^2 + 0.9^2 + (-0.1)^2} = \sqrt{0.04 + 0.81 + 0.01} = \sqrt{0.86} \approx 0.9273618$

#### b. Tính $\cos(\text{doctor}, \text{physician})$:
- Tích vô hướng:
  $$\text{dot}(v_{\text{doctor}}, v_{\text{physician}}) = (0.8 \times 0.7) + (0.1 \times 0.2) + (0.7 \times 0.8) = 0.56 + 0.02 + 0.56 = 1.14$$
- Cosine:
  $$\cos(v_{\text{doctor}}, v_{\text{physician}}) = \frac{1.14}{\sqrt{1.14} \cdot \sqrt{1.17}} = \sqrt{\frac{1.14}{1.17}} \approx \sqrt{0.97435897} \approx 0.987096 \approx 0.9871$$

#### c. Tính $\cos(\text{doctor}, \text{banana})$:
- Tích vô hướng:
  $$\text{dot}(v_{\text{doctor}}, v_{\text{banana}}) = (0.8 \times -0.2) + (0.1 \times 0.9) + (0.7 \times -0.1) = -0.16 + 0.09 - 0.07 = -0.14$$
- Cosine:
  $$\cos(v_{\text{doctor}}, v_{\text{banana}}) = \frac{-0.14}{1.0677078 \times 0.9273618} = \frac{-0.14}{0.9901515} \approx -0.141392 \approx -0.1414$$

### 4. Diễn giải kết quả
- $\cos(\text{doctor}, \text{physician}) \approx 0.9871$: Giá trị rất sát $1.0$, góc giữa hai vector xấp xỉ $\arccos(0.9871) \approx 9.2^\circ$, biểu thị sự tương đồng ngữ nghĩa cực kỳ mạnh mẽ giữa hai từ đồng nghĩa.
- $\cos(\text{doctor}, \text{banana}) \approx -0.1414$: Giá trị âm và gần $0$, góc kẹp lớn hơn $90^\circ$ ($\arccos(-0.1414) \approx 98.1^\circ$), khẳng định hai khái niệm không có liên kết ngữ nghĩa trong không gian vector.
- Kết quả số học khớp hoàn hảo với giả thuyết và dự đoán ban đầu.

---

## Bài 4 — Sparse versus Dense Representation

### 1. Nhận diện dạng biểu diễn
1. **Biểu diễn Sparse (Thưa):** Vector word-context có $10,000$ chiều nhưng chỉ có $30$ giá trị khác không ($99.7\%$ thành phần bằng $0$).
2. **Biểu diễn Dense (Dày đặc):** Word embedding có $300$ chiều và hầu hết các thành phần đều là các số thực liên tục khác $0$.

### 2. Vì sao Dense Representation thuận lợi hơn cho Semantic Similarity?
- **Không gian nén đặc trưng ẩn (Low-dimensional Latent Space):** Dense embedding nén thông tin từ $|V|$ chiều rời rạc xuống $d \ll |V|$ chiều liên tục. Mô hình buộc phải tổng quát hóa các mối quan hệ ngữ cảnh phân tán vào các đặc trưng trừu tượng chung (ví dụ: chiều giới tính, chiều hoàng tộc, chiều chủ đề y tế).
- **Nắm bắt đồng xuất hiện bậc hai (Second-order Co-occurrence):** Trong biểu diễn sparse, hai từ đồng nghĩa (`doctor` và `physician`) nếu không cùng xuất hiện với đúng các từ ngữ cảnh cụ thể trong corpus thì tích vô hướng bằng $0$ (trực giao giả tạo). Dense embedding học thông qua mạng nơ-ron giúp kéo các từ có ngữ cảnh tương đương về gần nhau trong không gian vector.
- **Hiệu quả tính toán và lưu trữ:** Giảm triệt để dung lượng bộ nhớ (vector 300 số thực thay vì ma trận khổng lồ $|V| \times |V|$) và đẩy nhanh tốc độ tính toán similarity trên GPU/CPU.

### 3. Dense Representation có chắc chắn "tốt hơn" trong mọi bài toán không?
**Trả lời:** **KHÔNG.** Không nên biến embedding thành một khẩu hiệu tuyệt đối hóa "dense luôn tốt hơn".

**Phản ví dụ & Trade-offs cụ thể:**
- **Tính diễn giải (Interpretability):** Sparse representation có tính minh bạch tuyệt đối — mỗi chiều vector ứng với một từ vựng xác định, con người dễ dàng giải thích tại sao hai văn bản giống nhau. Ngược lại, dense embedding là biểu diễn dạng "hộp đen" (black-box latent features), rất khó giải thích từng chiều cụ thể mang ý nghĩa gì.
- **Quy mô dữ liệu huấn luyện (Data hunger):** Dense embedding cần tập ngữ liệu đủ lớn (hàng triệu từ) để học trọng số trọng tâm. Khi ngữ liệu quá nhỏ (vài chục đến vài trăm câu), mô hình nơ-ron bị underfitting hoặc overfitting nghiêm trọng, vector học được sẽ bị nhiễu. Trong tình huống đó, phương pháp sparse đếm trực tiếp như TF-IDF/Count lại hoạt động vượt trội, ổn định và không cần thời gian huấn luyện.
- **Bài toán tìm kiếm chính xác (Exact Keyword Matching & Rare Entities):** Trong truy vấn văn bản luật pháp, mã linh kiện kỹ thuật hoặc tìm kiếm thực thể hiếm, biểu diễn sparse đảm bảo độ chính xác tuyệt đối (100% precision). Dense embedding dễ mắc lỗi "hallucination/semantic drift" khi gán độ tương đồng cao cho các từ có nghĩa gần nhau nhưng mang giá trị định danh khác nhau.

---

## Bài tập bổ sung (Mục 23 W3.pdf) — Tính toán Vector Analogy

### 1. Đề bài
Cho các vector giả định:
$$\vec{v}_{\text{king}} = [8, 2, 7]$$
$$\vec{v}_{\text{man}} = [5, 1, 5]$$
$$\vec{v}_{\text{woman}} = [5, 3, 5]$$

### 2. Thực hiện phép toán đại số vector
$$\vec{v}_{\text{target}} = \vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}}$$
$$\vec{v}_{\text{target}} = [8 - 5 + 5, \; 2 - 1 + 3, \; 7 - 5 + 5] = [8, 4, 7]$$

### 3. Giải thích loại quan hệ ngữ nghĩa
- **Chiều 1 và Chiều 3 ($[8, \cdot, 7]$):** Đại diện cho đặc tính **"Quyền lực / Hoàng gia" (Royalty / Nobility)**. Đặc tính này được bảo toàn nguyên vẹn qua phép biến đổi ($8 - 5 + 5 = 8$, $7 - 5 + 5 = 7$).
- **Chiều 2:** Đại diện cho đặc tính **"Giới tính" (Gender)**:
  - $\text{man}$ có giá trị giới tính là $1$, $\text{woman}$ có giá trị giới tính là $3$ (chênh lệch $\Delta_{\text{gender}} = +2$).
  - $\text{king}$ có giá trị $2$. Khi trừ $\vec{v}_{\text{man}}$, ta loại bỏ thành phần giống đực. Khi cộng $\vec{v}_{\text{woman}}$, ta dịch chuyển theo hướng giống cái: $2 - 1 + 3 = 4$.
- **Kết luận:** Vector kết quả $[8, 4, 7]$ mang đầy đủ đặc tính vương quyền của $\text{king}$ nhưng chuyển dịch sang giống cái, tương ứng chính xác với khái niệm $\vec{v}_{\text{queen}}$ (**Nữ hoàng**). Điều này minh chứng tính chất tuyến tính của không gian word embedding (linear substructure of embedding space).
