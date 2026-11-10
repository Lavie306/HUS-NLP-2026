# Báo Cáo Tính Toán Lý Thuyết & Bài Tập Định Lượng — LAB 08

**Học phần:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Chủ đề:** Pretrained Transformers và Tinh Chỉnh Mô Hình BERT (BERT Fine-Tuning)  
**Tác vụ thực hiện:** Calculation Exercise 1 (MLM Loss) & Calculation Exercise 2 (Tokenization)  

---

## Phần 1. Bài Tập Tính Toán 1 — Hàm Mất Mát Masked Language Modeling (MLM Loss)

### 1. Mô tả Ngữ cảnh & Thiết lập Bài toán

Trong giai đoạn tiền huấn luyện (Pre-training) của kiến trúc BERT, mô hình tiếp cận ngôn ngữ thông qua cơ chế tự giám sát (Self-Supervised Learning) với tác vụ che từ ngữ cảnh — **Masked Language Modeling (MLM)**. Thay vì dự đoán token kế tiếp theo một chiều tuần tự (như mô hình tự hồi quy GPT), BERT che ngẫu nhiên một tỷ lệ token trong câu (thông thường là $15\%$) và buộc các khối Transformer Encoder hai chiều (Bidirectional Attention) phải khôi phục lại chính xác danh tính của token bị che dựa trên toàn bộ ngữ cảnh hai phía xung quanh.

Xét câu ngữ cảnh ban đầu:
$$\mathcal{S}_{\text{gốc}} = \text{"The cat eats fish."}$$

Sau khi thực thi bước che từ ngẫu nhiên tại vị trí động từ:
$$\mathcal{S}_{\text{masked}} = \text{"The cat [MASK] fish."}$$

Mục tiêu học của mô hình là ước lượng phân phối xác suất có điều kiện trên toàn bộ không gian từ điển từ vựng $\mathcal{V}$:
$$P(w_t \mid w_{\neq t}; \theta)$$
với $w_t$ là token mục tiêu tại vị trí bị che $t$, và $w_{\neq t}$ là ngữ cảnh gồm tất cả các token còn lại trong câu.

Giả định trong ví dụ khảo sát, câu chỉ có đúng một vị trí bị che ($|M| = 1$), và sau khi vector biểu diễn ẩn tại vị trí `[MASK]` đi qua tầng chiếu tuyến tính cùng hàm Softmax, mô hình đưa ra phân phối xác suất dự đoán cho các ứng viên từ vựng như sau:

| Token ($w$) | Xác suất dự đoán của mô hình $P(w \mid x)$ | Trạng thái thực tế |
| :--- | :---: | :--- |
| `cat` | $0.10$ | Token sai |
| `dog` | $0.05$ | Token sai |
| **`eats`** | **$0.70$** | **Token nhãn chính xác (Ground Truth)** |
| `fish` | $0.15$ | Token sai |

Token mục tiêu thực tế cần khôi phục là: $y = \text{eats}$.

---

### 2. Công Thức & Tính Toán Mức Mất Mát Cơ Sở

Hàm mất mát Cross-Entropy cho tập các vị trí bị che $M$ trong tác vụ MLM được thiết lập theo nguyên lý hợp lý cực đại:
$$\mathcal{L}_{\text{MLM}} = - \sum_{i \in M} \log P(y_i \mid x)$$

Trong trường hợp đơn lẻ với duy nhất một token bị mask ($M = \{t\}$), công thức thu gọn thành:
$$\mathcal{L} = - \log P(y_t = \text{"eats"} \mid x) = - \log(0.70)$$

#### Các kết quả tính toán chi tiết theo các hệ cơ số:

1. **Theo Logarit tự nhiên ($\ln$, cơ số $e$) — Tiêu chuẩn thực thi trong Deep Learning & PyTorch (`torch.nn.CrossEntropyLoss`):**
   $$\mathcal{L} = - \ln(0.70) \approx -(-0.3566749) \approx \mathbf{0.3567}$$

2. **Theo Logarit cơ số 2 ($\log_2$) — Đo lường theo Lý thuyết Thông tin (đơn vị: shannons / bits):**
   $$\mathcal{L} = - \log_2(0.70) = - \frac{\ln(0.70)}{\ln(2)} \approx \frac{0.3566749}{0.693147} \approx \mathbf{0.5146} \text{ bits}$$

3. **Theo Logarit thập phân ($\log_{10}$):**
   $$\mathcal{L} = - \log_{10}(0.70) \approx \mathbf{0.1549}$$

---

### 3. Phân Tích Độ Biến Thiên & Bản Chất Toán Học

#### Câu hỏi 1: Nếu mô hình dự đoán xác suất token đúng đạt $P(\text{eats}) = 0.95$, giá trị loss thay đổi như thế nào?

* **Tính toán mức mất mát mới:**
  $$\mathcal{L}_{\text{mới}} = - \ln(0.95) \approx -(-0.051293) \approx \mathbf{0.0513}$$
  *(Quy đổi sang hệ nhị phân: $\mathcal{L}_{\text{mới}} = - \log_2(0.95) \approx \mathbf{0.0740} \text{ bits}$)*

* **Độ chênh lệch và tỷ lệ biến thiên:**
  $$\Delta \mathcal{L} = \mathcal{L}_{\text{mới}} - \mathcal{L}_{\text{cơ sở}} \approx 0.0513 - 0.3567 = \mathbf{-0.3054}$$
  Giá trị loss **giảm xấp xỉ $85.62\%$** so với mức ban đầu.

* **Phân tích bản chất học máy:**
  - Khi xác suất dự đoán cho từ khóa đúng tăng từ $0.70$ lên $0.95$, mô hình đạt được mức độ chắc chắn rất cao đối với nhãn mục tiêu.
  - Xét hàm số mất mát $f(p) = -\ln(p)$, đạo hàm bậc một là $f'(p) = -\frac{1}{p}$. Khi $p \to 1$, giá trị hàm số tiệm cận về $0$ và độ dốc thoải dần.
  - Điều này tạo ra gradient cập nhật rất nhỏ trong quá trình lan truyền ngược, báo hiệu mạng nơ-ron rằng các trọng số chú ý đã biểu diễn ngữ cảnh rất chuẩn xác cho mẫu này, ngăn ngừa sự dao động tham số không cần thiết.

---

#### Câu hỏi 2: Nếu mô hình chỉ dự đoán xác suất token đúng ở mức $P(\text{eats}) = 0.01$, giá trị loss thay đổi như thế nào?

* **Tính toán mức mất mát mới:**
  $$\mathcal{L}_{\text{mới}} = - \ln(0.01) \approx -(-4.605170) \approx \mathbf{4.6052}$$
  *(Quy đổi sang hệ nhị phân: $\mathcal{L}_{\text{mới}} = - \log_2(0.01) \approx \mathbf{6.6439} \text{ bits}$)*

* **Độ chênh lệch và tỷ lệ biến thiên:**
  $$\Delta \mathcal{L} = \mathcal{L}_{\text{mới}} - \mathcal{L}_{\text{cơ sở}} \approx 4.6052 - 0.3567 = \mathbf{+4.2485}$$
  Giá trị loss **tăng vọt gấp $12.91$ lần** so với mức ban đầu.

* **Phân tích bản chất học máy:**
  - Mô hình chỉ phân bổ $1\%$ niềm tin cho token đúng `eats`, đồng nghĩa với việc gán tới $99\%$ xác suất vào các phương án sai lầm khác. Đây là trường hợp mô hình phạm sai số nghiêm trọng nhưng lại "tự tin vào dự đoán sai".
  - Do tính chất giới hạn tiệm cận đứng của hàm số:
    $$\lim_{p \to 0^+} (-\ln p) = +\infty$$
  - Cross-Entropy áp dụng một cơ chế phạt phi tuyến lũy tiến cực kỳ khắt khe: khi xác suất đúng tiệm cận sát về $0$, hình phạt tiến dần ra vô hạn.
  - Giá trị loss khổng lồ này sẽ sinh ra gradient có độ lớn vượt trội khi lan truyền ngược, cưỡng bức các ma trận tham số (Query, Key, Value, Feed-Forward) phải điều chỉnh mạnh mẽ nhằm sửa chữa sai lệch nhận thức nghiêm trọng của mô hình.

---

#### Bảng đối chiếu phản ứng của hàm Loss theo xác suất dự đoán

| Kịch bản đánh giá | $P(\text{eats})$ | Loss ($\ln$) | Biến thiên $\Delta \mathcal{L}$ | Tỷ lệ so với gốc | Đánh giá trạng thái mô hình |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Sai lệch nghiêm trọng** | $0.01$ | $\mathbf{4.6052}$ | $+4.2485$ | Tăng $12.91$ lần | Bị trừng phạt cực nặng; gradient lớn ép cập nhật triệt để |
| **Mức cơ sở (đề bài)** | $0.70$ | $\mathbf{0.3567}$ | $0.0000$ | $100\%$ | Nhận diện đúng hướng, tiếp tục tối ưu hóa độ dốc |
| **Hội tụ xuất sắc** | $0.95$ | $\mathbf{0.0513}$ | $-0.3054$ | Giảm $85.62\%$ | Tự tin chính xác cao; loss tiệm cận 0, hội tụ ổn định |

---

#### Câu hỏi 3: Vì sao hàm Cross-Entropy lại là sự lựa chọn tối ưu cho bài toán Masked Language Modeling?

Hàm mất mát Cross-Entropy đóng vai trò là hòn đá tảng cho bài toán MLM nhờ bốn lý do khoa học và toán học then chốt:

1. **Bản chất của MLM là bài toán phân loại đa lớp trên không gian từ điển cực lớn ($|\mathcal{V}|$ classes):**
   Tại mỗi vị trí `[MASK]`, nhiệm vụ của mô hình không phải là hồi quy một giá trị thực liên tục, mà là chọn lựa một token định danh chuẩn xác trong tập hợp hữu hạn gồm hàng chục nghìn ứng viên rời rạc (từ điển WordPiece của BERT chứa $|\mathcal{V}| = 30,522$ tokens). Cross-Entropy là thước đo khoảng cách phân phối xác suất trực tiếp và tự nhiên nhất giữa phân phối một điểm (nhãn thực tế biểu diễn dạng vector one-hot) và phân phối dự đoán đa thức qua tầng Softmax.

2. **Tính tương đương với Nguyên lý Hợp lý Cực đại (Maximum Likelihood Estimation - MLE):**
   Tối thiểu hóa tổng hàm mất mát Cross-Entropy trên toàn bộ kho ngữ liệu huấn luyện:
   $$\min_{\theta} \left( -\sum_{t \in M} \log P(w_t \mid w_{\neq t}; \theta) \right) \iff \max_{\theta} \prod_{t \in M} P(w_t \mid w_{\neq t}; \theta)$$
   chính là việc tối đa hóa hàm hợp lý của văn bản tự nhiên theo ngữ cảnh hai chiều. Về mặt lý thuyết thông tin, quá trình này tương đương với việc cực tiểu hóa Phân kỳ Kullback-Leibler ($\mathcal{D}_{\text{KL}}$) giữa phân phối ngôn ngữ thực của loài người và phân phối mà mô hình tái hiện.

3. **Cơ chế Phạt Bất Đối Xứng dựa trên Khái niệm "Độ Bất Ngờ" (Surprisal):**
   Trong lý thuyết thông tin, lượng tin tự thân $I(x) = -\log P(x)$ phản ánh mức độ bất ngờ của một biến cố. Nếu mô hình đã dự đoán token đúng với xác suất tiệm cận $1$, lượng tin bất ngờ bằng $0 \to$ mô hình không chịu hình phạt. Ngược lại, nếu một từ ngữ hiển nhiên phải xuất hiện trong ngữ cảnh nhưng mô hình lại đánh giá xác suất gần $0$, mức độ bất ngờ tiến về vô cùng $\to$ mô hình phải chịu hình phạt vô cùng lớn. Cơ chế bất đối xứng này tạo động lực định hướng mạnh mẽ giúp mô hình phân bổ xác suất chính xác.

4. **Đặc tính Gradient tối ưu khi kết hợp với Softmax (Triệt tiêu hiện tượng nghẽn đạo hàm):**
   Nếu sử dụng hàm lỗi bình phương trung bình (MSE) cho bài toán xác suất Softmax, đạo hàm thường chứa thành phần tích $P_i(1 - P_i)$, dẫn đến hiện tượng triệt tiêu gradient (vanishing gradient) khi mô hình đưa ra dự đoán sai lệch lớn. Trái lại, khi kết hợp hàm kích hoạt Softmax với hàm mất mát Cross-Entropy, đạo hàm riêng đối với logit đầu vào $z_i$ trước Softmax mang dạng cực kỳ tinh gọn và tuyến tính:
   $$\frac{\partial \mathcal{L}}{\partial z_i} = P(w_i \mid x) - y_i$$
   Độ dốc gradient tỷ lệ trực tiếp với sai số xác suất $(P_i - y_i)$, giúp quá trình cập nhật trọng số bằng các thuật toán tối ưu (như AdamW) diễn ra mượt mà, ổn định và tránh bão hòa trên quy mô hàng tỷ tham số.

---

## Phần 2. Bài Tập Tính Toán 2 — Cơ Chế Tách Từ & Mặt Nạ Chú Ý (Tokenization & Attention Mask)

### 1. Yêu cầu Bài toán

Cho câu văn bản đầu vào:
$$\mathcal{T} = \text{"The cats are playing."}$$

Thực hiện các nội dung phân tích:
1. Tokenize chuỗi văn bản bằng bộ tách từ `bert-base-uncased`.
2. Xác định chuỗi token thu được (`token sequence`).
3. Xác định danh sách định danh token (`token IDs`).
4. Nhận diện và giải thích các token đặc biệt (`special tokens`).
5. Xác định mặt nạ chú ý tương ứng (`attention mask`).
6. So sánh định lượng và bản chất giữa ba cấp độ: `words`, `tokens`, và `subword tokens`.

---

### 2. Thực Thi Mã Nguồn Phân Tích

Sử dụng thư viện `transformers` chuẩn của HuggingFace:

```python
from transformers import AutoTokenizer

# Tải tokenizer WordPiece tương ứng với BERT-base-uncased
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "The cats are playing."

# 1. Tách từ cấp độ subword thô (chưa gán token đặc biệt)
raw_tokens = tokenizer.tokenize(text)

# 2. Mã hóa toàn diện (bổ sung [CLS], [SEP] và ánh xạ ID)
encoded = tokenizer(text)
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]
tokens_with_special = tokenizer.convert_ids_to_tokens(input_ids)

# 3. Minh họa khi đệm chuỗi (Padding tới max_length = 10)
encoded_pad = tokenizer(text, padding="max_length", max_length=10)
padded_ids = encoded_pad["input_ids"]
padded_mask = encoded_pad["attention_mask"]
padded_tokens = tokenizer.convert_ids_to_tokens(padded_ids)
```

---

### 3. Kết Quả Khảo Sát & Giải Thích Chi Tiết

#### 1 & 2. Chuỗi Token (`token sequence`)

* **Chuỗi token ngữ nghĩa thuần túy (Raw tokens sau khi chuẩn hóa):**
  $$\text{['the', 'cats', 'are', 'playing', '.']}$$
  *Nhận xét:* Vì cấu hình là `uncased`, chữ viết hoa `"The"` được chuẩn hóa tự động về chữ thường `"the"`. Dấu chấm câu `"."` được tách biệt thành một token độc lập.

* **Chuỗi token cấu trúc đầu vào nạp vào mô hình BERT (Full Sequence):**
  $$\text{['[CLS]', 'the', 'cats', 'are', 'playing', '.', '[SEP]']}$$

* *(Mở rộng)* **Chuỗi token khi thực hiện đệm (Padding cố định với `max_length = 10`):**
  $$\text{['[CLS]', 'the', 'cats', 'are', 'playing', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]']}$$

---

#### 3. Danh Sách Token IDs (`input_ids`)

Bảng tra cứu chỉ số định danh tương ứng trong không gian từ điển WordPiece ($|\mathcal{V}| = 30,522$):

| STT | Vị trí (Index) | Token | Token ID | Phân loại | Vai trò chức năng |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 0 | `[CLS]` | **101** | Special Token | Đánh dấu khởi đầu chuỗi, thu thập biểu diễn toàn câu |
| 2 | 1 | `the` | **1996** | Subword / Word | Mạo từ xác định (chuyển từ viết hoa sang viết thường) |
| 3 | 2 | `cats` | **8870** | Subword / Word | Danh từ số nhiều (tồn tại nguyên khối trong vocab) |
| 4 | 3 | `are` | **2024** | Subword / Word | Trợ động từ số nhiều thì hiện tại |
| 5 | 4 | `playing` | **2652** | Subword / Word | Động từ phân từ hiện tại (dạng -ing có sẵn trong vocab) |
| 6 | 5 | `.` | **1012** | Punctuation | Dấu ngắt câu |
| 7 | 6 | `[SEP]` | **102** | Special Token | Đánh dấu kết thúc chuỗi văn bản |

* **Vector `input_ids` tiêu chuẩn (không đệm):**
  $$[101, 1996, 8870, 2024, 2652, 1012, 102]$$

* *(Vector `input_ids` khi đệm tới `max_length = 10`):*
  $$[101, 1996, 8870, 2024, 2652, 1012, 102, 0, 0, 0]$$

---

#### 4. Nhận Diện & Vai Trò Của Các Token Đặc Biệt (`special tokens`)

Trong chuỗi trên, các token đặc biệt xuất hiện bao gồm:

1. **`[CLS]` (Classification Token — ID `101`):**
   - **Vị trí:** Luôn cố định tại chỉ số đầu tiên (index $0$) của chuỗi nạp vào mô hình.
   - **Vai trò:** Bản thân `[CLS]` không mang ngữ nghĩa từ điển ban đầu. Tuy nhiên, thông qua tất cả các tầng Self-Attention hai chiều của BERT, vị trí này được phép tương tác và chú ý tới toàn bộ các token khác trong chuỗi. Do đó, vector trạng thái ẩn ở tầng cuối cùng $h_{[\text{CLS}]} \in \mathbb{R}^{768}$ sẽ kết tinh toàn bộ thông tin ngữ cảnh của văn bản, trở thành đại diện tiêu chuẩn để đưa qua tầng phân loại (Linear Classifier) cho các bài toán phân loại câu hoặc phân tích cảm xúc.

2. **`[SEP]` (Separator Token — ID `102`):**
   - **Vị trí:** Đặt tại cuối mỗi câu độc lập (index $6$ trong ví dụ).
   - **Vai trò:** Đóng vai trò như ranh giới cấu trúc để phân tách các câu. Trong các tác vụ xử lý cặp câu (như Natural Language Inference - NLI, hoặc Question Answering), `[SEP]` dùng để ngăn cách giữa Câu A và Câu B (`[CLS] Đoạn văn [SEP] Câu hỏi [SEP]`).

3. **`[PAD]` (Padding Token — ID `0`):**
   - **Vị trí:** Đệm vào phần đuôi của chuỗi khi độ dài thực tế ngắn hơn `max_length`.
   - **Vai trò:** Giúp đồng bộ hóa kích thước các mẫu dữ liệu trong cùng một batch thành một tensor chữ nhật cố định (ví dụ: `[batch_size, max_length]`), tạo điều kiện tính toán song song tối ưu trên phần cứng GPU/TPU.

---

#### 5. Xác Định Mặt Nạ Chú Ý (`attention mask`)

Mặt nạ chú ý `attention_mask` là một vector nhị phân cùng chiều dài với `input_ids`, có chức năng chỉ thị cho ma trận Multi-Head Attention biết token nào là thông tin thực và vị trí nào là khoảng đệm kỹ thuật.

* **Trường hợp chuỗi nguyên bản (độ dài thực $L = 7$):**
  $$\text{attention\_mask} = [1, 1, 1, 1, 1, 1, 1]$$
  *Ý nghĩa:* Toàn bộ 7 token đều là thông tin hợp lệ, tham gia đầy đủ vào việc tính toán trọng số chú ý hai chiều.

* **Trường hợp chuỗi có đệm (Padding tới `max_length = 10`):**
  $$\text{attention\_mask} = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]$$
  *Ý nghĩa & Cơ chế toán học:*
  - Các vị trí mang giá trị **$1$** (index $0 \to 6$) được tính điểm chú ý bình thường.
  - Các vị trí mang giá trị **$0$** (index $7 \to 9$) tương ứng với các token `[PAD]`. Trước khi đưa qua hàm Softmax tính phân phối chú ý:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M_{\text{attn}}\right)V$$
    ma trận mặt nạ $M_{\text{attn}}$ sẽ cộng giá trị $-\infty$ (hoặc $-10,000.0$) vào các vị trí có mask bằng $0$.
  - Khi áp dụng hàm mũ Softmax:
    $$\lim_{x \to -\infty} e^x = 0$$
    do đó trọng số chú ý hướng về các token `[PAD]` bị triệt tiêu hoàn toàn về $0$. Cơ chế này ngăn chặn việc thông tin rác của token đệm làm sai lệch biểu diễn ngữ nghĩa của các từ thực sự trong câu.

---

#### 6. So Sánh Định Lượng: Words vs. Tokens vs. Subword Tokens

| Tiêu chí khảo sát | Số lượng | Danh sách cụ thể | Bản chất ngôn ngữ học & kiến trúc |
| :--- | :---: | :--- | :--- |
| **Words (Từ ngữ nghĩa)** | **4 từ** | `"The"`, `"cats"`, `"are"`, `"playing"` *(và 1 dấu câu `.`)* | Đơn vị từ tự nhiên theo ngữ pháp ngôn ngữ học truyền thống, phân cách nhau bởi khoảng trắng. |
| **Tokens (Đơn vị nạp vào BERT)** | **7 tokens** | `[CLS]`, `the`, `cats`, `are`, `playing`, `.`, `[SEP]` | Toàn bộ các định danh rời rạc cấu thành đầu vào của mạng nơ-ron, bao gồm cả các ký hiệu điều khiển cấu trúc (`special tokens`). |
| **Subword Tokens (Mảnh từ vựng WordPiece)** | **5 mảnh** | `the`, `cats`, `are`, `playing`, `.` | Các đơn vị từ vựng sinh ra từ thuật toán nén WordPiece dựa trên tần suất xuất hiện trong ngữ liệu tiền huấn luyện. |

#### Phân Tích Chuyên Sâu về Thuật Toán WordPiece:
- Trong ví dụ trên, toàn bộ 4 từ (`the`, `cats`, `are`, `playing`) cùng dấu chấm `.` đều là những đơn vị từ vựng cực kỳ phổ biến trong kho ngữ liệu tiếng Anh, do đó thuật toán WordPiece đã lưu trữ chúng dưới dạng các từ nguyên khối trong từ điển $30,522$ mục từ, không cần tách nhỏ thêm.
- **Hiện tượng phân mảnh Subword khi gặp từ hiếm (Out-of-Vocabulary - OOV):**
  Giả sử đầu vào xuất hiện những từ hiếm hoặc biến thể chính tả chưa từng có trong từ điển, ví dụ: `"The catss are playingg."`:
  - Tokenizer sẽ tự động phân rã:
    - `"catss"` $\longrightarrow$ `['cats', '##s']`
    - `"playingg"` $\longrightarrow$ `['playing', '##g']`
  - Tiền tố `##` đánh dấu rằng đây là một mảnh từ con (subword) ghép nối tiếp vào gốc từ đứng trước chứ không phải là một từ bắt đầu mới.
  - Khi đó, số lượng subword tokens sẽ tăng từ $5$ lên $7$ mảnh (chưa tính special tokens).
  - **Lợi ích cốt lõi:** Cơ chế này cho phép mô hình BERT đạt được khả năng bao phủ từ điển gần như vô hạn với một không gian từ vựng có kích thước vừa phải ($|\mathcal{V}| \approx 30,000$), triệt tiêu hoàn toàn vấn đề token lạ `[UNK]` (Unknown) mà vẫn bảo toàn được gốc từ vựng và cấu trúc ngữ pháp hình thái học (morphology).
