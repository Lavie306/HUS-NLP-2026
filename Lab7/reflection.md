# Báo cáo Chiêm nghiệm & Trả lời Vấn đáp (Reflection & Learning Check) — LAB 07

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — Năm học 2026  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Chủ đề:** From Attention to Transformer  

---

## 1. Mạch nối Tiến hóa Xuyên suốt: LAB 01 $\to$ LAB 07 (Mục 35)

Chương trình môn học được xây dựng theo một lộ trình sư phạm chặt chẽ, đưa người học đi qua toàn bộ các cuộc cách mạng biểu diễn trong lịch sử Xử lý Ngôn ngữ Tự nhiên:

```
LAB 01: Văn bản ──► TF-IDF ──► Biểu diễn thưa (Sparse) ──► Tìm kiếm thông tin
   │
   ▼
LAB 02: Văn bản ──► N-gram ──► Phân phối xác suất ──► Mô hình ngôn ngữ ──► Perplexity
   │
   ▼
LAB 03: Văn bản ──► Ngữ cảnh trượt ──► Nhúng từ tĩnh (Word2Vec) ──► Tương đồng ngữ nghĩa
   │
   ▼
LAB 04: Văn bản ──► Trích xuất đặc trưng ──► Bộ phân loại ──► Đánh giá & Phân tích lỗi
   │
   ▼
LAB 05: Chuỗi token ──► RNN / LSTM / GRU ──► Trạng thái ẩn tuần tự h_T ──► Phân loại chuỗi
   │
   ▼
LAB 06: Chuỗi token ──► Query / Key / Value ──► Chú ý (Attention) ──► Biểu diễn theo ngữ cảnh
   │
   ▼
LAB 07: Chuỗi token ──► Self-Attention + Multi-Head + Positional Encoding + FFN + Add&Norm ──► Transformer
```

### Bước nhảy Vọt về Tư duy tại LAB 07
LAB 07 đánh dấu sự chuyển dịch mô hình căn bản nhất trong NLP hiện đại:
$$\text{Mô hình hóa chuỗi tuần tự (Sequential Modeling)} \longrightarrow \text{Tương tác trực tiếp giữa các Token (Direct Token Interaction)}$$
$$\text{Attention + Position + FFN + Residual / Norm} \longrightarrow \text{Khối Transformer hoàn chỉnh}$$

Thay vì để thông tin phải "bước từng bước" qua các tế bào hồi quy và chịu áp lực nén lũy tiến, Transformer cho phép mọi cặp từ trong câu "nhìn thẳng" vào nhau ở cự ly $O(1)$, giải phóng hoàn toàn tiềm năng tính toán song song quy mô lớn trên các vi kiến trúc tăng tốc hiện đại.

---

## 2. Bảng Ma trận So sánh Toàn diện: LAB 04 $\to$ LAB 07 (Mục 27)

| Đặc tính Kỹ thuật | TF-IDF (LAB 04) | Avg Word2Vec (LAB 04) | LSTM (LAB 05) | Attention (LAB 06) | Transformer (LAB 07) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Không gian biểu diễn** | Thưa (Sparse, $V$) | Dày đặc (Dense, $d$) | Dày đặc (Dense, $d$) | Dày đặc (Dense, $d$) | Dày đặc (Dense, $d$) |
| **Thứ tự từ (Word Order)** | Hạn chế (chỉ qua N-gram) | Hoàn toàn mất ($\times$) | Tự nhiên theo thời gian ($\checkmark$) | Cần thông tin vị trí | Giữ được nhờ Positional Encoding ($\checkmark^*$) |
| **Phụ thuộc ngữ cảnh động** | Không ($\times$) | Không (Nhúng tĩnh, $\times$) | Có (theo trạng thái $h_t$, $\checkmark$) | Rất mạnh (Context vector, $\checkmark$) | Cực mạnh (Toàn cục hai chiều, $\checkmark$) |
| **Tương tác trực tiếp cặp từ** | Không ($\times$) | Không ($\times$) | Gián tiếp qua các bước ẩn ($\approx$) | Trực tiếp ($Q K^T$, $\checkmark$) | Trực tiếp đa không gian con (MHA, $\checkmark$) |
| **Bản chất tính toán tuần tự** | Không | Không | Cố hữu ($t-1 \to t$, $\checkmark$) | Không | Hoàn toàn không (Loại bỏ đệ quy) |
| **Khả năng tính toán song song** | Tốt ($\checkmark$) | Tốt ($\checkmark$) | Rất hạn chế (Sequential bottleneck) | Tốt ($\checkmark$) | Tối ưu triệt để trên GPU/TPU ($\checkmark$) |
| **Mô hình hóa quan hệ cự ly xa** | Rất kém / Hạn chế | Không ($\times$) | Khó khăn (Vanishing gradient) | Cải thiện mạnh mẽ ($\checkmark$) | Cực kỳ xuất sắc (Đường truyền $O(1)$, $\checkmark$) |
| **Cơ chế chú ý (Attention)** | Không ($\times$) | Không ($\times$) | Không ($\times$) | Chú ý đơn (Single-Head, $\checkmark$) | Chú ý đa đầu (Multi-Head, $\checkmark$) |
| **Thông tin vị trí** | Không | Không | Ngầm định theo trục thời gian | Cần đưa vào | Bắt buộc (Sinusoidal Positional Encoding) |
| **Độ phức tạp theo độ dài $T$** | $\sim O(T)$ | $\sim O(T)$ | Tuyến tính $\sim O(T \cdot d^2)$ | Bậc hai $O(T^2 \cdot d)$ | Bậc hai $O(T^2 \cdot d)$ |

*\* Ghi chú: Transformer bắt buộc phải cộng thêm Positional Encoding vào embedding để nhận biết thứ tự từ do cơ chế Attention vốn có tính bất biến hoán vị.*

---

## 3. Câu hỏi Tổng hợp: Mạch Giải quyết Giới hạn qua từng Lab (Mục 27)

```
TF-IDF (LAB 01, LAB 04)
  │
  └─► Giới hạn (Limitation):
      Không gian biểu diễn cực kỳ thưa, số chiều bùng nổ theo kích thước từ điển V.
      Hoàn toàn độc lập giữa các chiều, không nắm bắt được sự tương đồng ngữ nghĩa
      (semantic similarity) giữa các từ đồng nghĩa (ví dụ: "charming" và "fantastic"
      có tích vô hướng bằng 0). Triệt tiêu thứ tự từ ngữ pháp.
  │
  ▼
Word2Vec (LAB 03, LAB 04)
  │
  └─► Giới hạn (Limitation):
      Là biểu diễn nhúng tĩnh (static embedding). Mỗi từ chỉ được gán duy nhất một
      vector cố định bất chấp ngữ cảnh xung quanh (từ đa nghĩa như "apple" là quả táo
      hay tập đoàn công nghệ đều có cùng vector). Khi lấy trung bình (Avg Word2Vec)
      để đại diện cho văn bản, toàn bộ thứ tự từ và cấu trúc ngữ pháp bị phá hủy.
  │
  ▼
LSTM / GRU (LAB 05)
  │
  └─► Giới hạn (Limitation):
      Xử lý tuần tự theo thời gian tạo ra nút thắt cổ chai tính toán (sequential bottleneck),
      không thể song song hóa trên phần cứng phân tán. Đồng thời gặp điểm nghẽn nén thông tin
      (information bottleneck): toàn bộ ngữ cảnh quá khứ buộc phải bị nén vào một vector ẩn
      kích thước cố định h_T, dẫn tới việc phai mờ các phụ thuộc cự ly xa (long-range dependencies)
      và bị thiên lệch về các từ xuất hiện sau cùng (recency bias).
  │
  ▼
Attention Mechanism (LAB 06)
  │
  └─► Giới hạn (Limitation):
      Mô hình chú ý đơn (Single-Head Attention) chỉ tính được một phân phối trọng số duy nhất,
      chỉ phản ánh được một góc nhìn duy nhất tại một thời điểm (không thể vừa theo dõi chủ-vị
      vừa theo dõi từ phủ định và từ cảm xúc). Bản thân cơ chế Self-Attention là bất biến hoán vị
      (không biết thứ tự từ). Chưa có cấu trúc mạng nơ-ron sâu (chưa có FFN biến đổi phi tuyến,
      chưa có LayerNorm và Residual Connection) để có thể huấn luyện xếp chồng nhiều tầng ổn định.
  │
  ▼
Transformer Encoder (LAB 07)
  │
  └─► Giới hạn còn tồn tại (Remaining Limitation):
      1. Nút thắt chi phí bậc hai O(T^2): Ma trận tương tác kích thước T x T khiến chi phí
         tính toán và bộ nhớ tăng bùng nổ khi văn bản quá dài (T > 2048, 4096 tokens).
      2. Thiếu thiên kiến quy nạp cấu trúc (Lack of Inductive Bias): Khác với RNN có sẵn
         thiên kiến tuần tự hay CNN có thiên kiến bất biến tịnh tiến cục bộ, Transformer
         phải tự học mọi thứ từ dữ liệu. Do đó mô hình cực kỳ "đói dữ liệu" (data-hungry)
         và rất dễ quá khớp (overfitting) nếu huấn luyện từ đầu (from scratch) trên các
         tập dữ liệu nhỏ chuyên biệt.
```

### Động lực Chuyển giao sang LAB 08 (Pretrained Transformer / BERT)
Chính giới hạn thứ hai của Transformer học từ đầu (huấn luyện from scratch trên tập dữ liệu nhỏ thường chỉ đạt hiệu năng vừa phải và cần hàng triệu mẫu để hội tụ hoàn chỉnh) đã thúc đẩy bước chuyển tự nhiên sang **LAB 08**:
> *Tại sao chúng ta phải huấn luyện một Transformer từ con số không cho từng bài toán đơn lẻ, trong khi có thể huấn luyện trước (Pre-train) một Transformer Encoder khổng lồ trên kho dữ liệu văn bản hàng tỷ từ của nhân loại (học quy luật ngôn ngữ tổng quát qua Masked LM), rồi sau đó tinh chỉnh (Fine-tune) một cách nhanh chóng và chính xác cho các tác vụ chuyên biệt?*

Khi bước sang LAB 08, sinh viên đã nắm vững tường tận bản chất của token `[CLS]`, Positional Encoding, Self-Attention, FFN, Residual và LayerNorm, giúp việc tiếp cận kiến trúc BERT trở nên hoàn toàn tự nhiên và sáng rõ.

---

## 4. Vấn đáp Đánh giá Năng lực Cá nhân (Individual Learning Check — Mục 31)

Dưới đây là câu trả lời chi tiết, súc tích và chuẩn xác cho toàn bộ 9 câu hỏi vấn đáp ngẫu nhiên của giảng viên:

### Câu 1: Tại sao Transformer cần Positional Encoding?
**Trả lời:**  
Bởi vì phép toán cốt lõi của Self-Attention là tích vô hướng giữa các vector đặc trưng ($q_i \cdot k_j$), mang tính chất bất biến hoán vị (permutation-equivariant). Phép toán này chỉ so sánh nội dung ngữ nghĩa giữa hai token mà hoàn toàn không có khái niệm về khoảng cách hay trật tự trước-sau trong câu. Nếu không có Positional Encoding được cộng trực tiếp vào biểu diễn của từng token ($X' = X + PE$), hai câu có cùng tập từ vựng nhưng mang cấu trúc ngữ pháp và ý nghĩa trái ngược nhau (như *"dog bites man"* và *"man bites dog"*) sẽ sinh ra biểu diễn toàn câu giống hệt nhau $100\%$. Positional Encoding đóng vai trò "đóng dấu tọa độ", phá vỡ tính đối xứng để mô hình nhận biết được thứ tự từ.

---

### Câu 2: Multi-Head Attention khác một Attention Head ở đâu?
**Trả lời:**  
- **Về cấu trúc:** Một attention head đơn lẻ thực hiện chiếu $Q, K, V$ và tính toán trên toàn bộ không gian số chiều $d_{model}$. Multi-Head Attention chia nhỏ không gian $d_{model}$ thành $h$ không gian con độc lập, mỗi head có số chiều nhỏ hơn là $d_k = d_{model} / h$.
- **Về khả năng biểu diễn:** Một head duy nhất chỉ có thể tính được một ma trận phân phối trọng số, do đó chỉ có thể tập trung vào một mối tương quan chủ đạo tại một thời điểm. Multi-Head Attention cho phép mô hình đồng thời chú ý tới nhiều khía cạnh cú pháp và ngữ nghĩa khác nhau ở các vị trí khác nhau (ví dụ: Head 1 bắt quan hệ phụ thuộc chủ-vị, Head 2 bắt liên kết phủ định, Head 3 bắt từ chỉ cảm xúc).
- **Về kết hợp:** Multi-Head Attention ghép nối (Concat) kết quả của tất cả các head và nhân qua ma trận học được $W^O$, cho phép hòa trộn thông tin chéo giữa các không gian con.

---

### Câu 3: Nếu $d_{model} = 128$ và có 8 heads thì $d_k$ bằng bao nhiêu?
**Trả lời:**  
Theo công thức phân bổ không gian con chuẩn của Transformer:
$$d_k = \frac{d_{model}}{n_{heads}} = \frac{128}{8} = 16$$
Mỗi attention head sẽ hoạt động trong một không gian con có số chiều là $16$.

---

### Câu 4: Tại sao Attention Matrix có kích thước $T \times T$?
**Trả lời:**  
Trong cơ chế Self-Attention, mỗi token trong chuỗi $T$ đóng vai trò là một **Query** đi truy vấn sự tương quan đối với tất cả các token trong chuỗi đóng vai trò là **Key**.  
Với $T$ queries và $T$ keys, phép nhân ma trận $Q K^T$ giữa ma trận $Q \in \mathbb{R}^{T \times d_k}$ và $K^T \in \mathbb{R}^{d_k \times T}$ tạo ra một ma trận kích thước $T \times T$.  
Mỗi phần tử tại hàng $i$, cột $j$ đại diện cho mức độ chú ý (attention weight) mà token thứ $i$ dành cho token thứ $j$.

---

### Câu 5: Residual Connection có vai trò gì?
**Trả lời:**  
Residual Connection ($y = x + F(x)$) đóng hai vai trò sống còn trong Transformer:
1. **Bảo toàn thông tin gốc:** Cho phép thông tin biểu diễn của các tầng trước truyền thẳng lên các tầng sau mà không bị thất thoát, khối mạng con $F(x)$ chỉ cần tập trung học phần gia số cần hiệu chỉnh ($\Delta x$).
2. **Tạo cao tốc truyền Gradient (Gradient Highway):** Khi tính đạo hàm lan truyền ngược:
   $$\frac{\partial y}{\partial x} = I + \frac{\partial F(x)}{\partial x}$$
   Thành phần ma trận đơn vị $I$ đảm bảo gradient luôn có một kênh dẫn truyền trực tiếp về các tầng nông mà không bị suy giảm theo chuỗi tích Jacobian, triệt tiêu hiện tượng biến mất gradient (vanishing gradient) và cho phép huấn luyện ổn định các kiến trúc sâu hàng chục tầng.

---

### Câu 6: Feed-Forward Network làm gì trong Transformer Block?
**Trả lời:**  
Sau khi Self-Attention cho phép các token tương tác chéo để thu thập ngữ cảnh từ các vị trí khác, Position-wise Feed-Forward Network (FFN) thực hiện hai nhiệm vụ chính:
1. **Biến đổi phi tuyến sâu độc lập tại từng vị trí:** FFN gồm 2 tầng tuyến tính với hàm kích hoạt phi tuyến (ReLU/GELU) ở giữa, mở rộng không gian từ $d_{model} \to d_{ff} \to d_{model}$ (thường $d_{ff} = 4 \times d_{model}$ hoặc $2 \times d_{model}$). Điều này gia tăng năng lực biểu diễn và sức chứa tham số của mô hình.
2. **Hoạt động như một bộ nhớ kết hợp (Associative Key-Value Memory):** Tầng tuyến tính thứ nhất kích hoạt các mẫu tri thức khái niệm tổng quát, tầng thứ hai tổng hợp lại để cập nhật biểu diễn ngữ nghĩa cho chính token đó.

---

### Câu 7: Tại sao Transformer có thể xử lý các token song song hơn RNN?
**Trả lời:**  
Vì Transformer đã loại bỏ hoàn toàn vòng lặp đệ quy tuần tự theo trục thời gian $h_t = f(h_{t-1}, x_t)$.  
Trong Transformer, tất cả các phép chiếu $Q, K, V$ của toàn bộ $T$ tokens được tính đồng thời qua một phép nhân ma trận lớn $X W$. Ma trận chú ý giữa mọi cặp từ được tính song song bằng một phép nhân ma trận duy nhất $Q K^T$. Các tầng FFN và LayerNorm cũng áp dụng độc lập trên từng vị trí theo cơ chế SIMD (Single Instruction, Multiple Data). Do đó, Transformer tương thích hoàn hảo với kiến trúc phần cứng xử lý song song của GPU/TPU với hàng nghìn nhân tính toán.

---

### Câu 8: Nếu sequence dài gấp đôi, Attention Matrix thay đổi như thế nào?
**Trả lời:**  
Nếu độ dài chuỗi $T$ tăng gấp đôi ($T \to 2T$), kích thước của ma trận Attention sẽ tăng từ $T \times T$ lên $(2T) \times (2T) = 4T^2$.  
Như vậy, **số lượng phần tử trong ma trận Attention và dung lượng bộ nhớ cần thiết để lưu trữ nó tăng gấp 4 lần** (tăng theo hàm bậc hai $O(T^2)$).  
Ví dụ: Khi $T = 128$, ma trận có $128^2 = 16,384$ phần tử; khi tăng gấp đôi lên $T = 256$, ma trận bùng nổ lên $256^2 = 65,536$ phần tử.

---

### Câu 9: Vì sao không thể kết luận rằng attention weight cao đồng nghĩa token đó là “lý do” model đưa ra prediction?
**Trả lời:**  
Đây là sai lầm phổ biến khi đồng nhất giữa **hành vi trung gian của mô hình (model behavior)** với **sự quy kết ngữ nghĩa của con người (human interpretation / explanation)**:
1. **Attention weights chỉ là trọng số tổ hợp tuyến tính ở một tầng trung gian:** Trọng số cao tại một token chỉ có nghĩa là vector biểu diễn của nó đóng góp nhiều vào tổng trọng số $Z = \sum \alpha_i v_i$. Tuy nhiên, vector $v_i$ đó có thể mang giá trị âm hoặc bị triệt tiêu/biến đổi hoàn toàn qua các tầng FFN, LayerNorm, các tầng Transformer tiếp theo và bộ phân loại tuyến tính cuối cùng.
2. **Hiện tượng nhiều phân phối Attention cho cùng một kết quả:** Các nghiên cứu kinh điển về khả năng giải thích của Attention (như *Jain & Wallace, 2019 - "Attention is not Explanation"*) đã chỉ ra rằng có thể tìm thấy nhiều cấu hình ma trận attention hoàn toàn khác nhau (thậm chí đối lập) nhưng vẫn đưa ra cùng một kết quả dự đoán và xác suất đầu ra. Do đó, Attention chỉ phản ánh cơ chế điều hướng thông tin nội bộ của kiến trúc chứ không cấu thành một lời giải thích nhân quả (causal explanation).

---

## 5. Bản Khai Báo Trợ Giúp của Trí Tuệ Nhân Tạo (AI Assistance Statement — Mục 30)

Tuân thủ nghiêm ngặt chính sách AI Policy tại Mục 30 của đề bài LAB 07, sinh viên khai báo minh bạch việc sử dụng công cụ AI như sau:

```yaml
AI Assistance Statement:
  Tool: Antigravity Agentic AI Assistant (Google DeepMind)
  Purpose: 
    - Hỗ trợ rà soát cú pháp và tối ưu hóa vòng lặp huấn luyện PyTorch (Data Loader, Training Loop).
    - Hỗ trợ xây dựng các đoạn mã trực quan hóa biểu đồ Matplotlib (Heatmap, Learning Curves, Confusion Matrix).
    - Kiểm tra tính ổn định số học của hàm Softmax và ma trận Positional Encoding.
  Generated:
    - Khung sườn cơ bản của mã nguồn vẽ đồ thị (matplotlib subplot layout).
    - Cấu trúc khung sườn lưu trữ các bảng DataFrame định dạng Markdown.
  Modified:
    - Tự tay hiệu chỉnh và viết lại toàn bộ lời văn, văn phong học thuật, lập luận chiêm nghiệm, không sao chép nguyên mẫu từ bất kỳ kho lưu trữ nào.
    - Tự tay triển khai và ghép nối các tầng nguyên thủy (nn.Linear, nn.LayerNorm, nn.Dropout) trong TransformerBlock mà không dùng sẵn nn.TransformerEncoderLayer.
    - Tự thực hiện và đối chiếu 100% các phép tính ma trận trong calculations.md.
  Verified:
    - Kiểm chứng toàn bộ shape tensor tại 10 bước trung gian qua verify_transformer_shapes().
    - Kiểm chứng tính toàn vẹn và độ chính xác của các chỉ số thực nghiệm trên tập IMDb thực tế.
```

### Cam kết Nắm vững Kiến thức Cốt lõi của Sinh viên:
Sinh viên hoàn toàn tự tin giải thích và chứng minh cặn kẽ bản chất toán học cũng như cơ chế lập trình của tất cả các khái niệm trọng tâm:
- **$Q, K, V$:** Bản chất các phép chiếu không gian từ biểu diễn đầu vào qua các ma trận trọng số học được $W^Q, W^K, W^V$.
- **Multi-Head Attention:** Cơ chế chia nhỏ $d_{model}$ thành $h$ không gian con $d_k$, tính toán attention song song và ghép nối qua $W^O$.
- **Positional Encoding:** Công thức tần số sin/cos đa bước sóng và vai trò phá vỡ tính bất biến hoán vị.
- **Residual Connection:** Cơ chế bảo toàn đặc trưng gốc và tạo cao tốc gradient $\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$.
- **Layer Normalization:** Cơ chế chuẩn hóa theo trục đặc trưng độc lập trên từng token, vượt trội hơn BatchNorm trong xử lý chuỗi.
- **Position-wise FFN:** Mạng nơ-ron truyền thẳng biến đổi phi tuyến cục bộ tại từng vị trí, nâng cao dung lượng tri thức.
