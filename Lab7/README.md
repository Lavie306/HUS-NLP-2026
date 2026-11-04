# LAB 07 — Transformer Architecture
## From Attention to Transformer: Kiến trúc Xử lý Ngôn ngữ Tự nhiên Hiện đại

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — Năm học 2026  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Chủ đề:** From Attention to Transformer  

---

## 1. Giới thiệu Tổng quan & Bối cảnh Học thuật

Trong chương trình môn học:
- Ở **LAB 05**, sinh viên đã làm quen với các mô hình hồi quy tuần tự (RNN, LSTM, GRU). Chuỗi đầu vào được duyệt tuần tự qua từng bước thời gian $t = 1, \dots, T$. Mặc dù mô hình hóa được thứ tự từ, cấu trúc hồi quy vấp phải hai rào cản nền tảng: **điểm nghẽn tính toán tuần tự** (sequential bottleneck, không thể tính toán song song trên GPU) và **điểm nghẽn thông tin** (information bottleneck, nén toàn bộ ngữ cảnh vào một vector ẩn duy nhất $h_T$).
- Ở **LAB 06**, cơ chế **Attention** và **Self-Attention** được giới thiệu để giải quyết triệt để rào cản cự ly:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
  Self-Attention cho phép mọi cặp từ trong câu tương tác trực tiếp ở khoảng cách $O(1)$ và tính toán song song.

**LAB 07 đặt câu hỏi tiếp theo:**  
> *Nếu Attention cho phép các vị trí trong chuỗi tương tác trực tiếp và song song, làm thế nào để ghép Attention thành một kiến trúc mạng nơ-ron hoàn chỉnh, sâu, và huấn luyện ổn định để xử lý văn bản?*

Kiến trúc **Transformer** (Vaswani et al., 2017) trả lời bằng một hệ thống kết hợp 4 thành phần trụ cột:
1. **Positional Encoding (PE):** Bổ sung tọa độ vị trí dạng hình sin đa tần số, giải quyết tính bất biến hoán vị của Self-Attention.
2. **Multi-Head Self-Attention (MHA):** Tách không gian biểu diễn thành nhiều không gian con độc lập, nắm bắt đồng thời nhiều góc nhìn cú pháp và ngữ nghĩa.
3. **Position-wise Feed-Forward Network (FFN):** Biến đổi phi tuyến sâu cục bộ tại từng token, mở rộng dung lượng bộ nhớ tri thức.
4. **Residual Connections & Layer Normalization (Add & Norm):** Giữ nguyên dòng truyền gradient (gradient highway), ổn định phương sai biểu diễn, cho phép xếp chồng nhiều khối sâu.

---

## 2. Cấu trúc Thư mục Deliverables (Mục 32)

Toàn bộ các tệp báo cáo, mã nguồn, đồ thị và số liệu của LAB 07 được tổ chức chuẩn hóa theo đúng đặc tả:

```
d:/NLP/Lab7/
├── README.md                          # Báo cáo tổng quan, phân tích và hướng dẫn thực thi
├── calculations.md                    # Báo cáo chi tiết lý thuyết, bài tập tính toán và trace shape
├── prediction.md                      # Dự đoán lý thuyết trước thực nghiệm và bảng đối chiếu kết quả
├── transformer_block.py               # Cài đặt khối Transformer Encoder từ primitives và unit test shape
├── positional_encoding.py             # Cài đặt Sinusoidal PE, Experiment 1 & Permutation Experiment
├── transformer_classification.ipynb   # Jupyter Notebook thực thi toàn bộ luồng thực nghiệm
├── results.csv                        # Bảng tổng hợp số liệu định lượng (Acc, Prec, Rec, F1, Time, Params)
├── positional_encoding.png            # Biểu đồ heatmap PE và phân biệt câu hoán vị từ
├── attention_heads.png                # Trực quan hóa sự phân hóa đa dạng giữa 4 Attention Heads
├── learning_curves.png                # Đồ thị đường cong mất mát (Train/Val Loss) giữa LSTM và Transformer
├── confusion_matrix.png               # Ma trận nhầm lẫn của mô hình Transformer trên 2,500 mẫu test
├── error_analysis.md                  # Phân tích định tính 3 mẫu đúng, 3 mẫu sai và cảnh báo luận giải
└── reflection.md                      # Chiêm nghiệm mạch tiến hóa LAB 01->07, trả lời vấn đáp & AI statement
```

---

## 3. Kiến trúc Khối Transformer Encoder Tối giản (Mục 17 & 18)

Mô hình được xây dựng thủ công từ các tầng nguyên thủy (`nn.Linear`, `nn.LayerNorm`, `nn.Dropout`, `F.softmax`), tuyệt đối **không** dùng sẵn `nn.TransformerEncoderLayer`:

```
                    Đầu vào Tensor X: [B, T, d_model]
                                  │
                                  ▼
                     Embedding + Positional Encoding
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │ (Residual Connection 1)
                 ▼                                 │
     Multi-Head Self-Attention                     │
                 │                                 │
                 ▼                                 │
                 └──────────────► (+) ◄────────────┘
                                  │
                                  ▼
                              LayerNorm
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │ (Residual Connection 2)
                 ▼                                 │
        Position-wise FFN                          │
                 │                                 │
                 ▼                                 │
                 └──────────────► (+) ◄────────────┘
                                  │
                                  ▼
                              LayerNorm
                                  │
                                  ▼
                    Đầu ra Tensor: [B, T, d_model]
```

### Bảng Kiểm chứng Shape Tensor (Mục 18 — Shape Verification)
Cấu hình chuẩn: $B = 8, T = 100, d_{model} = 128, n_{heads} = 4 \implies d_k = 32, d_{ff} = 256$:

| Tầng mạng / Bước chuyển tiếp | Ký hiệu | Kích thước Tensor (Shape) | Ý nghĩa cấu trúc |
|:---|:---:|:---:|:---|
| **Input Tensor** | $X$ | `[8, 100, 128]` | Batch 8 câu, mỗi câu 100 token, vector 128 chiều |
| **Q, K, V Projections** | $Q, K, V$ | `[8, 4, 100, 32]` | Chiếu tách thành 4 heads, mỗi head 32 chiều |
| **Attention Scores** | $S = Q K^T / \sqrt{d_k}$ | `[8, 4, 100, 100]` | Ma trận tương tác cặp vị trí cho từng head |
| **Attention Weights** | $A = \text{softmax}(S)$ | `[8, 4, 100, 100]` | Phân phối xác suất chuẩn hóa trên từng hàng |
| **Head Context Outputs** | $Z_{heads} = A V$ | `[8, 4, 100, 32]` | Tổng có trọng số của các vector Value |
| **Concatenated Output** | $Z_{concat}$ | `[8, 100, 128]` | Ghép nối 4 heads trở lại không gian 128 chiều |
| **Attention Projection** | $Z_{out} = Z_{concat} W^O$ | `[8, 100, 128]` | Chiếu tuyến tính hòa trộn thông tin giữa các heads |
| **Add & Norm 1** | $X_1 = \text{LN}(X + Z_{out})$ | `[8, 100, 128]` | Residual connection và chuẩn hóa LayerNorm |
| **FFN Hidden Layer** | $H_{ffn} = \text{ReLU}(X_1 W_1 + b_1)$ | `[8, 100, 256]` | Mở rộng không gian phi tuyến lên $d_{ff} = 256$ |
| **FFN Output Layer** | $Z_{ffn} = H_{ffn} W_2 + b_2$ | `[8, 100, 128]` | Thu hẹp về chiều $d_{model} = 128$ |
| **Add & Norm 2 (Output)** | $\text{Out} = \text{LN}(X_1 + Z_{ffn})$ | `[8, 100, 128]` | Residual connection và LayerNorm hoàn tất block |

---

## 4. Tổng hợp Kết quả Thực nghiệm Định lượng

Tất cả các thử nghiệm được thực hiện trên tập dữ liệu IMDb (chia phân tầng: 10,000 huấn luyện, 2,500 validation, 2,500 kiểm thử; từ điển 20,000 từ).

### 4.1. Experiment 4: So sánh LSTM Baseline vs Transformer Encoder (Mục 21)

| Mô hình (Model) | Bản chất Biểu diễn | Số tham số | Thời gian huấn luyện (s) | Accuracy | Precision | Recall | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LSTM Baseline (LAB 05)** | Recurrent Hidden State ($h_T$) | $2,692,481$ | $20.25$ | $0.5800$ | $0.6604$ | $0.3363$ | $0.4456$ |
| **Transformer Encoder (4 Heads)** | Multi-Head Self-Attention + FFN | $2,692,353$ | $158.59$ | **$0.7088$** | **$0.8133$** | **$0.5450$** | **$0.6527$** |

> **Nhận định:** Transformer Encoder vượt trội toàn diện so với LSTM (F1 tăng từ $0.4456 \to 0.6527$, tức tăng $+20.71\%$, Accuracy tăng $+12.88\%$). Đường truyền thông tin trực tiếp $O(1)$ giữa mọi cặp từ giúp mô hình vượt qua hoàn toàn hiện tượng phai nhòa ngữ cảnh của tế bào hồi quy.

---

### 4.2. Experiment 2: Khảo sát Số lượng Attention Heads (Mục 19)
Khảo sát với $d_{model} = 128$ cố định:

| Số lượng Heads ($n_{heads}$) | Chiều mỗi Head ($d_k$) | Số tham số | Thời gian (s/epoch) | Accuracy | F1-Score |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 Head** | $128$ | $2,692,353$ | $62.78$ | $0.6544$ | $0.6749$ |
| **2 Heads** | $64$ | $2,692,353$ | $68.12$ | $0.6828$ | $0.7205$ |
| **4 Heads** | $32$ | $2,692,353$ | $79.68$ | **$0.7240$** | **$0.7298$** |
| **8 Heads** | $16$ | $2,692,353$ | $98.03$ | $0.7184$ | $0.6871$ |

> **Trả lời câu hỏi:**
> 1. *Tăng số heads có đồng nghĩa với tăng biểu diễn dimension không?*  
>    **Không.** Tổng số chiều biểu diễn vẫn giữ nguyên là $d_{model} = 128$.
> 2. *Khi $d_{model}$ cố định, tăng số heads làm $d_k$ thay đổi như thế nào?*  
>    $d_k$ giảm nghịch biến tỷ lệ: $d_k = d_{model} / n_{heads}$. Cấu hình $n_{heads} = 4$ ($d_k = 32$) đạt điểm cân bằng tối ưu. Khi $n_{heads} = 8$, $d_k$ thu hẹp còn $16$, dung lượng không gian con bị phân mảnh khiến F1 suy giảm.

---

### 4.3. Experiment 5: Sequence Length & Chi phí Bậc hai $O(T^2)$ (Mục 22)

| Độ dài chuỗi ($T$) | Tỷ lệ Tokens giữ lại | Số phần tử Attention Matrix ($T^2$) | Bộ nhớ tương đối | Thời gian (s/batch) | F1 ước tính |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **32** | $13.7\%$ | $1,024$ | $1.0\times$ | $0.0995$ | $0.6800$ |
| **64** | $27.0\%$ | $4,096$ | $4.0\times$ | $0.2231$ | $0.7300$ |
| **128** | $51.1\%$ | $16,384$ | $16.0\times$ | $0.5313$ | $0.7800$ |
| **256** | $76.5\%$ | $65,536$ | $64.0\times$ | $1.3695$ | $0.8300$ |

> **Trả lời câu hỏi:** *Tại sao Transformer không đơn giản giải quyết mọi vấn đề bằng cách tăng sequence length?*  
> Bởi vì ma trận Attention có kích thước $T \times T$. Khi tăng độ dài chuỗi gấp đôi ($128 \to 256$), số phép tính và dung lượng bộ nhớ lưu ma trận Attention tăng **gấp 4 lần** ($16,384 \to 65,536$, gấp $16\times \to 64\times$). Ở độ dài chuỗi hàng nghìn tokens, Transformer gặp nghẽn cổ chai bộ nhớ (Out-Of-Memory) và nghẽn băng thông truyền tải.

---

### 4.4. Experiment 6: So sánh Chiến lược Pooling (Mục 24 & 25)

| Chiến lược Pooling (Pooling Strategy) | Độ chính xác (Accuracy) | Điểm F1-Score | Thời gian huấn luyện (s) |
|:---|:---:|:---:|:---:|
| **Masked Mean Pooling** | **$0.7088$** | **$0.6527$** | $158.59$ |
| **First-token / CLS-style Pooling** | $0.5496$ | $0.5761$ | $173.72$ |

> **Nhận xét:** Masked Mean Pooling vượt trội rõ rệt ($+15.92\%$ Accuracy). Lý do là vì ở mô hình Transformer nông (1 tầng Encoder) học từ đầu không có token `[CLS]` được huấn luyện trước đặc thù, token đầu tiên không đủ dung lượng để tổng hợp toàn bộ thông tin toàn câu, trong khi Mean Pooling gom góp đều đặn tín hiệu có nghĩa từ tất cả các vị trí.

---

### 4.5. Thử nghiệm trên 5 Ca biên Phức tạp (Application Probe — Mục 23)

| Nhóm ca biên (Category) | Nội dung câu thử nghiệm | Nhãn dự đoán | Xác suất Positive | Đánh giá |
|:---|:---|:---:|:---:|:---:|
| **Câu ngắn** | *"A masterpiece of acting and directing, absolutely brilliant!"* | `Positive` | $0.9919$ | Hoàn hảo |
| **Câu dài** | *"I spent two full hours sitting through this production and although the visuals and costume design were mildly interesting, the narrative dragged endlessly without any coherent emotional arc."* | `Positive` | $0.8267$ | Nhầm lẫn do nửa đầu khen |
| **Phủ định (Negation)** | *"The movie is not good at all, completely devoid of any humor or charm."* | `Negative` | $0.1101$ | Bắt trúng phủ định |
| **Tương phản (Contrast)** | *"The cinematography and acting were exceptional, but the plot was a total catastrophe and utterly boring."* | `Negative` | $0.0047$ | Bắt trúng chuyển ý "but" |
| **Mơ hồ (Ambiguous)** | *"An eccentric experiment in modern cinema with bizarre pacing that will strongly divide audiences."* | `Positive` | $0.8956$ | Nghiêng về khen thử nghiệm |

---

## 5. Trực quan hóa Kết quả Thực nghiệm

Toàn bộ biểu đồ phân tích thực nghiệm đã được xuất bản độ phân giải cao (300 DPI):

1. **Positional Encoding & Permutation Test:**  
   [positional_encoding.png](file:///D:/NLP/Lab7/positional_encoding.png)  
   *Minh họa ma trận tần số sin/cos, chứng minh hai câu hoán vị "dog bites man" vs "man bites dog" trùng khít 100% khi thiếu PE và tách biệt rõ rệt khi có PE.*
2. **Attention Head Diversity:**  
   [attention_heads.png](file:///D:/NLP/Lab7/attention_heads.png)  
   *Trực quan hóa sự phân hóa chức năng của 4 Attention Heads (Head 1: lân cận, Head 2: cảm xúc, Head 3: liên từ tương phản, Head 4: phân tán toàn cục).*
3. **Đường cong Học tập (Learning Curves):**  
   [learning_curves.png](file:///D:/NLP/Lab7/learning_curves.png)  
   *So sánh động lực hội tụ hàm mất mát giữa LSTM và Transformer Encoder qua các epoch.*
4. **Ma trận Nhầm lẫn (Confusion Matrix):**  
   [confusion_matrix.png](file:///D:/NLP/Lab7/confusion_matrix.png)  
   *Ma trận nhầm lẫn chi tiết trên 2,500 mẫu kiểm thử của tập IMDb.*

---

## 6. Hướng dẫn Tái lập Kết quả Thực nghiệm (Replication Guide)

Mọi thực nghiệm đều được đóng gói độc lập, có thể tái lập nhanh chóng qua dòng lệnh PowerShell / Terminal:

### 1. Kiểm tra Shape và Logic Transformer Block:
```powershell
python transformer_block.py
```

### 2. Chạy Thử nghiệm Positional Encoding & Permutation:
```powershell
python positional_encoding.py
```

### 3. Huấn luyện Toàn bộ Mô hình và Xuất Biểu đồ / Số liệu:
```powershell
python run_all_experiments.py
```

### 4. Khởi chạy Jupyter Notebook:
Mở và thực thi toàn bộ các khối lệnh tuần tự trong tệp [transformer_classification.ipynb](file:///D:/NLP/Lab7/transformer_classification.ipynb).

---

## 7. Mạch nối Tiến hóa & Động lực Chuyển giao sang LAB 08

LAB 07 khép lại với nhận thức sâu sắc về sức mạnh kiến trúc của Transformer nhưng đồng thời cũng chỉ ra giới hạn của việc **huấn luyện Transformer từ đầu (from scratch) trên dữ liệu nhỏ**: mô hình thiếu inductive bias, cực kỳ "đói dữ liệu", và dễ bị quá khớp.

Điều này tạo ra động lực chuyển giao tự nhiên sang **LAB 08: Pretrained Transformer & Fine-Tuning (BERT)**:
$$\text{Pretraining trên kho ngữ liệu khổng lồ (Masked LM + Next Sentence Prediction)}$$
$$\downarrow$$
$$\text{Biểu diễn ngôn ngữ tổng quát sâu sắc}$$
$$\downarrow$$
$$\text{Fine-Tuning nhẹ nhàng cho tác vụ phân loại văn bản (Text Classification)}$$

Nhờ đã tự tay lắp ráp và phân tích cặn kẽ từng ma trận của khối Transformer Encoder ở LAB 07, sinh viên bước sang LAB 08 với nền tảng kiến trúc vững chắc, thấu hiểu tường tận bản chất của vector `[CLS]`, vị trí Positional Encoding, và cơ chế Multi-Head Self-Attention bên dưới các mô hình ngôn ngữ lớn hiện đại.
