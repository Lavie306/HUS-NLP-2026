# LAB 08 — Pretrained Transformers và Tinh Chỉnh Mô Hình BERT (BERT Fine-Tuning)
## From Training a Transformer to Using a Pretrained Language Model

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP)  
**Học kỳ:** Học kỳ I — Năm học 2026  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Chủ đề:** Pretrained Transformers and BERT Fine-Tuning  

---

## 1. Mạch Chuyển Tiếp Kiến Thức & Giới Thiệu Tổng Quan

Trong tiến trình học tập của học phần:
- Ở **LAB 07**, chúng ta đã tự tay lập trình từng thành phần toán học nguyên thủy của một khối **Transformer Encoder** (Positional Encoding, Multi-Head Self-Attention, Residual Connections, Layer Normalization, Feed-Forward Network) và huấn luyện trên bài toán phân loại từ con số không (*training from scratch*). Mặc dù kiến trúc Transformer vượt trội về khả năng tính toán song song so với RNN/LSTM, việc huấn luyện từ đầu đòi hỏi mô hình phải vừa tự khám phá các quy luật ngữ pháp cơ bản của ngôn ngữ loài người, vừa phải học phân biệt mục tiêu bài toán. Khi dữ liệu của tác vụ đích bị giới hạn, mô hình rất khó hội tụ và dễ rơi vào quá khớp.

**LAB 08 đặt câu hỏi bước ngoặt của NLP hiện đại:**
> *Nếu một mô hình Transformer đã được tiền huấn luyện trước (Pre-trained) trên một kho ngữ liệu khổng lồ hàng tỷ từ vựng, làm thế nào để chuyển giao và tận dụng kho tri thức biểu diễn ngôn ngữ đó cho một bài toán chuyên biệt mới (Fine-tuning)?*

### Sơ đồ mạch tiến hóa học phần:
$$\begin{array}{c}
\textbf{LAB 07: Transformer Trained from Scratch} \\
\Downarrow \\
\textbf{LAB 08: Pretraining} \longrightarrow \textbf{Masked Language Modeling (MLM)} \longrightarrow \textbf{BERT} \longrightarrow \textbf{Fine-tuning} \longrightarrow \textbf{Text Classification} \\
\Downarrow \\
\textbf{LAB 09: Token Classification / Named Entity Recognition (NER) / Information Extraction}
\end{array}$$

---

## 2. Danh Mục Hồ Sơ Nộp Bài (Deliverables Checklist — Mục 25)

Toàn bộ các tài liệu học thuật, mã nguồn, bảng dữ liệu định lượng và đồ thị của LAB 08 được bố trí chuẩn hóa tại thư mục `d:/NLP/Lab8/` theo đúng danh mục đề bài:

```text
d:/NLP/Lab8/
├── README.md                      # Báo cáo tổng quan, số liệu thực nghiệm, vấn đáp cá nhân và tính tái lập
├── W8.pdf                         # Đề bài và quy định thực hiện LAB 08
├── calculations.md                # Lời giải chi tiết Calculation Exercise 1 (MLM Loss) & Exercise 2 (Tokenization)
├── prediction.md                  # 3 dự đoán lý thuyết trước thực nghiệm và bảng đối chiếu thực nghiệm
├── pretrained_classification.py   # Module mã nguồn Python cài đặt lớp PretrainedClassifier và pipeline phân loại
├── tokenization.ipynb             # Jupyter Notebook minh họa quy trình Tokenization & Attention Mask (Mục 10 & 12.1)
├── fine_tuning.ipynb              # Jupyter Notebook thực thi toàn diện 5 thí nghiệm huấn luyện và đánh giá
├── experiments.ipynb              # Bản sao notebook lưu trữ toàn bộ lịch sử thực nghiệm trên môi trường GPU
├── results.csv                    # Bảng dữ liệu định lượng tổng hợp tất cả các chỉ số đo đạc từ 5 thí nghiệm
├── error_samples.csv              # Dữ liệu trích xuất các mẫu dự đoán sai thực tế từ tập Test IMDb
├── learning_curves.png            # Đồ thị so sánh đường cong suy giảm hàm mất mát giữa Pretrain và Scratch
├── confusion_matrix.png           # Ma trận nhầm lẫn của mô hình BERT Fine-Tuning trên tập kiểm thử IMDb
├── data_efficiency.png            # Đồ thị hiệu quả dữ liệu: So sánh Pretrained vs From-scratch (10% - 100%)
├── confidence_analysis.png        # Biểu đồ phân tích độ tin cậy và minh chứng nguyên lý Confidence != Correctness
├── error_analysis.md              # Báo cáo phân tích định tính lỗi thực tế và trả lời 6 câu hỏi bắt buộc (Q1 - Q6)
└── reflection.md                  # Bài luận phản tư 300-500 từ về lý do không train Transformer from scratch
```

---

## 3. Tính Tái Lập Thực Nghiệm (Reproducibility Protocol — Mục 23)

Để đảm bảo tính toàn vẹn khoa học và khả năng tái hiện kết quả thực nghiệm một cách minh bạch, toàn bộ các thông số môi trường và siêu tham số được công bố chi tiết:

| Tham số cấu hình | Giá trị thiết lập thực tế | Ý nghĩa / Ghi chú kỹ thuật |
| :--- | :--- | :--- |
| **Mô hình nền tảng (Backbone)** | `bert-base-uncased` | Kiến trúc 12 layers, hidden size 768, 12 attention heads, ~110M tham số |
| **Bộ tách từ (Tokenizer)** | WordPiece Tokenizer | Từ điển $|\mathcal{V}| = 30,522$ tokens, chuẩn hóa chữ thường (uncased) |
| **Tập dữ liệu (Dataset)** | IMDb Movie Reviews | Bài toán phân loại cảm xúc nhị phân (*Negative* vs. *Positive*) |
| **Phân chia mẫu (Splits)** | Train: 4,000 / Val: 1,000 / Test: 1,000 | Cân bằng tỷ lệ 50:50 giữa hai lớp nhãn trong mọi tập |
| **Độ dài chuỗi tối đa ($T$)** | $T = 128$ (mặc định) | Được khảo sát biến thiên $T \in \{64, 128, 256\}$ trong Exp 4 |
| **Kích thước Batch (Batch Size)** | $16$ | Kích thước batch phù hợp với dung lượng bộ nhớ VRAM |
| **Tốc độ học (Learning Rate)** | $2 \times 10^{-5}$ | Áp dụng Linear Warmup trong 10% bước đầu và Linear Decay |
| **Số Epochs huấn luyện** | $3$ epochs | Điểm cân bằng tối ưu giữa khả năng hội tụ và tránh overfitting |
| **Thuật toán tối ưu (Optimizer)** | AdamW | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$, Weight Decay = $0.01$ |
| **Hạt giống ngẫu nhiên (Seed)** | $42$ | Đảm bảo tính nhất quán khi chia tập dữ liệu và khởi tạo tham số |
| **Môi trường phần cứng (Hardware)** | GPU Tesla T4 (Google Colab) / Intel Core i7 | Bộ nhớ GPU 16GB, hỗ trợ CUDA và tăng tốc TensorFloat32 |
| **Thời gian huấn luyện trung bình** | $\approx 178.34$ giây | Đo đạc cho 3 epochs Full Fine-Tuning với 4,000 mẫu train |

---

## 4. Bảng Tổng Hợp Kết Quả Thực Nghiệm Định Lượng (Mục 14 — 18)

Dữ liệu được trích xuất trực tiếp và đồng bộ nguyên vẹn từ [results.csv](file:///d:/NLP/Lab8/results.csv):

| Mã thí nghiệm | Thiết lập thực nghiệm (Setting) | Khởi tạo (Init) | Encoder Trainable? | Kích thước Train | Max Length | Accuracy | Precision | Recall | Macro F1 | Thời gian (s) | # Tham số tối ưu |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1** | Transformer from scratch | Ngẫu nhiên | Có | 4,000 | 128 | **0.4980** | 0.0000 | 0.0000 | 0.0000 | 170.19s | 108.89M |
| **Exp 1** | **Pretrained BERT Fine-tuning** | **Pretrained** | **Có** | **4,000** | **128** | **0.8600** | **0.8390** | **0.8924** | **0.8649** | **178.34s** | **108.89M** |
| **Exp 2** | Frozen Encoder (Feature Extraction) | Pretrained | Không | 4,000 | 128 | **0.7790** | 0.7607 | 0.8167 | 0.7877 | **65.68s** | **1,538** |
| **Exp 3** | Pretrained BERT (10% Dữ liệu) | Pretrained | Có | 400 | 128 | **0.8260** | 0.8333 | 0.8167 | 0.8249 | 29.34s | 108.89M |
| **Exp 3** | From Scratch (10% Dữ liệu) | Ngẫu nhiên | Có | 400 | 128 | **0.5020** | 0.5020 | 1.0000 | 0.6684 | 29.32s | 108.89M |
| **Exp 3** | Pretrained BERT (25% Dữ liệu) | Pretrained | Có | 1,000 | 128 | **0.8440** | 0.8327 | 0.8625 | 0.8474 | 54.28s | 108.89M |
| **Exp 3** | From Scratch (25% Dữ liệu) | Ngẫu nhiên | Có | 1,000 | 128 | **0.5020** | 0.5020 | 1.0000 | 0.6684 | 53.97s | 108.89M |
| **Exp 3** | Pretrained BERT (50% Dữ liệu) | Pretrained | Có | 2,000 | 128 | **0.8470** | 0.8483 | 0.8466 | 0.8475 | 95.59s | 108.89M |
| **Exp 3** | From Scratch (50% Dữ liệu) | Ngẫu nhiên | Có | 2,000 | 128 | **0.5020** | 0.5020 | 1.0000 | 0.6684 | 95.06s | 108.89M |
| **Exp 4** | Độ dài chuỗi $T = 64$ | Pretrained | Có | 4,000 | 64 | **0.8230** | 0.8095 | 0.8466 | 0.8277 | 94.59s | 108.89M |
| **Exp 4** | Độ dài chuỗi $T = 256$ | Pretrained | Có | 4,000 | 256 | **0.9030** | 0.8917 | 0.9183 | 0.9048 | 370.93s | 108.89M |
| **Exp 5** | In-domain (IMDb $\to$ IMDb Test) | Pretrained | Có | 4,000 | 128 | **0.8600** | 0.8390 | 0.8924 | 0.8649 | — | — |
| **Exp 5** | Out-of-domain (IMDb $\to$ Amazon Reviews) | Pretrained | Có | 4,000 | 128 | **0.9040** | 0.8621 | 0.9690 | 0.9124 | — | — |

---

## 5. Phân Tích Chuyên Sâu 5 Thí Nghiệm Cốt Lõi

### Thí Nghiệm 1: Pretrained vs. Train From Scratch (Mục 14)
* **Kết quả đo đạc:** Pretrained BERT đạt Accuracy **$86.00\%$** và F1 **$0.8649$**, trong khi Transformer from scratch chỉ đạt **$49.80\%$** (dự đoán toàn bộ nhãn Negative).
* **Bản chất khoa học:** Một mạng Transformer khởi tạo ngẫu nhiên hoàn toàn không thể học đồng thời cấu trúc ngữ pháp ngôn ngữ tự nhiên và nhiệm vụ phân loại cảm xúc khi chỉ được cung cấp 4,000 mẫu dữ liệu. Việc kế thừa biểu diễn ngôn ngữ từ giai đoạn tiền huấn luyện là yếu tố quyết định để mô hình thoát khỏi cực tiểu suy biến.

### Thí Nghiệm 2: Feature Extraction (Frozen) vs. Full Fine-Tuning (Mục 15)
* **Kết quả đo đạc:** Đóng băng Encoder chỉ cần tối ưu hóa đúng $1,538$ tham số ở tầng Linear Head nhưng vẫn đạt tới **$77.90\%$** Accuracy trong $65.68$ giây (nhanh gấp gần 3 lần). Full Fine-Tuning giúp nâng Accuracy thêm **$+8.10\%$** lên $86.00\%$.
* **Bản chất khoa học:** Biểu diễn ẩn tại vị trí `[CLS]` của BERT vốn đã phân cụm ngữ nghĩa tương đối tách biệt. Tuy nhiên, việc cho phép gradient lan truyền ngược qua toàn bộ 12 tầng sẽ giúp tái định hình các ma trận Attention hướng vào từ ngữ cảm xúc chuyên biệt, đem lại bước nhảy vọt về chất lượng phân loại.

### Thí Nghiệm 3: Hiệu Quả Sử Dụng Dữ Liệu (Data Efficiency Curve — Mục 16)
* **Kết quả đo đạc:** Quan sát biểu đồ `data_efficiency.png`, Pretrained BERT chỉ cần $10\%$ dữ liệu ($400$ mẫu) đã đạt độ chính xác **$82.60\%$**, cao hơn áp đảo mô hình from scratch chạy trên $100\%$ dữ liệu ($49.80\%$). Khi tăng lượng mẫu từ 400 lên 4,000, độ chính xác tăng đều từ $82.60\% \to 84.40\% \to 84.70\% \to 86.00\%$.
* **Bản chất khoa học:** Mô hình tiền huấn luyện sở hữu tính hiệu quả mẫu (Sample Efficiency) vượt bậc, giúp các bài toán thực tiễn đạt độ chính xác khả thi ngay cả khi dữ liệu gán nhãn cực kỳ eo hẹp.

### Thí Nghiệm 4: Tác Động Của Độ Dài Chuỗi Tối Đa & Độ Phức Tạp $\mathcal{O}(T^2)$ (Mục 17)
* **Kết quả đo đạc:** Mở rộng $T = 64 \to 128 \to 256$ giúp Accuracy tăng từ $82.30\% \to 86.00\% \to 90.30\%$. Đổi lại, thời gian huấn luyện tăng vọt gần gấp 4 lần ($94.59\text{s} \to 370.93\text{s}$) và bộ nhớ GPU đỉnh tăng từ $4.7\text{GB}$ lên $8.5\text{GB}$.
* **Bản chất khoa học:** Phép nhân ma trận chú ý $QK^T$ trong Self-Attention có chi phí $\mathcal{O}(T^2)$. Do đó, việc tăng $T$ đòi hỏi sự cân nhắc đánh đổi khắt khe giữa lợi ích độ chính xác và gánh nặng chi phí tài nguyên phần cứng.

### Thí Nghiệm 5: Hiện Tượng Dịch Chuyển Miền Dữ Liệu (Domain Shift — Mục 18)
* **Kết quả đo đạc:** Mô hình fine-tune trên IMDb Movie Reviews khi kiểm tra trên Amazon Product Reviews đạt Accuracy **$90.40\%$** (cao hơn tập kiểm thử gốc $86.00\%$).
* **Bản chất khoa học:** Mặc dù chuyển đổi domain, Amazon Reviews có cấu trúc ngắn gọn, từ ngữ khen/chê trực diện hơn review phim ảnh, nên mô hình dễ dàng phát hiện tín hiệu cảm xúc. Tuy nhiên, điều này không đồng nghĩa mô hình "hiểu mọi domain"; nếu đem kiểm tra trên các văn bản học thuật y sinh hoặc pháp lý, hiệu năng chắc chắn sẽ suy giảm mạnh mẽ do sự phân kỳ phân phối từ vựng.

---

## 6. Trả Lời Vấn Đáp Kiểm Tra Năng Lực Cá Nhân (Individual Learning Check — Mục 27)

*Dưới đây là phần trả lời ngắn gọn, khúc chiết, diễn đạt bằng tư duy học thuật độc lập cho 12 câu hỏi bắt buộc:*

1. **Pretraining khác fine-tuning ở đâu?**  
   Pretraining là quá trình huấn luyện tự giám sát không nhãn trên kho văn bản khổng lồ để học biểu diễn ngôn ngữ tổng quát; Fine-tuning là việc tiếp tục tối ưu hóa mô hình đã pretrain trên một tập dữ liệu nhỏ có nhãn cho một bài toán cụ thể.

2. **MLM (Masked Language Modeling) là gì?**  
   Là nhiệm vụ tự giám sát che ngẫu nhiên một số token trong câu và buộc mô hình phải khôi phục lại các token đó dựa trên ngữ cảnh hai chiều xung quanh.

3. **Tại sao cần tokenizer?**  
   Vì mạng nơ-ron chỉ tính toán được trên các ma trận số thực; tokenizer làm nhiệm vụ phân đoạn văn bản thô thành các đơn vị rời rạc và ánh xạ chúng thành các chỉ số định danh số học nguyên thủy (`input_ids`).

4. **Subword token giải quyết vấn đề gì?**  
   Giải quyết triệt để rào cản từ ngoài từ điển (Out-Of-Vocabulary - OOV) và hạn chế kích thước từ vựng phình to, bằng cách bẻ nhỏ các từ hiếm thành các đơn vị từ con phổ biến mang gốc nghĩa hình thái học.

5. **Token `[CLS]` thường được sử dụng như thế nào?**  
   Luôn đứng ở vị trí đầu câu để tích hợp thông tin toàn cảnh qua các lớp Self-Attention; vector trạng thái ẩn cuối cùng tại `[CLS]` được dùng làm đầu vào cho bộ phân loại chuỗi văn bản.

6. **Mặt nạ chú ý (Attention mask) dùng để làm gì?**  
   Dùng để chỉ thị vị trí nào là token thật ($1$) và vị trí nào là đệm kỹ thuật (`[PAD]` mang giá trị $0$), nhằm gán trọng số chú ý tại các vị trí đệm về $0$ trong hàm Softmax, tránh làm nhiễu biểu diễn câu.

7. **Frozen encoder nghĩa là gì?**  
   Là thiết lập cố định toàn bộ trọng số của khối Transformer Encoder (`requires_grad = False`), chỉ cho phép cập nhật tham số ở tầng phân loại đầu ra (Classification Head).

8. **Fine-tuning thay đổi những tham số nào?**  
   Trong chế độ Full Fine-Tuning, toàn bộ tham số của mô hình (tất cả các ma trận Self-Attention, Feed-Forward, LayerNorm trong 12 tầng Encoder cùng với tầng Classification Head) đều được cập nhật gradient.

9. **Vì sao pretrained model có thể hữu ích khi downstream dataset nhỏ?**  
   Vì mô hình đã tích lũy sẵn nền tảng ngữ pháp và ngữ nghĩa phong phú; khi đối mặt với tập dữ liệu nhỏ, nó chỉ cần học thêm một ánh xạ phân lớp đơn giản thay vì phải học lại ngôn ngữ từ đầu.

10. **Pretrained model có đảm bảo không bị domain shift không?**  
    Hoàn toàn không. Mô hình vẫn bị suy giảm hiệu năng nghiêm trọng khi gặp các miền dữ liệu có phân phối từ vựng, tiếng lóng hoặc ngữ cảnh văn phong sai lệch lớn so với dữ liệu huấn luyện.

11. **Vì sao tăng sequence length làm chi phí attention tăng?**  
    Vì cơ chế Self-Attention tính toán sự tương tác giữa mọi cặp token trong câu, dẫn tới kích thước ma trận tương quan là $T \times T$, khiến thời gian và bộ nhớ tăng theo độ phức tạp bậc hai $\mathcal{O}(T^2)$.

12. **Confidence cao có đảm bảo prediction đúng không?**  
    Không. Giá trị Confidence chỉ là xác suất cực đại qua hàm Softmax phản ánh biên độ logit toán học, không phản ánh chân lý ngữ nghĩa; mô hình vẫn có thể sai lầm nghiêm trọng với độ tự tin vượt $99.5\%$.

---

## 7. Khai Báo Sử Dụng Công Cụ Trợ Giúp AI (AI Usage Declaration — Mục 24)

Tuân thủ nghiêm ngặt chuẩn mực liêm chính học thuật theo quy định tại Mục 24 của tài liệu LAB 08:

* **AI-assisted code:** Hỗ trợ chuẩn hóa cấu trúc bộ khung (boilerplate) cho module phân loại và định dạng các bảng trực quan hóa.
* **What was generated:** Cấu trúc lớp `PretrainedClassifier` cơ bản và các đoạn mã tiện ích xử lý định dạng đồ thị Matplotlib.
* **What was modified:** Thiết kế cơ chế trích xuất vector `[CLS]` chuẩn mực, cài đặt cờ khóa tham số `freeze_encoder` với `requires_grad = False`, bổ sung Dropout và thiết lập chính xác mặt nạ chú ý khi đệm theo batch.
* **What was verified:** Kiểm chứng thủ công các phép tính giải tích của hàm MLM Loss, xác minh kích thước các tensor đầu ra, kiểm tra sự khớp lệnh của ma trận nhầm lẫn và trực tiếp phân tích nguyên nhân ngữ nghĩa của 8 ca lỗi thực tế.

---

## 8. Hướng Dẫn Thực Thi Mã Nguồn (Execution Guide)

### Kiểm tra module phân loại và Pipeline:
```powershell
python D:/NLP/Lab8/pretrained_classification.py --demo_text "The movie was extraordinarily captivating!"
```

### Chạy chế độ Feature Extraction (Frozen Encoder):
```powershell
python D:/NLP/Lab8/pretrained_classification.py --freeze_encoder
```

### Mở và chạy các Notebook thực nghiệm:
Khởi chạy Jupyter Notebook hoặc mở trong Visual Studio Code:
- [tokenization.ipynb](file:///d:/NLP/Lab8/tokenization.ipynb): Minh họa tách từ, mã định danh và mặt nạ chú ý.
- [fine_tuning.ipynb](file:///d:/NLP/Lab8/fine_tuning.ipynb): Thực thi toàn bộ 5 thí nghiệm huấn luyện trên GPU.
