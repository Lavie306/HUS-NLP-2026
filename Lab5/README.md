# LAB 05 — SEQUENCE MODELS: RNN, LSTM AND GRU
## From Static Representations to Sequential Modeling

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — Năm học 2026  
**Trường:** ĐH Khoa học Tự nhiên — ĐHQGHN (VNU-HUS)  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Sinh viên thực hiện:** Nguyễn Trọng Thành  
**Hạn nộp:** 23:59 — 21/10/2026  

---

## 1. Giới thiệu và Bối cảnh (Context)

Ở **LAB 04**, sinh viên đã sử dụng các biểu diễn tĩnh như **TF-IDF** và **Average Word2Vec** để đưa văn bản vào bài toán phân loại cảm xúc (sentiment classification). Tuy nhiên, các phương pháp biểu diễn tĩnh bộc lộ một hạn chế căn bản: **văn bản là một chuỗi có thứ tự, nhưng representation tĩnh không mô hình hóa thứ tự một cách trực tiếp**. 

Ví dụ:
- \(S_1 =\) `"this movie is good"`
- \(S_2 =\) `"this movie is not good"`

Khi tính trung bình vector \(\vec{x} = \frac{1}{n} \sum_{i=1}^n \vec{w}_i\), thông tin vị trí bị triệt tiêu do tính giao hoán của phép cộng.

**LAB 05** chuyển sang tiếp cận theo mô hình hóa chuỗi (Sequential Modeling):
\[
\text{Text} \longrightarrow \text{Token sequence} \longrightarrow \text{Word embeddings} \longrightarrow \text{RNN / LSTM / GRU} \longrightarrow \text{Hidden state } (h_T) \longrightarrow \text{Classifier} \longrightarrow \text{Prediction}
\]

Mục tiêu cốt lõi của lab là giải thích:
- Tại sao sequence model có thể khai thác thông tin thứ tự và ngữ cảnh động?
- Tại sao Simple RNN vẫn gặp hiện tượng vanishing gradient khi chuỗi dài?
- Cơ chế cổng (gates) của LSTM và GRU giải quyết vấn đề lưu giữ thông tin xa như thế nào?
- Nút thắt cổ chai của mạng hồi quy và động lực dẫn tới cơ chế Attention (LAB 06) và Transformer (LAB 07).

---

## 2. Mục tiêu học tập (Learning Outcomes)

- [x] **LO1:** Giải thích sequence modeling trong NLP.
- [x] **LO2:** Mô tả cơ chế recurrent computation của RNN.
- [x] **LO3:** Tính một bước forward của RNN bằng tay và kiểm chứng với PyTorch.
- [x] **LO4:** Giải thích hidden state và luồng thông tin (information flow) qua chuỗi.
- [x] **LO5:** Nhận diện hiện tượng vanishing gradient và exploding gradient qua BPTT.
- [x] **LO6:** Giải thích động lực của LSTM và vai trò điều phối của các cổng (gates).
- [x] **LO7:** Mô tả sự khác nhau về kiến trúc giữa RNN, LSTM và GRU.
- [x] **LO8:** Xây dựng hoàn chỉnh sequence classification pipeline trên IMDb.
- [x] **LO9:** So sánh mô hình baseline không tuần tự (TF-IDF + LR) với RNN/LSTM/GRU.
- [x] **LO10:** Phân tích ảnh hưởng của sequence length, trade-off tính toán và padding/truncation.
- [x] **LO11:** Phân tích ma trận nhầm lẫn và nguyên nhân lỗi của sequence classifier.
- [x] **LO12:** Nhận diện giới hạn cố hữu của recurrent models và động lực dẫn tới Attention.

---

## 3. Cấu trúc thư mục nộp bài (Deliverables theo Mục 29 W5.pdf)

Toàn bộ các file yêu cầu theo đặc tả của giảng viên được tổ chức hoàn chỉnh tại thư mục `Lab5/`:

```text
Lab5/
├── W5.pdf                         # Đề bài và quy chuẩn thực hành Lab 05
├── README.md                      # Báo cáo tổng quan, hướng dẫn chạy và tóm tắt kết quả
├── calculations.md                # Tính toán giải tích từng bước (RNN, LSTM gates, BPTT)
├── prediction.md                  # 5 dự đoán độc lập ghi lại trước khi chạy thực nghiệm
├── rnn_cell.py                    # Cài đặt SimpleRNNCell từ đầu và kiểm chứng với PyTorch
├── run_experiments.py             # Pipeline thực nghiệm hoàn chỉnh, tự động và tái lập
├── sequence_classification.ipynb  # Jupyter Notebook chạy độc lập, lưu sẵn toàn bộ output và chart
├── results.csv                    # Bảng số liệu định lượng chi tiết của tất cả các mô hình
├── experiment_details.json        # Metadata thống kê dataset, learning history, probes
├── learning_curves.png            # Biểu đồ hàm mất mát Train/Val qua các epoch
├── confusion_matrix.png           # Ma trận nhầm lẫn của mô hình GRU trên held-out test set
├── sequence_lengths.png           # Histogram phân bố độ dài 50,000 bài đánh giá IMDb
├── length_tradeoff.png            # Biểu đồ trade-off giữa độ dài chuỗi, F1 và thời gian train
├── error_analysis.md              # Bảng phân tích chi tiết ít nhất 3 mẫu đúng và 3 mẫu sai
├── reflection.md                  # Báo cáo nhận xét, phân ánh học tập và mạch nối sang Attention
├── requirements.txt               # Danh sách thư viện phụ thuộc
└── gru_model.pt                   # Checkpoint trọng số mô hình GRU tốt nhất đã huấn luyện
```

---

## 4. Hướng dẫn cài đặt và Chạy thực nghiệm

### 4.1. Môi trường và Thư viện phụ thuộc
Khuyến nghị sử dụng Python 3.10+ hoặc 3.11+. Cài đặt các thư viện cần thiết:

```powershell
cd D:\NLP\Lab5
pip install -r requirements.txt
```

### 4.2. Chạy kiểm chứng Simple RNN Cell
Chạy file kiểm chứng tính toán một bước forward bằng tay đối chiếu với `torch.nn.RNNCell`:

```powershell
python rnn_cell.py
```
*Kết quả in ra màn hình:*
```text
============================================================
KIỂM TRA TÍNH TOÁN BÀI 1 & ĐỐI CHIẾU PYTORCH
============================================================
Custom SimpleRNNCell output : [[0.5849799  0.29131263]]
PyTorch nn.RNNCell output   : [[0.5849799  0.29131263]]
Analytical expected output  : [[0.5849799  0.29131263]]
Khớp tuyệt đối (allclose)   : True
============================================================
```

### 4.3. Chạy toàn bộ pipeline thực nghiệm
Chạy script tái lập tự động từ đầu đến cuối:

```powershell
python run_experiments.py
```
Pipeline sẽ:
1. Đọc và thống kê 50,000 mẫu dữ liệu IMDb, sinh biểu đồ `sequence_lengths.png`.
2. Chia tập stratified train/val/test và xây dựng từ vựng 12,000 từ chỉ trên tập train.
3. Chạy Baseline TF-IDF + Logistic Regression.
4. Huấn luyện và đánh giá RNN, LSTM, GRU tại `max_length = 100`.
5. Huấn luyện thử nghiệm độ dài `max_length` \(\in \{50, 100, 200\}\) trên GRU.
6. Xuất các file `results.csv`, `learning_curves.png`, `confusion_matrix.png`, `length_tradeoff.png`, `error_analysis.md`, `experiment_details.json`, và `gru_model.pt`.

### 4.4. Mở Jupyter Notebook kiểm tra trực quan
```powershell
jupyter notebook sequence_classification.ipynb
```
File notebook đã được chạy sẵn và lưu đầy đủ kết quả đầu ra, bảng số liệu biểu đồ và phân tích của từng cell.

---

## 5. Tóm tắt Kết quả Thực nghiệm Định lượng

Dưới đây là bảng tổng hợp kết quả đánh giá trên cùng tập held-out test set 1,000 bài đánh giá cân bằng (trích xuất từ `results.csv`):

| Mô hình | Biểu diễn | Max Length | Tham số | Thời gian train (s) | Epochs | % Token giữ lại | Validation Loss | Test Accuracy | Precision | Recall | Test F1 |
|---|---|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + LR** | TF-IDF Unigram | Toàn bộ | 12,001 | 2.85s | — | 100.0% | — | **0.8540** | **0.8430** | 0.8700 | **0.8563** |
| **Simple RNN** | Embedding + RNN | 100 | 776,385 | 17.19s | 2 | 41.63% | 0.6995 | 0.4940 | 0.4931 | 0.4300 | 0.4594 |
| **LSTM** | Embedding + LSTM | 100 | 801,345 | 31.90s | 2 | 41.63% | 0.6868 | 0.5530 | 0.5560 | 0.5260 | 0.5406 |
| **GRU** | Embedding + GRU | 100 | 793,025 | 12.21s | 2 | 41.63% | 0.6892 | **0.5730** | **0.5641** | **0.6420** | **0.6006** |
| **GRU (Len 50)** | Embedding + GRU | 50 | 793,025 | 6.66s | 2 | 21.48% | 0.6937 | 0.5390 | 0.5349 | 0.5980 | 0.5647 |
| **GRU (Len 200)**| Embedding + GRU | 200 | 793,025 | 167.60s | 2 | 68.93% | 0.6771 | **0.5670** | 0.5446 | **0.8180** | **0.6539** |

---

## 6. Tổng hợp các Phép đo Định tính (Probes)

### 6.1. Order Sensitivity Probe (Độ nhạy với thứ tự từ)
Kiểm tra khả năng phân biệt cặp câu đảo vị trí từ ngữ:

| Câu đầu vào | TF-IDF + LR \(P(\text{positive})\) | GRU \(P(\text{positive})\) | Nhận xét |
|---|:---:|:---:|---|
| `"this movie is not good"` | **0.4769** | **0.4383** | TF-IDF cho cùng 1 kết quả vì túi từ giống nhau. |
| `"this movie is good not"` | **0.4769** | **0.4469** | GRU nhạy cảm với thứ tự từ, phân biệt được 2 vị trí. |
| `"the film is not bad"` | **0.0284** | **0.4926** | TF-IDF bị lừa bởi từ "bad" nên dự đoán cực thấp. |
| `"the film is bad not"` | **0.0284** | **0.4822** | GRU phân biệt được nhờ luồng thông tin có thứ tự. |

### 6.2. Application Probe (Dự đoán trên 4 loại văn bản thực tế)
- **Review ngắn:** `"I really enjoyed this movie."` \(\longrightarrow P(\text{pos}) = 0.482\)
- **Review dài:** `"The film begins slowly with several unnecessary scenes, but the acting becomes powerful and the ending is wonderful. I enjoyed the movie overall."` \(\longrightarrow P(\text{pos}) = 0.427\)
- **Review phủ định:** `"I did not like this movie."` \(\longrightarrow P(\text{pos}) = 0.456\)
- **Review tương phản:** `"I thought the movie would be terrible, but it was surprisingly good."` \(\longrightarrow P(\text{pos}) = 0.420\)

### 6.3. Long-range Position Probe
Khi dịch chuyển vị trí cụm từ cảm xúc then chốt (*"Ultimately it was excellent."* vs *"Ultimately it was terrible."*) qua 3 vị trí (đầu, giữa, cuối) của bài viết, độ chênh lệch xác suất \(\Delta P < 0.01\), cho thấy mô hình mạng nơ-ron huấn luyện trong thời gian ngắn (2 epochs) vẫn bị chi phối bởi độ dài và chưa tối ưu hóa trọn vẹn trọng số phụ thuộc xa.

---

## 7. Chuẩn bị cho Kiểm tra Đánh giá Cá nhân (Individual Learning Check — Mục 28)

Dưới đây là câu trả lời cô đọng cho 8 câu hỏi phỏng vấn nhanh 3 phút của Giảng viên:

1. **RNN khác Average Word2Vec ở điểm nào?**
   - *Trả lời:* Average Word2Vec lấy trung bình cộng các vector từ, có tính giao hoán nên làm mất hoàn toàn thứ tự từ và cấu trúc ngữ pháp. RNN cập nhật trạng thái ẩn \(h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)\) tuần tự theo thời gian, lưu giữ thứ tự xuất hiện và biểu diễn ngữ cảnh động phụ thuộc vào các từ đi trước.
2. **Hidden state \(h_t\) có ý nghĩa gì?**
   - *Trả lời:* \(h_t\) là vector trạng thái nén toàn bộ thông tin ngữ cảnh của chuỗi token từ bước \(1\) đến bước \(t\). Nó hoạt động như một "bộ nhớ làm việc" truyền qua các bước thời gian.
3. **Tại sao RNN có thể gặp vanishing gradient?**
   - *Trả lời:* Khi backpropagation qua thời gian (BPTT), gradient \(\frac{\partial L}{\partial h_t}\) chứa tích chuỗi của nhiều ma trận Jacobian liên tiếp \(\prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}}\). Do đạo hàm hàm \(\tanh \in (0, 1]\), nếu các trị riêng của ma trận trọng số hồi quy \(W_{hh} < 1\), tích này suy giảm theo hàm mũ về 0 khi khoảng cách thời gian \((T - t)\) lớn, khiến gradient ở các bước đầu chuỗi bị biến mất.
4. **Forget gate của LSTM dùng để làm gì?**
   - *Trả lời:* Forget gate \(f_t = \sigma(W_f [x_t, h_{t-1}] + b_f)\) quyết định tỷ lệ thông tin cũ từ ô nhớ \(c_{t-1}\) được giữ lại hay xóa bỏ. Khi \(f_t \approx 1\), thông tin được bảo toàn nguyên vẹn; khi \(f_t \approx 0\), bộ nhớ được xóa sạch để nạp thông tin mới.
5. **Tại sao tăng sequence length có thể làm training chậm hơn?**
   - *Trả lời:* Vì mạng phải mở rộng (unroll) thêm nhiều bước thời gian. Mỗi bước tăng thêm đòi hỏi thêm các phép nhân ma trận thuận và ngược qua BPTT, làm chi phí tính toán tăng tuyến tính theo \(T\), đồng thời tốn thêm bộ nhớ RAM/GPU để lưu trữ các activation trung gian.
6. **GRU khác LSTM ở điểm nào?**
   - *Trả lời:* GRU không có ô nhớ (cell state \(c_t\)) riêng mà chỉ dùng hidden state \(h_t\). GRU chỉ có 2 cổng (Reset gate và Update gate) thay vì 3 cổng như LSTM (Forget, Input, Output), do đó GRU có ít hơn ~25% tham số, huấn luyện nhanh hơn và ít bị overfit trên tập dữ liệu nhỏ.
7. **Tại sao model có training loss giảm nhưng validation loss tăng?**
   - *Trả lời:* Đây là biểu hiện kinh điển của hiện tượng **Overfitting (Quá khớp)**. Mô hình bắt đầu học thuộc lòng nhiễu và các đặc trưng ngẫu nhiên trong tập huấn luyện thay vì học quy luật tổng quát, dẫn đến khả năng dự đoán trên dữ liệu chưa nhìn thấy (validation set) bị suy giảm.
8. **Nếu một review bị truncate ở 30 tokens, thông tin gì có thể bị mất?**
   - *Trả lời:* Độ dài trung bình của review IMDb là 231 từ. Cắt ở 30 tokens sẽ làm mất phần lớn nội dung chính, các câu chuyển ý tương phản (*"however", "but"*) và đặc biệt là nhận định kết luận cảm xúc của người viết thường được đặt ở cuối bài đánh giá.

---

## 8. Bảng kiểm tra Đánh giá Chéo (Peer Assessment Checklist — Mục 31)

- [x] **Dataset:** Mô tả đầy đủ 50,000 dòng; báo cáo phân bố 2 lớp; phân tích độ dài min/median/mean/max/quantiles; giải thích padding/truncation.
- [x] **Implementation:** SimpleRNNCell tự code từ đầu; forward computation chính xác; so sánh với PyTorch `nn.RNNCell`; chạy baseline TF-IDF + LR; so sánh RNN vs LSTM vs GRU.
- [x] **Experiment:** Thử nghiệm độ dài 50, 100, 200 tokens; vẽ biểu đồ Learning Curves (train/val loss); báo cáo đủ 4 metrics phân loại; đo lường chi phí tính toán và số tham số.
- [x] **Analysis:** Xuất biểu đồ Confusion Matrix; phân tích sâu ít nhất 3 mẫu đúng và 3 mẫu sai; phân tích ngữ cảnh, từ phủ định, cấu trúc tương phản; không kết luận phiến diện chỉ dựa vào Accuracy.
- [x] **Understanding:** Giải thích thấu đáo ý nghĩa Hidden State, cơ chế điều phối LSTM Gates, bản chất toán học của Vanishing Gradient, và mạch nối sang Attention (LAB 06).

---

## 9. Tuyên bố Sử dụng Trí tuệ Nhân tạo (AI Assistance Statement — Mục 27)

Tuân thủ nghiêm ngặt quy định tại Mục 27 (`W5.pdf`):
- **Công cụ sử dụng:** Antigravity AI Assistant (Google DeepMind).
- **Mục đích hỗ trợ:** Hỗ trợ viết mã boilerplate cho DataLoader, hỗ trợ cấu hình hiển thị biểu đồ Matplotlib, định dạng tài liệu Markdown chuẩn công thức toán học LaTeX, và kiểm tra tính toán đối chiếu.
- **Phần được sinh:** Khung mã nguồn PyTorch và cấu trúc bảng Markdown ban đầu.
- **Phần tự chỉnh sửa và hoàn thiện:** Kiểm chứng công thức toán học Bài 1 và Bài 5, viết phân tích sâu ngữ nghĩa các ca lỗi sai, kiểm chứng kết quả chạy thực nghiệm trên tập dữ liệu IMDb, đối chiếu tính toán với `torch.nn.RNNCell`.
- **Kiểm chứng xác thực:** Toàn bộ mã nguồn đã được chạy thực tế, lưu sinh ra các file ảnh biểu đồ, checkpoint `gru_model.pt`, và bảng số liệu `results.csv` tại thư mục bài làm.
