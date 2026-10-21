# BÁO CÁO NHẬN XÉT VÀ PHẢN ÁNH HỌC TẬP (REFLECTION REPORT) — LAB 05

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng — Học kỳ I, 2026  
**Chủ đề:** Sequence Models: RNN, LSTM and GRU — From Static Representations to Sequential Modeling  
**Căn cứ tài liệu:** Toàn bộ các yêu cầu, câu hỏi thảo luận và rubric tại `W5.pdf`

---

## 1. Phân tích tập dữ liệu và Độ dài chuỗi (Dataset & Sequence Length Analysis)

### 1.1. Thống kê toàn diện bộ dữ liệu IMDb Movie Reviews
Quá trình kiểm tra toàn bộ 50,000 dòng của tập dữ liệu IMDb Movie Reviews thu được các thông số định lượng cụ thể:
- **Tổng số mẫu:** 50,000 bài đánh giá phim.
- **Số lớp và phân bố:** Đúng 2 lớp cân bằng tuyệt đối (25,000 nhãn `positive` và 25,000 nhãn `negative`), không xảy ra hiện tượng mất cân bằng lớp (class imbalance).
- **Dữ liệu thiếu và trùng lặp:** Không có bất kỳ dòng nào bị thiếu văn bản (`missing_text = 0`) hoặc thiếu nhãn (`missing_labels = 0`). Có 418 bài đánh giá bị trùng lặp nguyên văn (`duplicate_documents = 418`) đã được loại bỏ trước khi chia tập huấn luyện.
- **Phân bố độ dài văn bản (sau khi tiền xử lý và tách từ):**
  - Độ dài ngắn nhất: **6 tokens**.
  - Trung vị (Median): **173.0 tokens**.
  - Trung bình (Mean): **231.28 tokens**.
  - Độ dài dài nhất: **2,477 tokens**.
  - Các phân vị quan trọng: Phân vị 50% là 173 tokens, phân vị 75% là 280 tokens, phân vị 90% là 452 tokens, và phân vị 95% là 589 tokens.

Biểu đồ phân bố độ dài (`sequence_lengths.png`) cho thấy phân bố lệch phải rất mạnh (heavy right-skewed tail). Phần lớn bài review tập trung trong khoảng từ 80 đến 300 từ, nhưng có một lượng bài đánh giá cực dài kéo dài tới hơn 1,000 đến gần 2,500 từ.

### 1.2. Tiêu chí lựa chọn `max_length` trong thực tế
> **Câu hỏi đặt ra trong W5.pdf:** *Nếu phải chọn `max_length`, nên chọn dựa trên thông tin nào? Không chấp nhận câu trả lời chỉ là "Chọn 100 vì model chạy nhanh".*

Việc chọn `max_length` là một bài toán tối ưu hóa đa mục tiêu cần dựa trên 4 căn cứ định lượng và thực tế:
1. **Phân vị tích lũy của độ dài (Cumulative Length Percentile):** Cần chọn giá trị bao phủ phần lớn các mẫu trong tập dữ liệu (ví dụ: chọn phân vị 75% tương đương 280 tokens, hoặc phân vị 90% tương đương 452 tokens) để hạn chế tối đa việc mất thông tin do cắt cụt.
2. **Tỷ lệ token có nghĩa được giữ lại (Percentage of Tokens Retained):** 
   - Tại `max_length = 50`: chỉ giữ lại được **21.48%** lượng token của toàn bộ tập dữ liệu (gần 79% nội dung bị vứt bỏ).
   - Tại `max_length = 100`: giữ lại được **41.63%** lượng token.
   - Tại `max_length = 200`: giữ lại được **68.93%** lượng token.
3. **Đặc trưng ngôn ngữ học của thể loại văn bản:** Trong đánh giá phim (movie reviews), người viết thường tóm tắt nội dung ở các đoạn mở đầu và đưa ra nhận định cảm xúc kết luận (*"overall", "in conclusion", "ultimately a masterpiece"*) ở phần cuối bài. Do đó, `max_length` quá ngắn sẽ làm mất hoàn toàn phần kết luận mang tính quyết định này.
4. **Ràng buộc tài nguyên tính toán và chi phí thời gian:** Chi phí tính toán của các mạng hồi quy theo thời gian unroll có độ phức tạp thời gian tuyến tính \(\mathcal{O}(T \times d)\) và bộ nhớ \(\mathcal{O}(T \times B \times d)\). Chuỗi càng dài thì thời gian lan truyền ngược BPTT càng lớn và bộ nhớ GPU/RAM càng tăng.

### 1.3. Phân biệt Padding và Truncation
- **Padding:** Thêm token giả định (thường là `PAD` với ID = 0) vào cuối chuỗi có độ dài ngắn hơn `max_length` để đưa tất cả các mẫu trong cùng một batch về cùng kích thước ma trận cố định. 
  - *Lưu ý quan trọng:* Padding không mang bất kỳ thông tin ngôn ngữ nào. Trong cài đặt chuẩn, kỹ thuật `pack_padded_sequence` được sử dụng để ngăn trạng thái ẩn cuối cùng tiếp tục cập nhật qua các token `PAD`, đảm bảo vector đặc trưng đưa vào phân loại phản ánh đúng từ kết thúc thực sự của văn bản.
- **Truncation:** Cắt bỏ vĩnh viễn phần văn bản vượt quá `max_length`. Nếu `max_length` quá nhỏ (ví dụ 30 tokens), gần như toàn bộ các cấu trúc ngữ pháp phức hợp, câu chuyển ý phản biện (*"I initially thought the movie would be terrible, but..."*), và lời kết luận chung cuộc đều bị triệt tiêu hoàn toàn.

---

## 2. Đánh giá Mô hình cơ sở và So sánh Kiến trúc Hồi quy (Baseline vs RNN vs LSTM vs GRU)

### 2.1. Bảng tổng hợp kết quả thực nghiệm định lượng

Dưới đây là kết quả đánh giá trên cùng tập kiểm thử độc lập (held-out test set 1,000 mẫu cân bằng) với cùng bộ từ vựng, kích thước embedding (64), kích thước hidden (64), batch size (64), tốc độ học Adam (0.001) và 2 epochs huấn luyện:

| Mô hình | Biểu diễn (Representation) | Max Length | Số tham số (Parameters) | Thời gian huấn luyện (giây) | Accuracy | Precision | Recall | F1-Score |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| **TF-IDF + LR** | TF-IDF Unigram (Sparse) | Toàn bộ | **12,001** | **~2.85s** | **0.8540** | **0.8430** | **0.8700** | **0.8563** |
| **Simple RNN** | Dense Embedding + RNN | 100 | 776,385 | ~17.19s | 0.4940 | 0.4931 | 0.4300 | 0.4594 |
| **LSTM** | Dense Embedding + LSTM | 100 | 801,345 | ~31.90s | 0.5530 | 0.5560 | 0.5260 | 0.5406 |
| **GRU** | Dense Embedding + GRU | 100 | 793,025 | **~12.21s** | **0.5730** | **0.5641** | **0.6420** | **0.6006** |

*(Ghi chú: Lớp Embedding chiếm 12,000 \(\times\) 64 = 768,000 tham số. Số tham số thuần của khối recurrent: RNN có 8,320 tham số, GRU có 24,960 tham số, và LSTM có 33,280 tham số).*

### 2.2. Phân tích đối chiếu với Dự đoán ban đầu (Predictions)
1. **Dự đoán 1 (Khả năng xử lý Word Order):** 
   - Hoàn toàn chính xác. Kiểm chứng qua Order Probe với cặp câu đảo thứ tự:
     - Câu 1: `"this movie is not good"` vs Câu 2: `"this movie is good not"`:
       - Mô hình TF-IDF + Logistic Regression cho ra **chính xác cùng một xác suất dự đoán** (\(P(\text{positive}) = 0.4769\)) cho cả 2 câu do vector túi từ giống hệt nhau.
       - Mô hình GRU cho ra **hai xác suất khác nhau** (\(P_1 = 0.4383\) vs \(P_2 = 0.4469\)), chứng minh mô hình hồi quy thực sự nhạy cảm với thứ tự xuất hiện của token.
2. **Dự đoán 2 (LSTM/GRU vs RNN trong phụ thuộc dài):**
   - Hoàn toàn chính xác. Simple RNN chỉ đạt F1 là **0.4594** (tương đương đoán ngẫu nhiên trên phân loại nhị phân), trong khi LSTM đạt **0.5406** và GRU đạt **0.6006**. Simple RNN bị suy thoái do vanishing gradient khiến mô hình khó học được mối liên kết qua 100 bước thời gian.
3. **Dự đoán 5 (LSTM có luôn thắng thế trên mọi điều kiện?):**
   - Hoàn toàn chính xác. Trong thực nghiệm này, GRU đạt F1 cao hơn LSTM (0.6006 so với 0.5406) đồng thời chạy nhanh hơn gần 2.6 lần (12.2s so với 31.9s). GRU có cấu trúc gọn gàng hơn (2 cổng thay vì 3 cổng, không có cell state riêng biệt), ít tham số hơn nên hội tụ nhanh hơn hẳn trong điều kiện giới hạn 2 epochs.
4. **Tại sao Baseline TF-IDF + Logistic Regression lại vượt trội các mô hình Deep Learning?**
   - Đây là một phát hiện vô cùng quan trọng: Trong bài toán phân loại cảm xúc IMDb với số lượng mẫu huấn luyện nhỏ (5,000 mẫu) và ngân sách huấn luyện ngắn (2 epochs trên CPU), các từ khóa mang cảm xúc đơn lẻ như *"superb", "brilliant", "boring", "awful"* đóng vai trò là những tín hiệu phân loại cực kỳ mạnh.
   - TF-IDF kết hợp Logistic Regression trực tiếp gắn trọng số lớn cho các từ khóa này mà không cần học biểu diễn từ đầu, đạt ngay F1 = **0.8563** chỉ sau 2.8 giây.
   - Ngược lại, các mô hình RNN/LSTM/GRU phải học đồng thời cả không gian embedding 64 chiều từ đầu (scratch) lẫn các ma trận cổng phi tuyến phức tạp. Hai epochs là quá ngắn để các mạng hồi quy hội tụ tối ưu trên dữ liệu nhỏ.

---

## 3. Ảnh hưởng của Độ dài Chuỗi và Đánh đổi Tính toán (Sequence Length Trade-off)

### 3.1. Kết quả thực nghiệm trên các độ dài khác nhau của mô hình GRU
Giữ nguyên toàn bộ cấu hình, chỉ thay đổi `max_length` \(\in \{50, 100, 200\}\):

| Max Length | % Token giữ lại | Thời gian huấn luyện (s) | Validation Loss (Epoch 2) | Test F1-Score |
|:---:|:---:|:---:|:---:|:---:|
| **50** | 21.48% | **6.66s** | 0.6937 | 0.5647 |
| **100** | 41.63% | **12.21s** | 0.6892 | 0.6006 |
| **200** | 68.93% | **167.60s** | **0.6771** | **0.6539** |

### 3.2. Phân tích biểu đồ Trade-off (`length_tradeoff.png`)
Biểu đồ `length_tradeoff.png` minh họa rõ nét sự đánh đổi (trade-off) mang tính cốt lõi trong NLP:
1. **Chất lượng mô hình tăng theo độ dài ngữ cảnh:** Khi tăng `max_length` từ 50 lên 200, F1-Score tăng từ **0.5647 lên 0.6539** (tăng gần 9 điểm phần trăm), và Validation Loss giảm đều đặn từ 0.6937 xuống 0.6771. Việc giữ lại 68.93% nội dung thay vì chỉ 21.48% cung cấp ngữ cảnh đầy đủ hơn để mô hình nắm bắt được cảm xúc chung.
2. **Chi phí thời gian tăng đột biến:** Thời gian huấn luyện tăng từ **6.66s (50 tokens) lên 12.21s (100 tokens), và bùng nổ lên tới 167.60s (200 tokens)** — tức tăng hơn 13.7 lần khi độ dài chuỗi tăng gấp đôi từ 100 lên 200!
3. **Kết luận:** Tồn tại một sự đánh đổi sâu sắc giữa việc mở rộng ngữ cảnh (context) và chi phí tài nguyên tính toán (computational cost). Trong các hệ thống thực tế, không thể tùy tiện tăng `max_length` lên vô hạn mà phải cân bằng giữa giới hạn phần cứng (GPU memory, latency) và mức cải thiện hiệu năng cận biên.

---

## 4. Đường cong Huấn luyện và Dấu hiệu Overfitting (Learning Curves)

Quan sát biểu đồ `learning_curves.png`:
- **Simple RNN:** Train loss giảm nhẹ từ 0.699 xuống 0.684, nhưng Validation Loss lại tăng từ 0.699 lên 0.700. Đây là dấu hiệu ban đầu điển hình của hiện tượng chững lại hoặc bắt đầu overfit sớm: mô hình cố gắng ghi nhớ một số mẫu huấn luyện nhưng khả năng tổng quát hóa trên tập kiểm định không cải thiện do gradient bị suy hao.
- **LSTM và GRU:** Cả Train Loss và Validation Loss đều giảm đồng nhịp qua 2 epochs (ví dụ GRU giảm từ ~0.692 xuống ~0.689 trên validation), chứng tỏ các cơ chế cổng kiểm soát dòng thông tin hiệu quả hơn và quá trình tối ưu hóa diễn ra ổn định.
- **Nhận định chung:** Hai epochs là một ngân sách ngắn. Để chẩn đoán toàn diện hiện tượng overfitting, cần theo dõi đồ thị qua 5–10 epochs để xác định điểm cực tiểu của validation loss (early stopping checkpoint).

---

## 5. Phân tích Ma trận Nhầm lẫn và Các Dạng Lỗi (Error Analysis)

Ma trận nhầm lẫn của mô hình GRU (100 tokens) trên 1,000 mẫu kiểm thử:
- True Negative (TN): **252** mẫu
- False Positive (FP): **248** mẫu
- False Negative (FN): **179** mẫu
- True Positive (TP): **321** mẫu

### Phân loại các nguyên nhân lỗi chính (từ `error_analysis.md`):
1. **Lỗi do Cắt cụt độ dài (Truncation):** 
   - Ví dụ các bài đánh giá dài từ 700 đến hơn 900 tokens (như bài review phim *"Secret Sunshine"* dài 719 từ hay *"Mute Witness"* dài 909 từ). 
   - Các bài này dành 100 từ đầu tiên để kể cốt truyện một cách trung tính hoặc mô tả các chi tiết u ám, tang tóc; nhận định khen ngợi của tác giả nằm ở các đoạn sau. Khi bị cắt tại token thứ 100, mô hình chỉ nhìn thấy các từ mang màu sắc ảm đạm và dự đoán sai thành tiêu cực (\(P \approx 0.45 - 0.48\)).
2. **Lỗi do Cấu trúc Tương phản (Contrast) và Chuyển ý:**
   - Nhiều bài review mở đầu bằng lời khen diễn viên hoặc kỹ xảo nhưng chuyển ý chê thậm tệ đạo diễn bằng các liên từ *"however", "but", "nevertheless"*. Mặc dù GRU có khả năng duy trì context, nếu các từ tích cực xuất hiện áp đảo về mặt từ vựng, vector trạng thái ẩn vẫn bị thiên lệch.
3. **Lỗi do Phủ định phức tạp (Complex Negation):**
   - Các cấu trúc như *"not without flaw"*, *"hardly a disappointment"*, *"I can't say I disliked it"* đòi hỏi việc kết hợp ngữ nghĩa tinh tế giữa phủ định kép hoặc bổ ngữ. Mô hình chuỗi với dung lượng nhỏ và số epoch ít chưa đủ khả năng giải mã các tầng ngữ nghĩa này.
4. **Không phải mọi lỗi đều do Sequence Modeling:**
   - Cần khẳng định rõ ràng: Một lỗi phân loại không mặc định quy kết là do kiến trúc RNN/LSTM/GRU không tốt. Các nguyên nhân ngoại cảnh như: từ vựng hiếm bị thay thế bằng mã `<UNK>`, nhãn dữ liệu có tính mơ hồ (ambiguous ground-truth), và giới hạn phần cứng buộc phải cắt ngắn câu đều đóng góp trực tiếp vào sai số.

---

## 6. Khảo sát Độ nhạy Vị trí Từ khóa (Long-Range Position Probe)

Trong thực nghiệm khảo sát vị trí (Section 21), từ khóa cảm xúc then chốt (*"Ultimately it was excellent."* vs *"Ultimately it was terrible."*) được đặt ở 3 vị trí khác nhau trong văn bản giả lập:
1. Đặt ở **Đầu câu (Beginning)**
2. Đặt ở **Giữa câu (Middle)**
3. Đặt ở **Cuối câu (End)**

Kết quả quan sát trên mô hình GRU:
- Độ chênh lệch xác suất \(\Delta P\) giữa câu chứa *"excellent"* và câu chứa *"terrible"* dao động rất nhỏ (\(< 0.01\)).
- **Phân biệt rạch ròi giữa quan sát thực nghiệm và lý thuyết:**
  - *Về mặt thực nghiệm (Empirical):* Mô hình trong điều kiện này thể hiện sự kém nhạy cảm với việc thay đổi từ khóa, nguyên nhân chủ yếu do ngân sách huấn luyện ngắn (2 epochs) khiến trọng số embedding và phân loại chưa phân hóa sắc nét.
  - *Về mặt lý thuyết (Theoretical):* Hiện tượng phụ thuộc xa bị suy hao trong RNN bắt nguồn từ tích chuỗi Jacobian BPTT \(\prod \frac{\partial h_k}{\partial h_{k-1}}\). Không được tùy tiện dùng một thí nghiệm đồ chơi (toy probe) nhỏ để khẳng định đã chứng minh được vanishing gradient, mà cần phân tích đúng bản chất toán học của cấu trúc mạng.

---

## 7. Mạch nối từ LAB 01 tới Attention và Transformer (Curriculum Synthesis)

### 7.1. Bảng so sánh tiến trình biểu diễn và mô hình hóa (LAB 01 \(\to\) LAB 05)

| Thuộc tính (Property) | TF-IDF + LR (LAB 01/04) | Average Word2Vec + LR (LAB 03/04) | Simple RNN (LAB 05) | LSTM (LAB 05) | GRU (LAB 05) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Biểu diễn thưa (Sparse)** | **Có** | Không | Không | Không | Không |
| **Biểu diễn dày (Dense)** | Không | **Có** | **Có** | **Có** | **Có** |
| **Bảo toàn thứ tự từ (Word Order)** | Rất hạn chế (chỉ qua n-gram) | **Không thể (Giao hoán)** | **Có (Tuần tự)** | **Có (Tuần tự)** | **Có (Tuần tự)** |
| **Trạng thái phụ thuộc ngữ cảnh** | Không | Không | **Có (\(h_t\))** | **Có (\(h_t, c_t\))** | **Có (\(h_t\))** |
| **Xử lý quan hệ xa (Long Dependency)** | Không | Không | Kém (Vanishing) | Cải thiện (Gates) | Cải thiện (Gates) |
| **Chi phí tính toán (Compute Cost)** | Cực thấp | Thấp | Cao hơn | Cao nhất | Cao |
| **Cơ chế biểu diễn từ** | Tần suất từ | Vector tĩnh toàn cục | Vector ngữ cảnh hóa tuần tự | Vector ngữ cảnh hóa tuần tự | Vector ngữ cảnh hóa tuần tự |

### 7.2. RNN/LSTM/GRU giải quyết được gì mà Average Word2Vec không giải quyết được?
1. **Khôi phục thông tin thứ tự thời gian:** Nhờ cơ chế \(h_t = f(x_t, h_{t-1})\), mô hình phân biệt được chủ ngữ, tân ngữ và các cụm cú pháp đảo ngược.
2. **Biểu diễn phụ thuộc ngữ cảnh động:** Từ `"like"` khi đứng sau `"not"` có trạng thái ẩn hoàn toàn khác với `"like"` đứng sau `"really"`.
3. **Khả năng tích lũy thông tin qua chuỗi:** Trạng thái ẩn \(h_T\) nén thông tin tiến trình của cả bài review thay vì chỉ tính trung bình bề mặt.

### 7.3. Những hạn chế cốt lõi còn tồn tại của Mạng Hồi quy (Recurrent Bottlenecks)
Mặc dù là bước nhảy vọt so với các biểu diễn tĩnh, RNN/LSTM/GRU vẫn vấp phải hai rào cản nền tảng:
1. **Nút thắt cổ chai thông tin (Information Bottleneck):**
   - Toàn bộ chuỗi văn bản dài \(T\) từ (có thể là 500 hay 1,000 từ) bắt buộc phải bị nén ép tuần tự vào một vector trạng thái ẩn \(h_T\) có số chiều cố định (ví dụ 64 hoặc 128 chiều). Điều này gây thất thoát thông tin nghiêm trọng đối với các chi tiết nằm ở đầu câu.
2. **Đường truyền thông tin cưỡng bức qua thời gian (Linear Distance Path):**
   - Để token thứ 100 tương tác được với token thứ 1, thông tin buộc phải đi qua 99 bước biến đổi trung gian:
     \[
     x_1 \to x_2 \to x_3 \to \dots \to x_{100}
     \]
   - Khoảng cách truyền thông tin là \(\mathcal{O}(T)\). Dù LSTM và GRU có các cổng giúp duy trì dòng thông tin, qua hàng trăm bước lặp, tín hiệu ban đầu vẫn bị pha loãng và biến dạng.
3. **Rào cản tính toán song song (Sequential Execution Barrier):**
   - Bước thời gian \(t\) không thể tính toán nếu chưa có kết quả của bước \(t-1\). Do đó, mạng hồi quy không thể tận dụng tối đa kiến trúc tính toán song song hàng nghìn nhân của các GPU hiện đại.

### 7.4. Động lực trực tiếp mở đường sang Attention (LAB 06) và Transformer (LAB 07)
Từ hai hạn chế trên, câu hỏi tự nhiên được đặt ra là:
> *Có cách nào để một token ở cuối câu có thể trực tiếp "nhìn" hoặc truy cập trực tiếp tới các token quan trọng ở đầu câu mà không cần phải truyền qua toàn bộ chuỗi mắt xích hồi quy dài đằng đẵng hay không?*

Câu trả lời chính là **Cơ chế Chú ý (Attention Mechanism)** trong **LAB 06**:
- Thay vì nén mọi thứ vào một vector \(h_T\), Attention cho phép mô hình lưu giữ toàn bộ các trạng thái ẩn \([h_1, h_2, \dots, h_T]\).
- Tại mỗi bước dự đoán, mô hình sử dụng bộ ba **Query — Key — Value** để tính trọng số tương quan trực tiếp với mọi vị trí, đưa khoảng cách truyền thông tin giữa hai token bất kỳ từ \(\mathcal{O}(T)\) về trực tiếp \(\mathcal{O}(1)\).
- Đây chính là nền tảng cốt lõi đưa ngành NLP tiến thẳng tới kiến trúc **Transformer (LAB 07)** — loại bỏ hoàn toàn tính hồi quy (recurrency) để đạt khả năng song song hóa tuyệt đối và khả năng hiểu ngữ cảnh vượt bậc trong các mô hình ngôn ngữ lớn (LLM) ngày nay.

---
*Báo cáo phản ánh này khép lại toàn bộ chuỗi kiến thức từ biểu diễn tĩnh (LAB 01–04) tới mô hình chuỗi (LAB 05), chuẩn bị nền tảng hoàn hảo cho cơ chế Attention trong LAB 06.*
