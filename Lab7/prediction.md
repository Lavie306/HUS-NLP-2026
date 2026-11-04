# Báo cáo Dự đoán & Đối chiếu Thực nghiệm (Prediction & Verification) — LAB 07

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — Năm học 2026  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Chủ đề:** From Attention to Transformer  

---

## 1. Các Dự đoán Trước Thực nghiệm (Giai đoạn Prediction — Mục 8)

Các dự đoán dưới đây được ghi lại trước khi tiến hành chạy các mô hình và thực nghiệm trong `run_all_experiments.py` và `transformer_classification.ipynb`. Mỗi dự đoán đều đi kèm căn cứ lý thuyết và mức độ tin cậy ban đầu.

---

### Prediction 1
- **Câu hỏi:** *Một Transformer có thể xử lý các token trong sequence song song hơn RNN không?*
- **Prediction:** **Có (Hoàn toàn vượt trội về khả năng song song hóa).**
- **Reason:**  
  Mạng hồi quy (RNN/LSTM/GRU) có bản chất cấu trúc phụ thuộc tuần tự theo bước thời gian: trạng thái ẩn $h_t = f(h_{t-1}, x_t)$ bắt buộc phải chờ trạng thái $h_{t-1}$ tính xong mới có thể thực hiện. Điều này tạo ra nút thắt cổ chai tính toán tuần tự (sequential bottleneck), khiến phần cứng hiện đại (GPU đa nhân) bị lãng phí chu kỳ chờ đợi.  
  Ngược lại, Transformer loại bỏ hoàn toàn đệ quy tuần tự. Toàn bộ các phép chiếu Query, Key, Value cho mọi vị trí trong câu được tính đồng thời dưới dạng phép nhân ma trận khối $X W$. Tiếp đó, ma trận điểm chú ý $Q K^T$ giữa mọi cặp token được tính toán song song tức thì. Do đó, Transformer khai thác tối đa năng lực xử lý song song của GPU/CPU đa lõi.
- **Confidence:** **5/5 (Rất cao)**

---

### Prediction 2
- **Câu hỏi:** *Nếu bỏ Positional Encoding, Transformer có còn biết thứ tự token không?*
- **Prediction:** **Không (Mô hình hoàn toàn mất khả năng nhận biết thứ tự từ).**
- **Reason:**  
  Phép toán Self-Attention cốt lõi là một phép toán bất biến hoán vị (permutation-equivariant). Tích vô hướng giữa hai token $q_i \cdot k_j$ hoàn toàn chỉ phụ thuộc vào nội dung ngữ nghĩa của vector $x_i$ và $x_j$, chứ không phụ thuộc vào việc token $i$ đứng trước hay đứng sau token $j$. Nếu ta tráo đổi vị trí của các token trong chuỗi đầu vào, các hàng tương ứng của ma trận đầu ra sau Attention cũng chỉ bị tráo đổi vị trí tương ứng mà không thay đổi giá trị. Khi kết hợp với các phép gộp toàn cục như Mean Pooling hay Sum Pooling, biểu diễn thu được là giống nhau hoàn toàn giữa các chuỗi bị hoán vị. Vì vậy, thiếu Positional Encoding, Transformer sẽ suy biến thành một mô hình dạng Túi từ liên tục (Continuous Bag-of-Words).
- **Confidence:** **5/5 (Rất cao)**

---

### Prediction 3
- **Câu hỏi:** *Nếu hai câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information có thể phân biệt chúng không?*
- **Prediction:** **Không thể phân biệt được (Biểu diễn toàn câu của hai chuỗi sẽ trùng khít 100%).**
- **Reason:**  
  Xét hai câu đảo ngữ kinh điển: $S_1 = \text{"dog bites man"}$ (Chó cắn người) và $S_2 = \text{"man bites dog"}$ (Người cắn chó). Hai câu chứa chính xác cùng một tập hợp từ vựng $\{\text{"dog"}, \text{"bites"}, \text{"man"}\}$.  
  Nếu không có Positional Encoding cộng vào embedding, ma trận đầu vào $X_{S_2}$ thực chất chỉ là một phép hoán vị dòng (row permutation matrix $P$) của $X_{S_1}$: $X_{S_2} = P X_{S_1}$.  
  Do Self-Attention thỏa mãn tính chất hoán vị: $\text{Attention}(P X) = P \cdot \text{Attention}(X)$, tập hợp các vector ngữ cảnh tại các từ là đồng nhất, chỉ đảo thứ tự dòng. Khi áp dụng Masked Mean Pooling để lấy vector biểu diễn đại diện phân loại câu:
  $$h_{mean}(S_2) = \frac{1}{3} \sum_{t=1}^3 (P Z)_{t} = \frac{1}{3} \sum_{t=1}^3 Z_t = h_{mean}(S_1)$$
  Vector đại diện của $S_1$ và $S_2$ hoàn toàn bằng nhau, dẫn tới khoảng cách Euclid bằng $0$ và độ tương đồng Cosine bằng $1.0$. Mô hình bất lực trong việc phân biệt hai câu mang ý nghĩa trái ngược này.
- **Confidence:** **5/5 (Tuyệt đối)**

---

### Prediction 4
- **Câu hỏi:** *Hai attention heads có nhất thiết học cùng một attention pattern không?*
- **Prediction:** **Không nhất thiết (và trên thực tế chúng được khuyến khích học các pattern phân kỳ, đa dạng).**
- **Reason:**  
  Mỗi head $i$ sở hữu bộ tham số chiếu độc lập $(W_i^Q, W_i^K, W_i^V)$, được khởi tạo ngẫu nhiên theo các phân phối trọng số khác nhau. Trong quá trình tối ưu hóa qua hàm mất mát và lan truyền ngược, ma trận chiếu đầu ra $W^O$ kết hợp thông tin từ tất cả các head. Để giảm thiểu hàm mất mát tổng thể, mạng sẽ tự động phân công nhiệm vụ chuyên biệt cho từng head để khai thác nhiều khía cạnh bổ trợ: một head có thể học mối liên kết cú pháp gần (local window), một head chú ý vào các từ cảm xúc chủ đạo, một head chú ý vào các từ nối tương phản (như "but", "however"). Tuy nhiên, vẫn có khả năng một số head bị dư thừa (redundant) nếu dung lượng mô hình lớn hơn mức cần thiết của tác vụ.
- **Confidence:** **4/5 (Cao)**

---

### Prediction 5
- **Câu hỏi:** *Tăng số attention heads có chắc chắn làm accuracy tăng không?*
- **Prediction:** **Không chắc chắn (Hiệu năng sẽ bão hòa và có thể suy giảm nếu số head tăng quá mức).**
- **Reason:**  
  Trong kiến trúc Transformer chuẩn, tổng số chiều $d_{model}$ được giữ cố định để kiểm soát tổng lượng tham số và chi phí tính toán. Khi tăng số heads $n_{heads}$, số chiều của mỗi không gian con bắt buộc phải thu hẹp lại:
  $$d_k = \frac{d_{model}}{n_{heads}}$$
  Nếu tăng số heads quá mức (ví dụ $d_{model} = 128$ mà chọn $n_{heads} = 8$ hoặc $16$), số chiều mỗi head chỉ còn $d_k = 16$ hoặc $8$. Khi không gian con quá hẹp, tích vô hướng $\frac{q \cdot k}{\sqrt{d_k}}$ bị hạn chế năng lực biểu diễn, các head không đủ dung lượng để mô hình hóa các mối quan hệ ngữ nghĩa phức tạp. Hơn nữa, việc phân mảnh thành quá nhiều head nhỏ có thể làm phân tán tín hiệu gradient và khiến mô hình khó hội tụ, đặc biệt trên tập dữ liệu kích thước nhỏ hoặc vừa.
- **Confidence:** **4/5 (Cao)**

---

### Prediction 6
- **Câu hỏi:** *Transformer có luôn nhanh hơn RNN trên mọi sequence length không?*
- **Prediction:** **Không (Transformer chỉ chiếm ưu thế tốc độ ở chuỗi có độ dài ngắn và trung bình; ở chuỗi cực dài, Transformer sẽ chậm hơn đáng kể so với RNN).**
- **Reason:**  
  Độ phức tạp tính toán và bộ nhớ của hai kiến trúc có sự khác biệt cơ bản:
  - **RNN:** Có độ phức tạp tuyến tính $O(T \cdot d^2)$ theo độ dài chuỗi $T$. Bộ nhớ trạng thái ẩn chỉ là $O(d)$.
  - **Transformer:** Cơ chế Self-Attention tính toán ma trận tương tác giữa tất cả các cặp vị trí, dẫn tới độ phức tạp tính toán và bộ nhớ bậc hai $O(T^2 \cdot d)$.  
  Mặc dù Transformer có lợi thế tính toán song song trên GPU giúp thời gian trên mỗi batch nhanh hơn ở độ dài thông thường ($T \le 512$), nhưng khi độ dài chuỗi tăng vọt lên hàng nghìn hoặc chục nghìn tokens ($T \ge 4096, 16384$), chi phí bộ nhớ bậc hai $T^2$ bùng nổ khủng khiếp (ví dụ $10000^2 = 10^8$ phần tử cho mỗi head), gây nghẽn băng thông bộ nhớ (memory-bound) và tràn VRAM (Out-Of-Memory). Lúc này, chi phí của Transformer lớn hơn rất nhiều so với bước chuyển tuần tự đơn giản của RNN.
- **Confidence:** **5/5 (Rất cao)**

---

## 2. Đối chiếu & Kiểm chứng Thực nghiệm (Prediction Verification — Mục 28)

Sau khi hoàn thành toàn bộ chuỗi thực nghiệm trên tập dữ liệu chuẩn IMDb trong `run_all_experiments.py`, bảng dưới đây tổng hợp kết quả đối chiếu giữa dự đoán ban đầu và bằng chứng thực nghiệm thu được.

### Bảng Kiểm chứng Kết quả Dự đoán:

| Dự đoán (Prediction) | Nội dung Dự đoán Ban đầu | Kết quả Thực nghiệm Định lượng | Đánh giá | Diễn giải & Phân tích Chi tiết |
|:---:|:---|:---|:---:|:---|
| **1** | Transformer xử lý song song và nhanh hơn RNN | Trên cùng tập dữ liệu IMDb (10,000 mẫu, 2 epochs), Transformer tận dụng tối đa tính toán tensor ma trận song song. Tuy nhiên, tổng thời gian huấn luyện trên CPU của Transformer (158.59s) cao hơn LSTM (20.25s) do khối lượng phép tính FLOPs lớn hơn nhiều (Self-Attention + FFN 2 tầng + LayerNorm). | **Đúng một phần (Partially Correct)** | *Cần phân biệt rạch ròi giữa "khả năng song song hóa (parallelizability)" và "tổng thời gian thực thi (wall-clock time)".* Transformer song song hóa triệt để cấu trúc tính toán (không có vòng lặp thời gian), nhưng chứa nhiều phép biến đổi ma trận hơn và có thêm khối FFN. Trên GPU, tính song song giúp Transformer bứt phá; trên CPU lõi đơn/hạn chế, khối lượng tính toán lớn khiến thời gian chạy tổng thể cao hơn. |
| **2** | Bỏ Positional Encoding thì Transformer không nhận biết được thứ tự từ | Thử nghiệm 1 và Permutation Test xác nhận: khi bỏ PE, đảo thứ tự câu không làm thay đổi phân phối giá trị của context vector sau Mean Pooling. | **Chính xác (Correct)** | Thực nghiệm chứng minh tính bất biến hoán vị của Self-Attention. Thiếu PE, mô hình suy biến thành Bag-of-Words và mất hoàn toàn ngữ pháp thứ tự. |
| **3** | Không phân biệt được "dog bites man" và "man bites dog" nếu thiếu PE | Khoảng cách Euclid giữa hai câu khi không có PE là **$0.000000\times 10^0$**, Cosine Similarity là **$1.000000$** (trùng nhau 100%). Khi có PE, khoảng cách tăng lên **$0.2216$**, Cosine Similarity giảm xuống **$0.9942$** (tách biệt rõ ràng). | **Chính xác (Correct)** | Hoàn toàn khớp với dự đoán lý thuyết. Dữ liệu thực nghiệm chứng minh PE là điều kiện tiên quyết bắt buộc để phân biệt các câu hoán vị từ. |
| **4** | Các Attention Heads học các mẫu chú ý khác nhau | Trực quan hóa 4 Attention Heads trên câu mẫu cho thấy: Head 1 tập trung vào token lân cận, Head 2 tập trung vào từ cảm xúc ("fantastic", "charming"), Head 3 bắt trúng liên từ chuyển ngoặt ("but"), Head 4 phân bố dàn trải toàn cục. | **Chính xác (Correct)** | Các head thực sự phân hóa đa dạng và nắm bắt các khía cạnh cú pháp/ngữ nghĩa độc lập đúng như thiết kế kỳ vọng. |
| **5** | Tăng số Heads không chắc chắn làm tăng Accuracy/F1 | Kết quả khảo sát $n_{heads} \in \{1, 2, 4, 8\}$: $n_{heads} = 4$ đạt F1 cao nhất ($0.7298$, Acc $0.7240$). Khi tăng lên $n_{heads} = 8$, $d_k$ giảm xuống $16$, F1 giảm xuống $0.6871$ (Acc $0.7184$). | **Chính xác (Correct)** | Tồn tại điểm tối ưu về số lượng head. Khi $d_k$ quá nhỏ, sức chứa biểu diễn của từng head bị suy giảm gây tổn hại hiệu năng tổng thể. |
| **6** | Transformer không luôn nhanh hơn RNN ở mọi độ dài chuỗi | Khảo sát độ dài chuỗi $T \in \{32, 64, 128, 256\}$ cho thấy số phần tử ma trận attention tăng bậc hai $T^2$: từ $1,024 \to 65,536$ ($64\times$). Thời gian tính toán trên batch tăng vọt từ $0.0995\text{s} \to 1.3695\text{s}$ (gấp $13.7\times$). | **Chính xác (Correct)** | Chi phí $O(T^2)$ là rào cản nền tảng khiến Transformer truyền thống không thể tùy tiện tăng độ dài chuỗi mà không gặp nghẽn tài nguyên. |

---

## 3. Bài học Rút ra từ Sự sai lệch Dự đoán 1
Ở Prediction 1, nhận định *"Transformer tính toán song song hơn RNN"* là hoàn toàn chính xác về mặt lý thuyết kiến trúc mạng. Tuy nhiên, việc suy diễn rằng *"tính toán song song hơn đồng nghĩa với việc thời gian huấn luyện đo bằng giây (wall-clock training time) luôn nhanh hơn"* là một sai lầm phổ biến khi bỏ qua:
1. **Tổng lượng phép tính (Total FLOPs):**  
   Một khối Transformer Encoder gồm rất nhiều tầng con: 4 phép chiếu tuyến tính trong MHA ($W^Q, W^K, W^V, W^O$), phép tính Softmax kích thước $T \times T$, 2 tầng tuyến tính trong FFN ($d_{model} \to d_{ff} \to d_{model}$ với $d_{ff} = 2 \times d_{model}$ hoặc $4 \times d_{model}$), cùng 2 tầng LayerNorm. Tổng số phép nhân ma trận trên mỗi token lớn hơn đáng kể so với một tế bào LSTM đơn giản.
2. **Môi trường phần cứng thực thi (Hardware Constraints):**  
   Tính ưu việt về tốc độ của Transformer chỉ bộc lộ rõ nét nhất trên các hệ thống có năng lực tính toán ma trận song song ồ ạt như GPU có hàng nghìn nhân CUDA hoặc Tensor Cores. Trên CPU với số luồng hữu hạn, ưu thế song song bị hạn chế trong khi gánh nặng FLOPs lại chiếm ưu thế, dẫn tới thời gian huấn luyện trên CPU lâu hơn.
