# LAB 06 — Nhận xét, Phân tích & Suy ngẫm (Reflection)

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — Năm học 2026  
**Trường:** ĐH Khoa học Tự nhiên — ĐHQGHN (VNU-HUS)  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Sinh viên thực hiện:** Nguyễn Trọng Thành  

---

## 1. Mạch chuyển dịch tư duy: Từ Recurrent State (LAB 05) sang Context-Aware Representation (LAB 06)

Ở **LAB 05**, chúng ta đã tiếp cận bài toán mô hình hóa ngôn ngữ dựa trên giả định tuần tự (sequential processing). Mô hình mạng hồi quy (RNN, LSTM, GRU) xử lý từng token tại từng bước thời gian:
$$h_t = \text{Cell}(h_{t-1}, x_t)$$

Cách tiếp cận này đặt ra câu hỏi cốt lõi: **"Làm thế nào để duy trì thông tin qua một chuỗi dài?"**  
Để giải quyết sự suy giảm gradient (vanishing gradient), LSTM và GRU đưa vào các cổng (gates) để điều tiết luồng thông tin. Tuy nhiên, kiến trúc hồi quy vẫn vấp phải hai rào cản nền tảng:
1. **Điểm nghẽn biểu diễn (Information Bottleneck):** Mọi thông tin của toàn bộ văn bản bắt buộc phải nén vào trạng thái ẩn cuối cùng $h_T$. Khi $T$ lớn, tín hiệu từ các từ xuất hiện sớm bị phai nhạt dần trước các từ xuất hiện muộn (recency bias).
2. **Nút thắt tuần tự (Sequential Bottleneck):** Không thể tính bước $t$ khi chưa tính xong bước $t-1$. Điều này triệt tiêu hoàn toàn khả năng tận dụng sức mạnh tính toán song song quy mô lớn của phần cứng hiện đại (GPU/TPU).

**LAB 06** mang lại một bước ngoặt kiến trúc: thay vì ép chuỗi vào một trạng thái ẩn đơn lẻ, cơ chế **Attention** đặt ra câu hỏi: **"Làm thế nào để chọn lọc trực tiếp thông tin liên quan từ bất kỳ vị trí nào trong chuỗi?"**

Nhờ phép chiếu $Q, K, V$, mọi cặp token $(i, j)$ trong chuỗi đều có một đường kết nối trực tiếp (direct shortcut) với độ dài đường truyền bằng 1:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

---

## 2. Bảng so sánh toàn diện: RNN/LSTM/GRU vs. Self-Attention

Dựa trên lý thuyết và các số liệu đo lường thực tế giữa hai bài Lab:

| Tiêu chí so sánh | RNN / LSTM / GRU (LAB 05) | Self-Attention (LAB 06) | Ý nghĩa thực tiễn |
|:---|:---|:---|:---|
| **Mô hình hóa thứ tự chuỗi** | Tự nhiên có sẵn qua cấu trúc hồi quy ($t-1 \to t$) | **Không có sẵn**; bất biến với hoán vị nếu không có Positional Encoding | Self-Attention đơn giản không phân biệt được `"not good"` và `"good not"`. |
| **Trạng thái ẩn (Hidden State)** | Bắt buộc phải duy trì qua các bước $h_1, \dots, h_T$ | Không bắt buộc; biểu diễn được tổng hợp linh hoạt qua weighted sum | Tránh được việc phải "ghi nhớ" toàn bộ chuỗi trong một vector đơn lẻ. |
| **Tính toán tuần tự (Sequential steps)** | Bắt buộc $O(T)$ bước tuần tự | **$O(1)$ bước tuần tự**; tính toán song song đồng thời trên toàn bộ ma trận | Cho phép tăng tốc độ huấn luyện vượt trội trên các hệ thống phần cứng. |
| **Tương tác trực tiếp giữa hai vị trí bất kỳ** | Hạn chế; phải truyền qua $O(\|i - j\|)$ bước trung gian | **Trực tiếp $O(1)$** cho mọi cặp $(i, j)$ | Xử lý hoàn hảo phụ thuộc xa (long-range dependencies). |
| **Khả năng song song hóa (Parallelism)** | Rất hạn chế (bị chặn bởi phụ thuộc thời gian) | **Rất cao** (phép nhân ma trận tiêu chuẩn trên GPU) | Thời gian huấn luyện giảm mạnh (3.42s vs 27.93s). |
| **Độ phức tạp tính toán theo độ dài chuỗi $T$** | $O(T \cdot d^2)$ (tuyến tính theo độ dài chuỗi) | **$O(T^2 \cdot d)$** (bậc hai theo độ dài chuỗi) | Self-Attention gặp bùng nổ chi phí tính toán khi văn bản quá dài. |
| **Dung lượng bộ nhớ lưu trữ activations** | $O(T \cdot d)$ | **$O(T^2)$** (lưu trữ ma trận attention $T \times T$) | Nguy cơ tràn bộ nhớ (OOM) khi $T \ge 10,000$. |
| **Biểu diễn ngữ cảnh (Context)** | Vector hồi quy $h_T$ bị suy giảm | Tổ hợp tuyến tính linh hoạt $c_i = \sum_j \alpha_{ij} v_j$ | Cho phép truy vấn ngữ cảnh riêng biệt cho từng token. |

---

## 3. Phân tích kết quả thực nghiệm định lượng trên IMDb

Bảng đối chiếu trích xuất từ `results.csv` trên cùng tập kiểm thử IMDb (1,000 mẫu cân bằng, độ dài cắt ngắn $100$ tokens, 2 epochs huấn luyện):

| Mô hình | Kiểu biểu diễn ngữ cảnh | Test Accuracy | Test F1 | Thời gian huấn luyện (s) | Số lượng tham số |
|:---|:---|:---:|:---:|:---:|:---:|
| **LSTM (LAB 05)** | Final hidden state $h_T$ | $0.5530$ | $0.5406$ | $27.93$ giây | $801,345$ |
| **Self-Attention (LAB 06)** | Weighted Context + Mean Pooling | **$0.6650$** | **$0.6660$** | **$3.42$ giây** | **$780,353$** |

### Các quan sát then chốt:
1. **Độ chính xác và F1 vượt trội:** Mô hình Self-Attention đạt F1 là $0.6660$, cao hơn LSTM ($0.5406$) tới $12.5\%$. Lý do là vì với câu đánh giá phim dài tới 100 từ, việc lấy trực tiếp trung bình có trọng số qua Attention cho phép mô hình thu thập các tín hiệu cảm xúc nằm rải rác ở khắp các vị trí trong câu mà không bị pha loãng như khi nén vào $h_T$.
2. **Tốc độ huấn luyện siêu việt:** Mặc dù số tham số tương đương nhau (~780K vs ~801K), Self-Attention chỉ mất **$3.42$ giây** để hoàn thành 2 epochs, nhanh hơn LSTM gấp **$8.1$ lần** ($27.93$ giây) ngay trên môi trường CPU. Điều này minh chứng cho ưu thế triệt tiêu vòng lặp thời gian tuần tự của Attention.
3. **So sánh với Baseline tĩnh (TF-IDF + LR):** Cần lưu ý rằng mô hình TF-IDF + Logistic Regression ở Lab 05 vẫn đạt Accuracy $0.854$. Điều này cho thấy với tập dữ liệu nhỏ (5,000 mẫu train) và số epoch hạn chế (2 epochs), việc học các ma trận chiếu $W_Q, W_K, W_V$ cùng embedding từ đầu vẫn cần thêm thời gian hội tụ và dữ liệu lớn hơn để phát huy tối đa tiềm năng.

---

## 4. "Attention không miễn phí": Những giới hạn cố hữu của Attention

Một quan niệm sai lầm phổ biến là: *"Attention tốt hơn RNN nên Attention luôn vượt trội và thay thế hoàn toàn RNN trong mọi tình huống."*  
Thực tế, Attention giải quyết được bài toán phụ thuộc xa và song song hóa, nhưng lại tạo ra **những vấn đề mới**:

### 4.1. Chi phí bậc hai $O(T^2)$ và bài toán tài liệu dài
- Ma trận Attention giữa $T$ tokens đòi hỏi tính và lưu trữ $T^2$ phần tử.
- Với $T = 100$, ma trận chỉ có $10,000$ phần tử ($40$ KB).
- Với $T = 10,000$, ma trận phình to thành $100,000,000$ phần tử ($400$ MB cho 1 head của 1 mẫu).
- Trong các mô hình Transformer hiện đại (Batch=64, Heads=32, Layers=32), việc xử lý văn bản dài hàng trăm ngàn tokens đòi hỏi hàng trăm GB VRAM chỉ riêng cho ma trận attention. Trong khi đó, các mô hình hồi quy cải tiến (như RWKV, Mamba, State Space Models) đang quay trở lại nhờ chi phí tuyến tính $O(T)$.

### 4.2. Mất mát thông tin thứ tự (Lack of Inductive Bias on Order)
- Trong công thức Self-Attention thuần túy:
  $$\text{scores}_{ij} = \frac{(x_i W_Q)(x_j W_K)^T}{\sqrt{d_k}}$$
  Nếu ta tráo đổi vị trí của hai token trong câu (ví dụ: `"good not"` thay vì `"not good"`), tập hợp các vector Key và Query vẫn giữ nguyên. Nếu không có cơ chế bổ sung thông tin vị trí (**Positional Encoding**), mô hình Attention hoàn toàn **bất biến với thứ tự từ (permutation invariant)**. Đây là một điểm yếu nghiêm trọng so với RNN vốn có cơ chế thứ tự bẩm sinh.

---

## 5. Ranh giới nhận thức: Trọng số Attention KHÔNG PHẢI là "Khả năng suy luận" (Attention is Not Explanation)

Trong cộng đồng nghiên cứu NLP, việc trực quan hóa heatmap attention thường dễ dẫn tới sự ngộ nhận rằng mô hình "đã hiểu ngữ pháp" hoặc "đang suy luận logic như con người". 

Thí nghiệm trong Lab 06 làm rõ hai nguyên lý quan trọng:
1. **Attention là phân phối trọng số toán học:** Nó chỉ phản ánh mức độ đóng góp số học của các vector Value vào vector Context trong không gian ẩn để cực tiểu hóa hàm mất mát (loss).
2. **Không đồng nhất tương quan với nhân quả:** Việc từ `"sleeping"` có attention cao vào từ `"cat"` trong câu ví dụ của Experiment 3 là do các vector đặc trưng được thiết kế có tích vô hướng cao. Trong các mô hình học sâu thực tế, một token nhận attention cao có thể chỉ đóng vai trò là một "bộ đệm trung gian" (information sink), hoặc mô hình học theo tương quan giả trong tập huấn luyện.
3. Do đó, khi báo cáo và phân tích lỗi, chúng ta chỉ được phép đưa ra các **giả thuyết (hypotheses)** dựa trên phân phối trọng số, tuyệt đối không được khẳng định rằng mô hình có "tư duy" hay "hiểu" ngữ pháp.

---

## 6. Bước đệm dẫn tới LAB 07: Từ Attention Layer tới Transformer Architecture

LAB 06 đã giúp chúng ta giải phẫu cơ chế Attention từ gốc rễ: tính toán bằng tay, lập trình từ đầu bằng NumPy, kiểm soát tính ổn định số học và tích hợp một tầng Attention đơn giản vào PyTorch.

Tuy nhiên, để xây dựng một kiến trúc hoàn chỉnh có thể thay thế hoàn toàn mạng hồi quy, chúng ta cần giải quyết các bài toán còn dang dở:
1. **Làm thế nào để mô hình chú ý vào nhiều khía cạnh khác nhau cùng lúc?**  
   $\implies$ Giải pháp: **Multi-Head Attention** (sẽ học ở LAB 07), chia không gian biểu diễn thành nhiều không gian con độc lập (ví dụ: 1 head chú ý cú pháp, 1 head chú ý ngữ nghĩa, 1 head chú ý chủ ngữ - vị ngữ).
2. **Làm thế nào để bù đắp sự thiếu hụt thứ tự từ?**  
   $\implies$ Giải pháp: **Positional Encoding** (Sinusoidal hoặc Learnable Embeddings) cộng trực tiếp vào Word Embeddings trước khi đi vào Attention.
3. **Làm thế nào để huấn luyện mạng sâu hàng chục tầng mà không bị suy giảm thông tin?**  
   $\implies$ Giải pháp: **Transformer Block** kết hợp Residual Connections (Skip-connections), Layer Normalization và Position-wise Feed-Forward Networks (FFN).

Những nền móng vững chắc của LAB 06 chính là chiếc chìa khóa để bước vào thế giới của kiến trúc Transformer và các Mô hình ngôn ngữ lớn (LLMs) trong LAB 07.
