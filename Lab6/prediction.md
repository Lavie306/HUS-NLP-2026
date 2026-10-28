# LAB 06 — Prediction & Verification (Dự đoán trước thí nghiệm và Đối chiếu)

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — Năm học 2026  
**Sinh viên thực hiện:** Nguyễn Trọng Thành  

Tài liệu này ghi lại các dự đoán giả thuyết (hypotheses) được xây dựng dựa trên nguyên lý toán học trước khi chạy thí nghiệm, kèm theo phần đối chiếu thực nghiệm chi tiết sau khi chạy mã nguồn trong `attention_numpy.py` và `run_lab6.py`.

---

## Prediction 1 — Độ lớn của Attention Score và Trọng số Softmax

**Câu hỏi:** Nếu một key $k_i$ có tích vô hướng (dot-product) lớn hơn hẳn các key khác, attention weight của nó sẽ như thế nào?

- **Prediction (Dự đoán):** Key đó sẽ nhận attention weight lớn nhất trong toàn bộ chuỗi. Đồng thời, tỷ lệ chênh lệch về trọng số xác suất (attention weight) sau softmax sẽ lớn hơn rất nhiều so với tỷ lệ chênh lệch điểm số (score) ban đầu.
- **Reason (Lý do):** Hàm Softmax chuẩn hóa điểm số thông qua hàm mũ cơ số tự nhiên $f(s) = \exp(s)$. Vì hàm mũ là hàm đồng biến nghiêm ngặt ($\frac{d}{dx} e^x > 0$), thứ tự ưu tiên của các điểm số được bảo toàn tuyệt đối ($s_i > s_j \iff \alpha_i > \alpha_j$). Hơn nữa, do hàm mũ có tính lồi và tốc độ tăng siêu tuyến tính, khoảng cách giữa các điểm số bị khuếch đại theo cấp số nhân: nếu hai điểm số hơn kém nhau $\Delta s = 1.0$ đơn vị, hàm mũ của chúng sẽ hơn kém nhau gấp $e^1 \approx 2.718$ lần.
- **Confidence (Mức độ tự tin):** 95%
- **Đối chiếu thực nghiệm (Post-experiment Verification):**
  - **Khẳng định đúng 100%.**
  - Trong **Bài 2**, khi score vector là $s = [1, 2, 3]$ (khoảng cách đều $1.0$ đơn vị), trọng số sau softmax lần lượt là $\alpha = [0.0900, 0.2447, 0.6652]$. Trọng số của phần tử lớn nhất chiếm tới $66.52\%$, gấp gần $2.72$ lần phần tử thứ hai ($24.47\%$) và gấp hơn $7.39$ lần phần tử nhỏ nhất ($9.00\%$).
  - Trong **Experiment 1**, từ `"good"` có score cao nhất ($0.90$) và nhận attention weight cao nhất áp đảo so với các từ phụ vị ngữ khác.

---

## Prediction 2 — Các điểm số Attention hoàn toàn đồng nhất

**Câu hỏi:** Nếu tất cả các attention scores đều bằng nhau, ví dụ $s = [1, 1, 1, 1]$, các attention weights sẽ phân phối như thế nào?

- **Prediction (Dự đoán):** Các trọng số attention sẽ phân phối đều tuyệt đối (uniform distribution), mỗi phần tử nhận đúng $1/4 = 0.25$ ($25\%$).
- **Reason (Lý do):** Khi $s_1 = s_2 = \dots = s_n = C$, ta có $\exp(s_i) = e^C$ với mọi $i$. Tổng mẫu số là $\sum_{j=1}^n \exp(s_j) = n \cdot e^C$. Khi đó:
  $$\alpha_i = \frac{e^C}{n \cdot e^C} = \frac{1}{n}$$
  Giá trị tuyệt đối của các điểm số không quyết định phân phối; chỉ có độ lệch tương đối giữa các scores mới tạo ra tính chọn lọc. Trong trường hợp này, vector ngữ cảnh $c = \sum_i \frac{1}{n} v_i = \frac{1}{n} \sum_i v_i$ thoái hóa thành phép lấy trung bình cộng đơn giản (simple average pooling), cơ chế attention mất hoàn toàn khả năng chọn lọc thông tin.
- **Confidence (Mức độ tự tin):** 99%
- **Đối chiếu thực nghiệm (Post-experiment Verification):**
  - **Khẳng định đúng 100%.**
  - Đây chính là nền tảng toán học của thuật toán **Stable Softmax**: việc trừ đi $\max(x)$ tương đương với việc trừ đi cùng một hằng số $C$, không làm thay đổi phân phối xác suất đầu ra.
  - Trong thí nghiệm Temperature, khi $T \to +\infty$, các tỷ số $s_i / T \to 0$ (tiến về bằng nhau), entropy tiến dần về $\ln(3) \approx 1.0986$ nat, ứng với phân phối đều.

---

## Prediction 3 — Xuất hiện điểm số vượt trội cực đoan (Outlier Score)

**Câu hỏi:** Nếu có một score lớn vượt trội so với các scores còn lại, ví dụ $s = [1, 1, 10]$, attention sẽ tập trung như thế nào?

- **Prediction (Dự đoán):** Trọng số attention sẽ tập trung gần như toàn bộ ($> 99.9\%$) vào phần tử thứ ba, hai phần tử đầu chỉ nhận phần dư không đáng kể ($< 0.05\%$). Phân phối thoái hóa thành phép chọn lọc cứng (hard selection / argmax).
- **Reason (Lý do):** Thành phần $\exp(10) \approx 22026.47$, trong khi $\exp(1) \approx 2.72$. Tỷ lệ chênh lệch là:
  $$\frac{\exp(10)}{\exp(1)} = e^9 \approx 8103.08 \text{ lần}$$
  Khi cộng lại, tổng mẫu số xấp xỉ $22031.9$. Phần tử thứ ba chiếm $\frac{22026.47}{22031.9} \approx 99.975\%$, còn hai phần tử đầu mỗi phần tử chỉ chiếm $\approx 0.012\%$. Tình huống này đẩy softmax vào vùng bão hòa, làm đạo hàm cục bộ triệt tiêu về 0, gây nguy cơ nghẽn học (gradient vanishing) trong quá trình lan truyền ngược.
- **Confidence (Mức độ tự tin):** 95%
- **Đối chiếu thực nghiệm (Post-experiment Verification):**
  - **Khẳng định đúng 100%.**
  - Trong thí nghiệm Temperature, khi hạ $T = 0.5$ với vector score $s = [4, 2, 0]$ (tương đương vector tỷ số $[8, 4, 0]$), trọng số lớn nhất ngay lập tức đạt $98.17\%$, đẩy entropy xuống chỉ còn $0.093$ nat.
  - Khi không áp dụng scaling $\sqrt{d_k}$ với số chiều $d_k$ lớn (như $d_k = 1024$), dot-product có thể lên tới hàng chục, khiến softmax bị đông cứng tại một token duy nhất.

---

## Prediction 4 — Chi phí tính toán khi chiều dài chuỗi tăng mạnh

**Câu hỏi:** Nếu chiều dài chuỗi tăng từ 10 tokens lên 1,000 tokens, chi phí tính toán và bộ nhớ của Self-Attention sẽ thay đổi như thế nào?

- **Prediction (Dự đoán):** Số lượng phần tử của ma trận Attention $A \in \mathbb{R}^{T \times T}$ sẽ tăng chính xác $10,000$ lần (từ $10^2 = 100$ lên $1000^2 = 1,000,000$ phần tử). Thời gian tính toán trên CPU/GPU sẽ tăng theo độ phức tạp bậc hai $O(T^2)$.
- **Reason (Lý do):** Trong Self-Attention, mỗi token đóng vai trò là một Query cần phải tính tích vô hướng với tất cả $T$ Keys trong chuỗi. Do đó, ma trận scores và ma trận weights đều có kích thước $T \times T$. Chi phí nhân ma trận $Q K^T$ là $O(T^2 \cdot d_k)$, và bộ nhớ lưu trữ ma trận attention weights tỷ lệ thuận với $T^2$.
- **Confidence (Mức độ tự tin):** 90%
- **Đối chiếu thực nghiệm (Post-experiment Verification):**
  - **Khẳng định đúng 100% về mặt lý thuyết ma trận.**
  - Thực tế đo đạc trong `run_lab6.py` và bảng chi phí cho thấy:
    - $T = 10 \implies 100$ phần tử ($400$ Bytes).
    - $T = 1,000 \implies 1,000,000$ phần tử ($4$ MB cho một ma trận đơn lẻ).
    - $T = 10,000 \implies 100,000,000$ phần tử ($400$ MB cho một head đơn lẻ; khi nhân với Batch 64 và 8 Heads, tổng dung lượng vượt $200$ GB, gây tràn bộ nhớ Out-Of-Memory).
  - Về thời gian thực thi: ở các giá trị $T$ nhỏ ($T \le 100$), thời gian chạy bị chi phối bởi overhead khởi tạo thư viện; nhưng từ $T \ge 500$ trở đi, thời gian thực thi tăng bám sát đường cong parabol bậc hai $O(T^2)$.

---

## Prediction 5 — Tính chuẩn xác và khả năng "lý luận" (Reasoning) của Attention

**Câu hỏi:** Trọng số Attention có luôn tập trung vào đúng những tokens mà con người cho là quan trọng nhất về mặt ngôn ngữ học không?

- **Prediction (Dự đoán):** **Không luôn luôn.** Mặc dù attention thường có xu hướng gán trọng số cao hơn cho các từ khóa mang sắc thái cảm xúc hoặc thực từ (content words), mô hình hoàn toàn có thể tập trung vào các hư từ (stop words, dấu câu, từ nối) hoặc phân bổ trọng số lệch lạc.
- **Reason (Lý do):** 
  1. Trọng số Attention được tối ưu hóa gián tiếp thông qua hàm mất mát phân loại (Cross-Entropy Loss), chứ không hề có nhãn giám sát trực tiếp buộc mô hình phải chú ý vào từ nào.
  2. Mô hình có thể học các tương quan thống kê giả (spurious correlations) hoặc cơ chế thu thập ngữ cảnh gián tiếp (ví dụ: dùng một từ đệm làm "bộ nhớ tạm" gom thông tin).
  3. Giá trị vector ngữ cảnh cuối cùng còn phụ thuộc chặt chẽ vào ma trận Value $V$ và tầng phân loại (Linear Classifier) phía sau. Một từ có attention weight cao nhưng vector Value có độ lớn nhỏ thì đóng góp thực tế vào logit phân loại vẫn có thể rất thấp.
- **Confidence (Mức độ tự tin):** 85%
- **Đối chiếu thực nghiệm (Post-experiment Verification):**
  - **Khẳng định đúng 100%.**
  - Kết quả phân tích trong `error_analysis.md` chỉ ra rằng nhiều câu kiểm thử IMDb bị dự đoán sai dù mô hình gán attention cao vào các từ như `"in"`, `"and"`, `"make"`.
  - Đặc biệt, với câu tương phản: *"I thought this movie would be bad, but it was actually excellent."*, mô hình đơn giản không có Positional Encoding đã phân tán attention vào cả từ phủ định lẫn từ khen ngợi, dẫn tới dự đoán xác suất tích cực thấp ($P = 0.004$).
  - Điều này chứng minh luận điểm cốt lõi của bài Lab: **Attention Weight là bằng chứng về phân phối trọng số toán học, không phải là bằng chứng về khả năng hiểu ngôn ngữ hay suy luận (Reasoning) của mô hình.**
