# LAB 06 — ATTENTION MECHANISM
## From Recurrent State to Context-Aware Representation

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — Năm học 2026  
**Trường:** ĐH Khoa học Tự nhiên — ĐHQGHN (VNU-HUS)  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Sinh viên thực hiện:** Nguyễn Trọng Thành  
**Hạn nộp:** 23:59 — 28/10/2026  

---

## 1. Giới thiệu và Bối cảnh (Context)

Ở **LAB 05**, chúng ta đã chuyển đổi từ biểu diễn tĩnh (Static Representations như TF-IDF, Avg Word2Vec) sang mô hình hóa chuỗi (Sequential Modeling) bằng mạng hồi quy (RNN, LSTM, GRU). Chuỗi đầu vào được xử lý tuần tự qua từng bước thời gian và nén toàn bộ ngữ cảnh vào trạng thái ẩn cuối cùng:
$$\text{Text} \longrightarrow \text{Tokens} \longrightarrow \text{Embedding} \longrightarrow \text{RNN / LSTM / GRU} \longrightarrow h_T \longrightarrow \text{Classifier}$$

Tuy nhiên, cấu trúc hồi quy tuần tự bộc lộ hai rào cản nền tảng:
1. **Điểm nghẽn thông tin (Information Bottleneck):** Khi độ dài chuỗi $T$ lớn, toàn bộ thông tin ngữ cảnh buộc phải bị nén vào một vector cố định duy nhất $h_T$. Tín hiệu từ các từ xuất hiện sớm bị phai nhạt dần trước các từ xuất hiện muộn (recency bias), khiến việc nắm bắt các phụ thuộc xa (long-range dependencies) gặp nhiều khó khăn.
2. **Nút thắt tuần tự (Sequential Bottleneck):** Không thể tính toán bước $t$ khi chưa hoàn thành bước $t-1$. Điều này triệt tiêu hoàn toàn khả năng tận dụng tính toán song song quy mô lớn của phần cứng hiện đại (GPU/TPU).

**LAB 06** giới thiệu một cơ chế kiến trúc đột phá: **Attention Mechanism**.
Thay vì ép toàn bộ sequence vào một vector đơn lẻ $h_T$, Attention cho phép mô hình kiểm tra toàn bộ các trạng thái $h_1, h_2, \dots, h_T$ và tự động gán trọng số tập trung vào những phần của chuỗi liên quan nhất tới truy vấn đang cần:

$$\begin{matrix}
\text{Query } (Q) \\
\downarrow \\
\text{So khớp với Keys } (K) \longrightarrow \text{Attention Scores } (S) \\
\downarrow \\
\text{Chuẩn hóa bằng Softmax} \longrightarrow \text{Attention Weights } (A) \\
\downarrow \\
\text{Tổng có trọng số Values } (V) \longrightarrow \text{Context Vector } (C)
\end{matrix}$$

---

## 2. Mục tiêu học tập (Learning Outcomes)

- [x] **LO1:** Giải thích motivation của attention và giải quyết điểm nghẽn của recurrent state $h_T$.
- [x] **LO2:** Tính attention score giữa query và key bằng tích vô hướng (dot-product).
- [x] **LO3:** Tính attention weights bằng hàm Softmax và chứng minh tính chất $\sum \alpha_i = 1$.
- [x] **LO4:** Tính context vector từ attention weights và values.
- [x] **LO5:** Phân biệt và giải thích vai trò của bộ ba Query, Key và Value.
- [x] **LO6:** Phân biệt dot-product attention và scaled dot-product attention (chia $\sqrt{d_k}$).
- [x] **LO7:** Tự lập trình Attention và Self-Attention hoàn toàn từ đầu bằng NumPy (`attention_numpy.py`).
- [x] **LO8:** Trực quan hóa ma trận Attention weights bằng Heatmap.
- [x] **LO9:** Phân tích ảnh hưởng của tham số nhiệt độ (temperature) và scaling tới độ tập trung của Softmax.
- [x] **LO10:** So sánh biểu diễn recurrent (LSTM) với biểu diễn dựa trên attention (Self-Attention) trên tập IMDb.
- [x] **LO11:** Nhận diện giới hạn chi phí tính toán $O(T^2)$ và sự thiếu vắng thứ tự từ khi không có Positional Encoding.
- [x] **LO12:** Hiểu attention như bước chuyển tiếp bản lề tới kiến trúc Transformer (LAB 07).

---

## 3. Cấu trúc thư mục & Tệp bàn giao (Deliverables)

Thư mục `Lab6/` chứa đầy đủ tất cả các tệp theo đúng quy định tại **Mục 32 (Deliverables)** của `W6.pdf`:

```text
Lab6/
├── README.md                      # Báo cáo tổng hợp, hướng dẫn chạy và AI statement
├── calculations.md                # Lời giải chi tiết các bài tập tính tay (Bài 1 -> Bài 4, scaling, temp, self-attn)
├── prediction.md                  # 5 dự đoán trước thí nghiệm kèm lý giải và đối chiếu thực nghiệm
├── attention_numpy.py             # Triển khai Attention từ đầu bằng NumPy (stable softmax, scaled attn, self-attn)
├── self_attention.ipynb           # Jupyter Notebook hoàn chỉnh, có thể chạy lại và lưu đầy đủ kết quả
├── results.csv                    # Bảng so sánh định lượng giữa LSTM (Lab 05) và Self-Attention (Lab 06)
├── attention_heatmap.png          # Heatmap ma trận Self-Attention minh họa câu "cat"
├── confusion_matrix.png           # Ma trận nhầm lẫn của mô hình Self-Attention trên tập test IMDb
├── error_analysis.md              # Phân tích chi tiết 3 mẫu dự đoán đúng và 3 mẫu dự đoán sai trên tập test
├── reflection.md                  # Nhận xét sâu sắc, đối chiếu với Lab 05 và cầu nối sang Transformer (Lab 07)
├── requirements.txt               # Danh sách thư viện Python cần thiết
├── experiment_details.json        # Bản ghi thông số và lịch sử huấn luyện chi tiết
├── attention_model.pt             # Trọng số mô hình PyTorch đã huấn luyện
├── query_attention.png            # Biểu đồ phân phối attention theo Query (Sentiment vs Topic)
├── temperature_experiment.png     # Biểu đồ khảo sát phân phối và entropy theo nhiệt độ T
└── sequence_length_cost.png       # Biểu đồ phân tích chi phí tính toán bậc hai O(T^2)
```

---

## 4. Hướng dẫn chạy và Tái lập kết quả (Reproduction Guide)

Từ thư mục gốc dự án (`d:\NLP`), kích hoạt môi trường Python 3.13:

```powershell
# 1. Cài đặt các thư viện cần thiết
pip install -r Lab6/requirements.txt

# 2. Chạy kiểm tra các bài tính và hàm NumPy cơ bản
python Lab6/attention_numpy.py

# 3. Mở Jupyter Notebook xem và chạy toàn bộ thí nghiệm & kết quả:
jupyter notebook Lab6/self_attention.ipynb
```

Sau khi hoàn tất, mở `Lab6/self_attention.ipynb` trong Jupyter Lab / VS Code để xem trực tiếp các bảng kết quả và biểu đồ.

---

## 5. Tóm tắt kết quả bài tập tính toán & Lý thuyết

1. **Bài 1 — Attention Scores:**
   - Với $q = [1, 2]^T$ và các key $k_1=[1,0]^T, k_2=[0,1]^T, k_3=[1,1]^T$:
   - $s_1 = 1, s_2 = 2, s_3 = 3 \implies s = [1, 2, 3]$.
2. **Bài 2 — Softmax Weights:**
   - $\alpha = \text{softmax}([1, 2, 3]) = [0.090031, 0.244728, 0.665241]$.
   - Tổng $\sum \alpha_i = 1.0$. Phần tử có score lớn nhất nhận weight lớn nhất vì hàm $e^x$ đồng biến và có tính lồi khuếch đại độ lệch.
3. **Bài 3 — Context Vector:**
   - Với $\alpha = [0.1, 0.2, 0.7]$ và $v_1=[1,0]^T, v_2=[0,1]^T, v_3=[2,2]^T$:
   - $c = [1.5, 1.6]^T$. Value $v_3$ đóng góp nhiều nhất ($[1.4, 1.4]^T$).
4. **Bài 4 — Toàn bộ quy trình:**
   - $q = [1, 0]^T \implies s = [1, 0, 1] \implies \alpha = [0.4223, 0.1554, 0.4223] \implies c = [1.2670, 0.7330]^T$.
5. **Scaling & Temperature:**
   - Chia cho $\sqrt{d_k}$ giúp thu hẹp biên độ điểm số, tránh việc softmax bị bão hòa (peaked) gây vanishing gradient khi $d_k$ lớn.
   - Nhiệt độ thấp ($T = 0.5$) làm phân phối tập trung (Max weight = $0.9817$, Entropy = $0.0930$ nats). Nhiệt độ cao ($T = 2.0$) làm phân phối phẳng hơn (Max weight = $0.6652$, Entropy = $0.8324$ nats).
6. **Numerical Stability:**
   - Trừ $\max(x)$ trước khi lấy hàm mũ: $\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$. Phép trừ này bảo toàn toán học 100% nhờ triệt tiêu $e^{-\max(x)}$ và triệt tiêu nguy cơ overflow với các số lớn như $[1000, 1001, 1002]$.

---

## 6. Đánh giá định lượng trên IMDb (Experiment 4)

Mô hình Self-Attention Classifier được huấn luyện 2 epochs trên cùng tập dữ liệu IMDb cân bằng (5,000 train, 1,000 val, 1,000 test; max_length = 100 tokens), sử dụng chung hàm tiền xử lý và tách token của Lab 05 để đảm bảo tính khách quan:

| Mô hình | Kiểu biểu diễn ngữ cảnh | Test Accuracy | Test Precision | Test Recall | Test F1 | Thời gian huấn luyện | Số tham số |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LSTM (LAB 05)** | Final hidden state $h_T$ | $0.5530$ | $0.5560$ | $0.5260$ | $0.5406$ | $27.93$ giây | $801,345$ |
| **Self-Attention (LAB 06)** | Weighted Context + Mean Pooling | **$0.6650$** | **$0.6640$** | **$0.6680$** | **$0.6660$** | **$3.42$ giây** | **$780,353$** |

### Nhận định:
- **Tốc độ:** Self-Attention hoàn thành huấn luyện trong **3.42 giây**, nhanh hơn LSTM (**27.93 giây**) tới **8.1 lần** trên cùng môi trường CPU nhờ loại bỏ vòng lặp tuần tự.
- **Chất lượng:** Test F1 đạt **$0.6660$**, vượt trội hơn hẳn so với LSTM ($0.5406$), chứng minh khả năng tổng hợp ngữ cảnh rải rác trong câu hiệu quả hơn so với việc ép toàn bộ thông tin vào trạng thái ẩn cuối $h_T$.

---

## 7. Giới hạn cố hữu & Điểm quan trọng (Key Takeaways)

1. **"Attention không miễn phí" ($O(T^2)$ Complexity):**
   - Kích thước ma trận attention là $T \times T$. Khi $T$ tăng từ $10$ lên $1,000$, số phần tử tăng gấp $10,000$ lần. Khi $T = 10,000$, ma trận có $100$ triệu phần tử, chiếm $400$ MB bộ nhớ cho 1 head đơn lẻ. Đây là rào cản tính toán lớn đối với tài liệu dài.
2. **Sự thiếu vắng thông tin thứ tự từ:**
   - Self-Attention cơ bản không có khái niệm vị trí. Hoán vị hai từ trong câu sẽ cho cùng tập hợp scores. Cần bổ sung **Positional Encoding** (sẽ học ở LAB 07) để mô hình phân biệt được `"not good"` và `"good not"`.
3. **Attention không đồng nghĩa với Reasoning:**
   - Trọng số Attention chỉ là phân phối xác suất toán học dùng để kết hợp các vector Value, **tuyệt đối không phải là bằng chứng** chứng minh mô hình có tư duy ngôn ngữ hay hiểu ngữ pháp.

---

## 8. Tuyên bố hỗ trợ từ AI (AI Assistance Statement)

Tuân thủ đúng quy định tại **Mục 30 (AI policy)** của `W6.pdf`:

- **Tool:** Google DeepMind Advanced Agentic Coding Assistant (Gemini 3.8 Flash High).
- **Purpose:** Hỗ trợ boilerplate code tự động hóa thực nghiệm (`run_lab6.py`, `build_notebook.py`), hỗ trợ plotting trực quan hóa bằng Matplotlib và hỗ trợ định dạng tài liệu Markdown / KaTeX.
- **Generated:** Khung code huấn luyện PyTorch DataLoader/Optimizer và code vẽ đồ thị.
- **Modified:** Tự tay hiệu chỉnh cấu trúc ma trận Attention trong NumPy để đảm bảo không gọi thư viện dựng sẵn; tối ưu hóa Stable Softmax và bổ sung các hàm tính Entropy; chỉnh sửa cách tính Masked Mean Pooling.
- **Verified:** Tự tay tính toán và kiểm chứng toàn bộ các bài toán Bài 1, Bài 2, Bài 3, Bài 4, Scaling, Temperature và Self-Attention; kiểm tra đối chiếu từng số liệu trong `results.csv`, `error_analysis.md`, `prediction.md` và `calculations.md`.
