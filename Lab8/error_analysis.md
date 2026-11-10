# Báo Cáo Phân Tích Lỗi & Đánh Giá Độ Tin Cậy Thực Nghiệm — LAB 08

**Học phần:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Chủ đề:** Pretrained Transformers và Tinh Chỉnh Mô Hình BERT  
**Đối tượng khảo sát:** Mô hình `PretrainedClassifier` (`bert-base-uncased` Fine-Tuning)  
**Tập dữ liệu đánh giá:** IMDb Test Set ($1,000$ mẫu bình luận độc lập, cân bằng nhãn)  

---

## 1. Đánh Giá Định Lượng & Ma Trận Nhầm Lẫn (Mục 19)

Quá trình tinh chỉnh toàn phần (Full Fine-Tuning) mô hình BERT trên $4,000$ mẫu bình luận IMDb mang lại các chỉ số hiệu năng trên tập kiểm thử $1,000$ mẫu như sau (chi tiết đối chiếu tại biểu đồ `confusion_matrix.png`):

* **Độ chính xác tổng thể (Accuracy):** $\mathbf{86.00\%}$
* **Macro F1-Score:** $\mathbf{0.8649}$
* **Chỉ số lớp Positive:**
  - **Precision:** $0.8390$ ($83.90\%$)
  - **Recall:** $0.8924$ ($89.24\%$)
  - **F1-Score:** $0.8649$
* **Chỉ số lớp Negative:**
  - **Precision:** $0.8845$ ($88.45\%$)
  - **Recall:** $0.8276$ ($82.76\%$)
  - **F1-Score:** $0.8551$

### Phân Tích Ma Trận Nhầm Lẫn (Confusion Matrix):

$$\begin{array}{c|cc}
& \textbf{Dự đoán Negative} & \textbf{Dự đoán Positive} \\
\hline
\textbf{Thực tế Negative (500)} & 414 \text{ (TN)} & 86 \text{ (FP)} \\
\textbf{Thực tế Positive (500)} & 54 \text{ (FN)} & 446 \text{ (TP)} \\
\end{array}$$

* **Nhận định định lượng:**
  1. Mô hình thể hiện xu hướng thiên vị nhẹ về phía tích cực (Positive bias), với độ nhạy (Recall) đối với lớp Positive đạt tới $89.24\%$, trong khi Recall của lớp Negative chỉ đạt $82.76\%$.
  2. Số lượng lỗi **Dương tính giả (False Positive - FP = 86 mẫu)** cao hơn đáng kể so với **Âm tính giả (False Negative - FN = 54 mẫu)**. Điều này cho thấy khi gặp các văn bản có văn phong khen ngợi mỉa mai hoặc các bài phê bình dài chứa nhiều từ vựng tích cực ở đoạn mở đầu, mô hình rất dễ bị cuốn theo các tín hiệu cảm xúc bề mặt.
  3. Tổng cộng có $140$ mẫu phân loại sai ($14\%$). Việc chỉ báo cáo con số Accuracy $86\%$ hoàn toàn che giấu những góc khuất này, đòi hỏi phải tiến hành phân tích định tính từng ca lỗi thực tế.

---

## 2. Bảng Phân Tích Định Tính Các Ca Lỗi Điển Hình (Mục 20)

Dưới đây là bảng khảo sát chi tiết $8$ ca lỗi tiêu biểu được trích xuất trực tiếp từ tập kiểm thử thực tế (`error_samples.csv`), tuân thủ nghiêm ngặt quy định tối thiểu của đề bài ($3$ False Positives, $3$ False Negatives, và $2$ trường hợp sai lệch với độ tự tin cực hạn):

| Nhóm | ID mẫu | Trích đoạn nội dung bình luận (Text Snippet) | Chiều dài (Tokens) | Nhãn thật | Nhãn đoán | Độ tự tin (Conf) | Phân loại lỗi (Error Type) | Giả thuyết nguyên nhân khoa học (Hypothesis) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **FP 1** | `#460` | *"The story of a woman (Ann) on her death bed... First of all the good. The period detail in the movie is great. The dresses, hair, cars... quality acting... Vanessa Redgrave is quite good... Claire Danes is beautiful... Now for the bad, which unfortunately is everything else... boring... cliché... lame..."* | 711 | **neg** | **pos** | **0.9973** | **Long-range dependency & Truncation** | Văn bản có độ dài $711$ tokens nhưng cấu hình mô hình bị giới hạn ở `max_length = 128`. Tác giả thiết kế cấu trúc bài viết theo hướng mở đầu bằng lời khen trang phục và diễn viên (*"great", "quality acting", "quite good", "beautiful"*), sau đó mới chuyển sang chê bai cốt truyện (*"bad", "boring", "lame"*). Do cơ chế cắt chuỗi (truncation), BERT chỉ đọc được 128 token đầu chứa toàn từ ngữ tích cực và bỏ lỡ hoàn toàn phần kết luận tiêu cực ở đuôi. |
| **FP 2** | `#434` | *"The figure of empress Elizabeth of Austria... mostly associated with Romy Schneider and the Sissi trilogy (1950s) where beauty, gentleness, sweetness... a powerful portrayal... However, Jean Daniel Verhaeghe's film, though made 50 years later, appears to be a wrong depiction... 3/10"* | 1067 | **neg** | **pos** | **0.9968** | **Context comparison & Truncation** | Người viết sử dụng thủ pháp so sánh đối chiếu: ca ngợi bộ phim kinh điển thập niên 1950 để làm thước đo chê bai bộ phim chuyển thể hiện đại. 128 token đầu ngập tràn mỹ từ dành cho bản phim cũ (*"beauty", "gentleness", "sweetness"*). Khi bị cắt ngắn, cơ chế Self-Attention gán các đặc trưng tích cực này cho chủ thể bài viết hiện tại. |
| **FP 3** | `#968` | *"This film is the freshman effort of Stephanie Beaton and her new production company. While it suffers from a few problems, as every low budget production does... the passion is evident... the performances are heartfelt... check out 'The Bagman'..."* | 259 | **neg** | **pos** | **0.9952** | **Ambiguous sentiment & Polite phrasing** | Tác giả sử dụng giọng điệu lịch sự, khích lệ nỗ lực của một đoàn làm phim độc lập kinh phí thấp. Mặc dù tổng thể đánh giá tiêu cực (chỉ cho điểm trung bình yếu), các từ ngữ biểu đạt trực diện lại mang sắc thái động viên nhẹ nhàng (*"freshman effort", "heartfelt", "passion"*), làm lu mờ ranh giới phân lớp của hàm Softmax. |
| **FN 1** | `#750` | *"Yes this a B- grade horror. But at least the producers does not pretend this flick is manna from heaven. The plot is corny, a psychotic killer... blood, gore... but it delivers exactly what fans want... worth watching a few times..."* | 265 | **pos** | **neg** | **0.9967** | **Domain-specific expression & Genre tropes** | Trong thể loại phim kinh dị hạng B (B-movie), các thuật ngữ như *"corny plot"*, *"psychotic killer"*, *"blood, gore"* chính là những giá trị giải trí mà đối tượng khán giả đặc thù tìm kiếm. BERT tiếp nhận các từ ngữ bề mặt có trọng số tiêu cực mạnh và suy diễn sai bản chất hài lòng của người xem. |
| **FN 2** | `#462` | *"I've recently watched this movie... we have a lot of fun! This movie is a masterpiece of trash. It is so utterly terrible that you cannot stop laughing... weak script and low-cost budget had created an amazing mix of foolish things..."* | 163 | **pos** | **neg** | **0.9953** | **Sarcasm & Oxymoron (Nghịch hợp)** | Cụm từ đảo nghĩa nghệ thuật *"masterpiece of trash"* (kiệt tác rác rưởi) thể hiện sự thích thú tột bậc đối với dòng phim "dở đến mức phát cười" (so bad it's good). Ma trận Self-Attention bị chi phối nặng nề bởi các từ vựng tiêu cực dồn dập (*"trash", "terrible", "weak script", "foolish"*), dẫn đến việc bỏ sót ngữ dụng học trào phúng. |
| **FN 3** | `#748` | *"Peter Jacksons version(s) are better films overall... That being said, they are not my favorite... Ralph Bakshi LOTR has a superior atmosphere... overall voice acting... I'll give it a score of 8½ out of 10."* | 1327 | **pos** | **neg** | **0.9953** | **Mixed sentiment & Comparative critique** | Bài phê bình chuyên sâu so sánh đa chiều giữa hai tác phẩm điện ảnh kinh điển. Tác giả phân tích sắc sảo hàng loạt khuyết điểm kỹ xảo của bản phim hoạt hình 1978, nhưng chốt lại điểm số xuất sắc 8.5/10. Sự đan xen phức tạp giữa khen và chê khiến vector ẩn `[CLS]` bị nhiễu loạn về phía nhãn tiêu cực. |
| **High Conf 1** | `#598` | *"I'm grateful to Cesar Montano in reviving the Visayan film... Downside: the corny and stilted acting of the American antagonist... lack of authenticity in costume... bright red kit-car... Without those small details, would've been 9/10..."* | 246 | **pos** | **neg** | **0.9952** | **Negation & Defect detailing** | Tác giả bày tỏ lòng biết ơn đối với đạo diễn đã hồi sinh dòng phim địa phương (tổng thể khen 9/10), nhưng dành phần lớn dung lượng câu chữ để liệt kê chi tiết các hạt sạn diễn xuất và phục trang. Mô hình bị thiên lệch bởi tần suất xuất hiện cục bộ của các từ ngữ phê phán. |
| **High Conf 2** | `#62` | *"Okay, this may not be the most sophisticated movie... Actually, there isn't a bit of sophisticatedness in it. It's puerile, adolescent, inoffensive, idiotic..And utterly hilarious. Basically, Richie and Eddie run the worst hotel in Britain... hilarious antics..."* | 218 | **pos** | **neg** | **0.9950** | **Self-deprecating comedy tropes** | Thể loại hài hình thể (slapstick) sử dụng hàng loạt tính từ mang tính hạ thấp bản thân (*"puerile", "adolescent", "idiotic", "worst hotel"*) nhằm tôn vinh sự hài hước tột cùng (*"utterly hilarious"*). BERT không được trang bị đủ tri thức văn hóa để phân biệt phong cách tự trào trào lộng này. |

---

## 3. Phân Tích Độ Tin Cậy: $\text{Confidence} \neq \text{Correctness}$ (Mục 21)

Dựa trên kết quả trực quan hóa tại đồ thị `confidence_analysis.png`:

```
Độ tin cậy được tính:
p = Softmax(logits)
confidence = max(p)
```

Phân bổ dự đoán theo 3 nhóm mức độ tin cậy:
1. **High Confidence ($> 0.80$):** Chiếm trên $88\%$ tổng số mẫu kiểm thử.
2. **Medium Confidence ($0.60 - 0.80$):** Chiếm khoảng $8\%$.
3. **Low Confidence ($< 0.60$):** Chiếm dưới $4\%$.

### Hiện Tượng "Tự Tin Thái Quá Vào Dự Đoán Sai" (Overconfident Errors):
- Quan sát chấn động nhất trong thực nghiệm là: **Toàn bộ $8$ mẫu lỗi phân tích ở Bảng trên đều có mức độ tin cậy toán học vượt ngưỡng $99.5\%$ ($P > 0.995$)**.
- Mô hình đưa ra phán quyết sai lệch $180^\circ$ nhưng lại khẳng định xác suất gần như tuyệt đối ($P \approx 1.0$).
- **Bản chất khoa học:** Hàm Softmax có xu hướng khuếch đại sự chênh lệch giữa các logit đầu ra thông qua hàm số mũ $e^{z_i}$. Khi một câu chứa các từ vựng mang cực tính mạnh xuất hiện trong cửa sổ chú ý, mô hình sẽ đẩy logit của lớp tương ứng lên rất cao. Giá trị Confidence này đơn thuần là **thước đo toán học về khoảng cách hình học tới siêu phẳng phân tách trong không gian đặc trưng đã học**, hoàn toàn **không phản ánh mức độ thấu hiểu chân lý ngữ nghĩa thực sự của con người**.
- **Kết luận:** $\mathbf{\text{Confidence} \neq \text{Correctness}}$. Một mô hình có độ tự tin cao hoàn toàn có thể là một mô hình đang sai lầm nghiêm trọng. Điều này đặt ra yêu cầu cấp thiết về việc cân chuẩn độ tin cậy (Temperature Scaling / Calibration) trong các hệ thống NLP triển khai thực tế.

---

## 4. Trả Lời Toàn Diện 6 Câu Hỏi Bắt Buộc (Mục 28: Q1 — Q6)

### Q1: Model nào hưởng lợi nhiều hơn khi training data ít?
* **Khẳng định:** **Mô hình Pretrained Transformer (BERT)** hưởng lợi vượt trội áp đảo khi dữ liệu huấn luyện khan hiếm.
* **Chứng cứ thực nghiệm (Evidence):**  
  Từ kết quả Thí nghiệm 3 (biểu đồ `data_efficiency.png`), khi cắt giảm dữ liệu huấn luyện xuống chỉ còn $10\%$ ($400$ mẫu):
  - Pretrained BERT vẫn duy trì độ chính xác ấn tượng **$82.60\%$** và Macro F1 **$0.8249$** (chỉ suy giảm nhẹ $3.40\%$ so với khi dùng trọn vẹn $4,000$ mẫu).
  - Trong khi đó, mô hình Transformer from scratch trên $10\%$ dữ liệu sụp đổ hoàn toàn về mức **$50.20\%$** (chỉ tương đương với việc tung đồng xu ngẫu nhiên).
* **Bản chất khoa học (Interpretation):**  
  Mô hình Pretrained đã hoàn tất việc học cấu trúc cú pháp, trường từ vựng và quy luật ngữ pháp từ hàng tỷ từ trong giai đoạn tự giám sát. Đối với downstream task, nó chỉ cần một lượng nhỏ mẫu ví dụ để học cách định vị siêu phẳng phân lớp. Ngược lại, mô hình from-scratch phải tối ưu hóa hàng triệu tham số từ số không, nên việc thiếu dữ liệu sẽ lập tức đẩy mô hình vào trạng thái suy biến hoàn toàn (mode collapse).

---

### Q2: Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?
* **Khẳng định:** Điều đó chứng minh **sức mạnh chuyển giao tri thức mang tính quyết định của việc học biểu diễn ngôn ngữ tổng quát hai chiều (Bidirectional Contextual Representation) qua nhiệm vụ Masked Language Modeling (MLM)**.
* **Chứng cứ thực nghiệm (Evidence):**  
  Ở Thí nghiệm 1, trên cùng một tập dữ liệu $4,000$ mẫu, Pretrained BERT đạt Accuracy **$86.00\%$** (F1 = $0.8649$), trong khi Transformer from-scratch chỉ đạt **$49.80\%$** (F1 = $0.0000$).
* **Bản chất khoa học (Interpretation):**  
  Sự vượt trội này chứng minh rằng một mạng nơ-ron không thể tự động thấu hiểu ngữ nghĩa tự nhiên nếu chỉ được cung cấp một vài nghìn câu bình luận cục bộ. Khởi tạo trọng số từ mô hình tiền huấn luyện cung cấp một "tọa độ không gian tham số" (initialization landscape) cực kỳ tối ưu, giúp thuật toán Gradient Descent dễ dàng thoát khỏi các cực tiểu địa phương nghèo nàn và hội tụ tới trạng thái tổng quát hóa cao.

---

### Q3: Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?
* **Khẳng định:** Điều đó có ý nghĩa rằng **không gian đặc trưng tĩnh kết tinh tại vector ẩn `[CLS]` của mô hình Pretrained đã chứa đựng đầy đủ các thuộc tính phân tách ngữ nghĩa cần thiết cho bài toán hạ nguồn**.
* **Chứng cứ thực nghiệm (Evidence):**  
  Ở Thí nghiệm 2, mô hình Frozen Encoder (chỉ huấn luyện đúng một tầng Linear với $1,538$ tham số) đã đạt Accuracy **$77.90\%$** và F1 **$0.7877$**, chỉ thấp hơn Full Fine-Tuning $8.10\%$ nhưng thời gian huấn luyện nhanh gấp gần 3 lần ($65.68\text{s}$ so với $178.34\text{s}$).
* **Bản chất khoa học (Interpretation):**  
  Các cụm biểu diễn ngữ nghĩa của BERT sau pre-training đã tự động phân ly tương đối rõ ràng. Bộ phân loại tuyến tính đơn giản có thể dễ dàng tìm ra một mặt phẳng phân cách chấp nhận được mà không cần phải can thiệp sâu vào các tầng trích xuất đặc trưng bên dưới.

---

### Q4: Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?
* **Khẳng định:** Điều đó gợi ý rằng **tác vụ đích có tính chất chuyên biệt hóa cao hoặc phân phối miền dữ liệu có sự khác biệt so với mục tiêu tổng quát của pre-training, đòi hỏi các cơ chế chú ý phải được tái căn chỉnh (Task/Domain Adaptation)**.
* **Chứng cứ thực nghiệm (Evidence):**  
  Full Fine-Tuning giúp tăng vọt độ chính xác từ $77.90\%$ lên $86.00\%$ ($+8.10\%$, giảm được gần $37\%$ số lượng mẫu dự đoán sai).
* **Bản chất khoa học (Interpretation):**  
  Mục tiêu tiền huấn luyện MLM tập trung vào việc khôi phục token bị che dựa trên ngữ cảnh chung, chứ không tối ưu hóa việc phân cực cảm xúc. Khi mở khóa (unfreeze) toàn bộ mạng, gradient lan truyền ngược xuyên suốt 12 khối Transformer sẽ "uốn nắn" lại các ma trận Query, Key, Value trong từng đầu chú ý để tập trung cao độ vào các từ ngữ đánh giá khen/chê đặc thù của điện ảnh, giúp kéo giãn khoảng cách cụm nhãn trong không gian tiềm ẩn.

---

### Q5: Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?
* **Khẳng định:** Nguyên nhân căn bản xuất phát từ **hiện tượng Dịch chuyển miền dữ liệu (Domain Shift) về phân phối từ vựng, văn phong biểu đạt và giả định ngữ cảnh**.
* **Chứng cứ & Diễn giải:**  
  Mặc dù trong Thí nghiệm 5, khi chuyển từ IMDb sang Amazon Reviews, hiệu năng vẫn duy trì ở mức cao ($90.40\%$) do cả hai đều là dạng đánh giá thương mại phổ thông; nhưng nếu chuyển sang các miền dữ liệu chuyên sâu như văn bản y sinh (PubMed), hồ sơ pháp lý, hoặc mạng xã hội chứa nhiều tiếng lóng (Twitter/TikTok), hiệu năng của mô hình fine-tune trên IMDb sẽ sụt giảm nghiêm trọng.
* **Bản chất khoa học (Interpretation):**  
  Mô hình fine-tuned đã bị quá khớp cục bộ (locally overfitted) vào tập từ vựng điện ảnh (*"director", "plot", "acting", "cinematography"*). Khi gặp miền dữ liệu mới với các thực thể lạ, cấu trúc câu khác biệt và từ ngữ chuyên ngành không xuất hiện trong tập huấn luyện, các cơ chế chú ý đã học sẽ mất phương hướng, dẫn đến sai lệch nhận thức nghiêm trọng.

---

### Q6: Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?
* **Khẳng định:** **TUYỆT ĐỐI KHÔNG NÊN tiếp tục tăng sequence length.**
* **Chứng cứ & Cơ sở toán học từ LAB 06–07:**  
  1. *Độ phức tạp tính toán bậc hai $\mathcal{O}(T^2)$:*  
     Trong kiến trúc Self-Attention, ma trận chú ý có kích thước $T \times T$. Chi phí tính toán phép nhân ma trận $QK^T$ và bộ nhớ lưu trữ ma trận trọng số chú ý tỷ lệ thuận với bình phương độ dài chuỗi $T^2$.
  2. *Dữ liệu thực nghiệm Thí nghiệm 4:*  
     Khi tăng độ dài từ $T = 128$ lên $T = 256$, thời gian huấn luyện tăng hơn gấp đôi (từ $178.34\text{s}$ lên $370.93\text{s}$) và mức tiêu thụ đỉnh RAM GPU tăng từ $4.68\text{GB}$ lên $8.49\text{GB}$ (tăng $81\%$).
  3. *Nguyên lý điểm thắt thông tin (Information Bottleneck):*  
     Trong các bài bình luận, cảm xúc cốt lõi của người viết thường tập trung ở vài câu mở đầu hoặc phần tóm tắt kết luận ở cuối. Việc mở rộng $T$ vượt quá ngưỡng bão hòa thông tin chỉ nạp thêm các đoạn văn kể lại cốt truyện (nhiễu thông tin), khiến chi phí tài nguyên tăng vọt mà không mang lại giá trị gia tăng thực tiễn.

---

## 5. Nguyên Tắc Phương Pháp Luận: Chuỗi Lập Luận Khoa Học

Để đảm bảo tính chuẩn xác trong nghiên cứu NLP, người nghiên cứu bắt buộc phải tuân thủ nghiêm ngặt chuỗi lập luận ba bước:

$$\text{\textbf{Observation}} \text{ (Quan sát số liệu thô)} \longrightarrow \text{\textbf{Evidence}} \text{ (Bằng chứng thực nghiệm đối chứng)} \longrightarrow \text{\textbf{Interpretation}} \text{ (Suy diễn bản chất khoa học)}$$

* **Cảnh báo lỗi ngụy biện:**  
  Không bao giờ được nhảy vội từ một con số định lượng đơn lẻ:
  $$\text{Accuracy} = 86.0\% \centernot\implies \text{"Mô hình BERT đã thực sự thấu hiểu ngôn ngữ và cảm xúc con người."}$$

* **Lập luận chuẩn mực được thiết lập:**
  1. **Quan sát (Observation):** Mô hình BERT fine-tuning đạt độ chính xác $86.00\%$ trên $1,000$ mẫu kiểm thử IMDb và đạt Recall $89.24\%$ trên nhãn Positive.
  2. **Bằng chứng (Evidence):** Mô hình phân loại xuất sắc các câu có chứa các từ khóa cảm xúc rõ ràng và phân bố ở đoạn đầu văn bản. Tuy nhiên, mô hình hoàn toàn bất lực và đưa ra dự đoán sai lệch với độ tự tin $> 99.5\%$ đối với các câu chứa nghệ thuật nghịch hợp trào phúng (*"masterpiece of trash"*), phong cách tự trào của phim hài, hoặc các bài đánh giá dài bị cắt cụt do giới hạn $T = 128$.
  3. **Suy diễn khoa học (Interpretation):** Mô hình BERT đã học được cách khai thác các mẫu tương quan từ vựng bề mặt và cấu trúc ngữ cảnh cục bộ tốt hơn rất nhiều so với mô hình ngẫu nhiên. Tuy nhiên, nó vẫn chưa sở hữu năng lực suy luận ngữ dụng học sâu (Pragmatics), chưa hiểu được ý đồ ẩn dụ tinh tế của con người, và vẫn bị giới hạn cố hữu bởi chiều dài cửa sổ chú ý.
