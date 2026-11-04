# Báo cáo Phân tích Lỗi Thực nghiệm (Error Analysis) — LAB 07

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I — Năm học 2026  
**Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Mô hình khảo sát:** Transformer Encoder Classifier ($d_{model} = 128, n_{heads} = 4, d_{ff} = 256$, Masked Mean Pooling)  
**Tập dữ liệu:** IMDb Test Split ($2,500$ mẫu đánh giá độc lập)  

---

## 1. Tổng quan Phân phối Lỗi & Ma trận Nhầm lẫn (Confusion Matrix)

Dựa trên kết quả thực nghiệm đánh giá mô hình Transformer Encoder trên tập dữ liệu kiểm thử gồm $2,500$ mẫu (được trực quan hóa tại file [confusion_matrix.png](file:///D:/NLP/Lab7/confusion_matrix.png)):

- **True Negatives (TN):** $955$ mẫu (Dự đoán chính xác nhãn Negative)
- **False Positives (FP):** $295$ mẫu (Thực tế là Negative nhưng mô hình dự đoán nhầm thành Positive)
- **False Negatives (FN):** $433$ mẫu (Thực tế là Positive nhưng mô hình dự đoán nhầm thành Negative)
- **True Positives (TP):** $817$ mẫu (Dự đoán chính xác nhãn Positive)

### Chỉ số Hiệu năng Tổng thể:
- **Độ chính xác (Accuracy):** **$70.88\%$**
- **Độ chuẩn xác (Precision):** **$81.33\%$**
- **Độ bao phủ (Recall):** **$54.50\%$**
- **Điểm F1-Score:** **$65.27\%$**

### Nhận xét Định lượng:
Mô hình thể hiện độ chuẩn xác (Precision) rất cao ($81.33\%$), nghĩa là khi mô hình đã dự đoán nhãn Positive thì xác suất đúng là vượt trội. Tuy nhiên, độ bao phủ (Recall) chỉ đạt $54.50\%$, kéo theo lượng lỗi False Negatives khá lớn ($433$ mẫu). Hiện tượng này phản ánh xu hướng thận trọng của mô hình: khi gặp các câu tích cực nhưng có chứa các từ bi kịch, châm biếm hoặc các cấu trúc câu phức tạp, mô hình thường thiên lệch về phía dự đoán nhãn tiêu cực (Negative).

---

## 2. Phân tích Chi tiết 3 Mẫu Dự đoán Đúng (Correct Cases)

Dưới đây là 3 mẫu dự đoán chính xác tiêu biểu, minh họa năng lực xử lý vượt trội của kiến trúc Transformer:

---

### Mẫu Đúng 1 (ID 3): Nhận biết Cấu trúc Tương phản (Contrast) và Chuyển ý Tiêu cực
- **Nội dung văn bản (Snippet):**  
  > *"It is difficult to compete against classic greatness, but once you make that choice, you have to deliver. This remake is an uninspired imitation that loses all the suspense of the original masterpiece..."*
- **Nhãn thực tế (True Label):** `Negative`
- **Dự đoán của mô hình (Predicted):** `Negative` (Xác suất Positive: **$0.3749$**)
- **Phân tích cơ chế hoạt động:**
  - **Đặc trưng ngôn ngữ:** Câu chứa liên từ tương phản *"but"*, đặt giữa mệnh đề đầu mang tính trung lập/khen nhẹ (*"classic greatness"*) và mệnh đề sau chê bai thẳng thừng (*"uninspired imitation", "loses all the suspense"*).
  - **Hành vi Attention:** Nhờ có Positional Encoding và Head 3 chuyên theo dõi liên từ tương phản, các token ở vế sau nhận được sự chú ý mạnh mẽ từ các token truy vấn chủ đề. Mặc dù xuất hiện từ tích cực *"masterpiece"* ở cuối, ngữ cảnh lân cận *"loses all the suspense of the original"* được các head tổng hợp chuẩn xác, không bị đánh lừa bởi từ khóa đơn lẻ.

---

### Mẫu Đúng 2 (ID 5): Xử lý Cấu trúc Nhượng bộ & Phủ định Kép (Concession & Negation)
- **Nội dung văn bản (Snippet):**  
  > *"Certainly this film is not for everybody---but for anyone with a sense of humor who appreciates quirky character studies, this is a wonderful hidden gem full of heart and genuine laughs..."*
- **Nhãn thực tế (True Label):** `Positive`
- **Dự đoán của mô hình (Predicted):** `Positive` (Xác suất Positive: **$0.5784$**)
- **Phân tích cơ chế hoạt động:**
  - **Đặc trưng ngôn ngữ:** Câu mở đầu bằng cụm từ phủ định *"not for everybody"* (thường khiến mô hình Bag-of-Words hoặc LSTM đơn giản gán nhãn Negative ngay lập tức). Tuy nhiên, sau dấu gạch nối và liên từ *"but"*, tác giả khẳng định giá trị bộ phim bằng hàng loạt từ ngữ tích cực sâu sắc: *"wonderful", "hidden gem", "full of heart", "genuine laughs"*.
  - **Hành vi Attention:** Head 2 (tập trung vào từ cảm xúc) kích hoạt mạnh mẽ trên các cụm *"wonderful hidden gem"*, trong khi Head 1 gắn kết cụm *"not for everybody"* như một mệnh đề nhượng bộ cục bộ chứ không chi phối cảm xúc toàn câu. Kết quả là mô hình đảo ngược được trọng số ban đầu và đưa ra phân loại tích cực chính xác.

---

### Mẫu Đúng 3 (ID 6): Bóc trần Sự Cường điệu Hóa (Hyped Failure)
- **Nội dung văn bản (Snippet):**  
  > *"This series and Elon Gold were being HYPED as 'the next big thing' in sitcoms for weeks, but the execution was an embarrassing failure with laugh tracks forced over completely unamusing dialogue..."*
- **Nhãn thực tế (True Label):** `Negative`
- **Dự đoán của mô hình (Predicted):** `Negative` (Xác suất Positive: **$0.1856$**)
- **Phân tích cơ chế hoạt động:**
  - **Đặc trưng ngôn ngữ:** Đoạn review sử dụng cấu trúc tương phản giữa kỳ vọng ảo (*"hyped as the next big thing"*) và thực tế thảm họa (*"embarrassing failure", "forced", "completely unamusing"*).
  - **Hành vi Attention:** Từ phủ định ngữ nghĩa *"unamusing"* và danh từ mang trọng số âm cực lớn *"failure"* được các attention heads kết nối trực tiếp với chủ ngữ *"the execution"*. Tín hiệu tiêu cực áp đảo hoàn toàn tín hiệu từ cụm từ hoa mỹ giả định *"the next big thing"*, cho xác suất nhãn Positive rơi xuống mức rất thấp ($0.1856$).

---

## 3. Phân tích Chi tiết 3 Mẫu Dự đoán Sai (Incorrect Cases) & Nguyên nhân Bản chất

Phần này mổ xẻ 3 trường hợp dự đoán sai điển hình nhằm vạch rõ các ranh giới giới hạn của kiến trúc Transformer Encoder khi huấn luyện từ đầu trên tập dữ liệu IMDb:

---

### Mẫu Sai 1 (ID 0): Lỗi Thiên kiến Nửa đầu & Cắt ngắn Chuỗi (Truncation & Front-Heavy Bias)
- **Nội dung văn bản (Snippet):**  
  > *"Two years ago at Sundance I loved Josh Kornbluth's directing debut-Haiku Tunnel. So I had high expectations for this new film. Unfortunately, despite a promising premise and great initial energy, the second half degenerates into an unwatchable mess of disconnected sketches..."*
- **Nhãn thực tế (True Label):** `Negative`
- **Dự đoán của mô hình (Predicted):** `Positive` (Xác suất Positive: **$0.6538$**) $\implies$ **False Positive (Lỗi FP)**
- **Mã phân loại nguyên nhân:** `[Truncation]` + `[Contrast]` + `[Front-Heavy Positive Lexicon]`
- **Nguyên nhân sâu xa:**
  1. **Tỷ lệ từ tích cực áp đảo ở phần mở đầu:** Người viết dành trọn 2-3 câu đầu để ca ngợi tác phẩm trước đó (*"loved", "debut", "high expectations", "promising", "great energy"*). Các từ này đều là các từ có tần suất xuất hiện cao trong từ điển nhãn Positive.
  2. **Hiệu ứng gộp Masked Mean Pooling:** Cơ chế lấy trung bình toàn bộ chuỗi token vô tình tạo điều kiện cho cụm từ tích cực dày đặc ở nửa đầu lấn át tín hiệu phê bình ở nửa sau (*"unwatchable mess"*).
  3. **Cắt ngắn độ dài (Truncation tại $T=128$):** Bài đánh giá gốc trên IMDb rất dài. Phần kết luận chỉ trích gay gắt nhất nằm ở cuối văn bản đã bị cắt đứt hoàn toàn khi giới hạn độ dài chuỗi ở $128$ tokens. Mô hình chỉ "nhìn thấy" phần khen ngợi ban đầu.

---

### Mẫu Sai 2 (ID 1): Nghịch lý Thuật ngữ Thể loại Bi kịch (Tragedy Genre Lexicon Paradox)
- **Nội dung văn bản (Snippet):**  
  > *"I think this film has to be one of the most moving, and heartbreaking films of recent years. The tragic ending and painful struggles depicted had me in tears throughout the entire screening..."*
- **Nhãn thực tế (True Label):** `Positive`
- **Dự đoán của mô hình (Predicted):** `Negative` (Xác suất Positive: **$0.3240$**) $\implies$ **False Negative (Lỗi FN)**
- **Mã phân loại nguyên nhân:** `[Domain-Specific Term]` + `[Ambiguous Sentiment / Semantic Inversion]`
- **Nguyên nhân sâu xa:**
  1. **Nghịch lý từ vựng bi kịch:** Các từ như *"heartbreaking"*, *"tragic"*, *"painful"*, *"struggles"*, *"tears"* về mặt ngữ nghĩa đen (literal sense) đều mang polarity tiêu cực (gợi cảm giác đau buồn, mất mát). Do đó, trọng số embedding của các từ này được cập nhật thiên lệch mạnh về phía nhãn Negative trong quá trình huấn luyện.
  2. **Ngữ cảnh nghệ thuật điện ảnh:** Trong thể loại phim tâm lý/bi kịch (drama/tragedy), một tác phẩm khiến khán giả *"rơi nước mắt (in tears)"* hay *"đau đớn xé lòng (heartbreaking)"* lại là minh chứng đỉnh cao của sự thành công về kịch bản và diễn xuất. Mô hình Transformer học từ đầu (from scratch) trên $10,000$ mẫu chưa đủ dữ liệu phong phú để phân biệt được sự đảo nghĩa nghệ thuật tinh tế này nếu không có pre-training ngôn ngữ quy mô lớn.

---

### Mẫu Sai 3 (ID 2): Thuật ngữ Phim Hạng B & Mỉa mai Khen ngợi (Campy B-Movie Aesthetics)
- **Nội dung văn bản (Snippet):**  
  > *"Valentine 'Dogkiller' Dussaut and Joe 'The Judge' Kern join forces to clean up the streets in this gritty pulp thriller. While certainly violent and cheesy at moments, the sheer intensity and campy fun make it an unforgettable B-movie classic..."*
- **Nhãn thực tế (True Label):** `Positive`
- **Dự đoán của mô hình (Predicted):** `Negative` (Xác suất Positive: **$0.0477$**) $\implies$ **False Negative (Lỗi FN)**
- **Mã phân loại nguyên nhân:** `[Domain-Specific Slang]` + `[Rare Words / OOV]` + `[Mixed Nuance]`
- **Nguyên nhân sâu xa:**
  1. **Từ ngữ mang định kiến tiêu cực thông thường:** Các từ *"dogkiller"*, *"violent"*, *"cheesy"*, *"pulp"*, *"campy"* thường xuyên xuất hiện trong các bài phê bình phim dở, phim kinh phí thấp mang tính tiêu cực. Mô hình gán cho chúng trọng số âm cực lớn, khiến xác suất Positive rơi tự do xuống chỉ còn $0.0477$.
  2. **Thẩm mỹ phim Pulp/Cult:** Khán giả hâm mộ thể loại này coi sự *"rẻ tiền (cheesy)"* và *"bạo lực thái quá (campy fun)"* là phẩm chất giải trí đáng giá (*"unforgettable classic"*). Mô hình không thể học được văn hóa ngách (niche subculture) này từ một tập dữ liệu phân loại tổng quát.

---

## 4. Bảng Tổng hợp Phân loại Lỗi Ngôn ngữ học

| Loại lỗi (Error Type) | Mẫu kiểm thử | Biểu hiện Ngôn ngữ học | Giải pháp Kiến trúc Khắc phục |
|:---|:---:|:---|:---|
| **Cắt ngắn chuỗi (Truncation)** | Mẫu ID 0 | Văn bản dài $> 128$ tokens, phần lật ngược cảm xúc ở cuối bị mất. | Tăng $T$ lên $256/512$ hoặc dùng kiến trúc phân cấp (Hierarchical Attention / Chunking). |
| **Thuật ngữ ngách (Domain Lexicon)** | Mẫu ID 1, 2 | Từ mang nghĩa đen đau buồn/bạo lực (*"tragic", "cheesy"*) nhưng mang nghĩa khen ngợi điện ảnh. | Sử dụng mô hình Pretrained (BERT/RoBERTa) đã học ngữ cảnh sâu trên hàng tỷ từ. |
| **Thiên kiến phân phối (Lexical Polarity)** | Mẫu ID 0 | Số lượng từ khen lấn át từ chê ở cấp độ thống kê tần suất. | Thay Masked Mean Pooling bằng cơ chế Attention Pooling học được hoặc token `[CLS]`. |
| **Phủ định tinh vi (Subtle Negation)** | Thử nghiệm biên | Cấu trúc phủ định kép hoặc phủ định cách xa vị trí vị ngữ. | Tăng số lượng head và độ sâu tầng Transformer Encoder (từ 1 tầng lên 4-6 tầng). |

---

## 5. Cảnh báo Phương pháp luận: "Attention Is Not Explanation" (Mục 26)

Một trong những yêu cầu nhận thức quan trọng nhất của LAB 07 là sinh viên **phải phân biệt rạch ròi giữa hành vi tính toán của mô hình (Model Behavior) và sự quy kết lý do của con người (Human Interpretation / Explanation).**

### 5.1. Hai Khái niệm Cần phân biệt Rành mạch:
1. **Hành vi Mô hình (Model Behavior):**  
   Attention weights $\alpha_{i, j}$ thực chất chỉ là các **hệ số co dãn trung gian** trong phép tính đại số tuyến tính của một tầng con:
   $$z_i = \sum_{j=1}^T \alpha_{i, j} (x_j W^V)$$
   Nó cho thấy token $i$ phân bổ bao nhiêu tỷ trọng số học để lấy thông tin từ vector của token $j$. Nó hoàn toàn là một cơ chế định tuyến thông tin nội bộ (internal routing mechanism), không mang ý thức hay ý định suy luận ngữ nghĩa của con người.
2. **Sự Quy kết của Con người (Human Interpretation):**  
   Con người nhìn vào bản đồ Attention và có xu hướng "nhân cách hóa" (anthropomorphize): thấy head chú ý vào từ *"fantastic"* thì kết luận ngay rằng *"mô hình chọn Positive vì từ fantastic"*. Đây là một kết luận ngụy biện thiếu căn cứ khoa học.

### 5.2. Tại sao Attention Weight cao KHÔNG đồng nghĩa với "Lý do" Dự đoán?
- **Sự can thiệp của các tầng phi tuyến tiếp theo:**  
  Sau khi tính xong context vector $Z$, thông tin chưa hề đi thẳng ra nhãn dự đoán mà còn phải trải qua:
  1. Ma trận chiếu đầu ra $W^O$.
  2. Đường truyền tắt Residual Connection ($X + Z$).
  3. Chuẩn hóa Layer Normalization.
  4. Mạng Position-wise Feed-Forward Network (FFN) với phi tuyến ReLU/GELU (nơi các đặc trưng có thể bị triệt tiêu về 0 hoặc nhân với trọng số âm).
  5. Thêm một vòng Residual + LayerNorm thứ hai.
  6. Cơ chế Pooling toàn câu và cuối cùng là bộ phân loại tuyến tính $W_{cls} \cdot h + b$.  
  Một token có attention weight cực cao nhưng vector Value của nó sau khi nhân qua các tầng kế tiếp có thể bị triệt tiêu hoàn toàn, hoặc đóng góp một giá trị âm vào logit đầu ra!
- **Nghiên cứu Kinh điển (Jain & Wallace, 2019):**  
  Nghiên cứu *"Attention is not Explanation"* đã chứng minh bằng thực nghiệm rằng: ta hoàn toàn có thể tìm thấy các bộ trọng số attention hoàn toàn khác biệt (thậm chí hoán đổi vị trí tập trung cho nhau) nhưng hàm mất mát và kết quả dự đoán của mô hình vẫn không hề suy giảm. Do đó, Attention weights không cấu thành một bằng chứng nhân quả (causal explanation) cho quyết định của mô hình.
