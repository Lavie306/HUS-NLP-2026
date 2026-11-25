# Calculations — LAB 10: Extractive Question Answering

## 1. Calculation Exercise 1 — Start / End (Mục 6)

Giả sử context sau khi tokenize theo từng từ (đánh số từ `0`):

| Token Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Token** | `The` | `capital` | `of` | `France` | `is` | `Paris` | `.` |

1. Với Question: `What is the capital of France?` và Answer: `Paris`:
   $$\text{start} = 5, \qquad \text{end} = 5$$
   (Ở đây `end` là vị trí **bao gồm / inclusive**, nên trong Python slicing lấy `tokens[start : end + 1] = tokens[5:6] = ["Paris"]`.)

2. Với Answer: `capital of France`:
   $$\text{start} = 1, \qquad \text{end} = 3$$
   Kiểm tra: `tokens[1:4] = ["capital", "of", "France"]`.

---

## 2. Calculation Exercise 2 — Start Loss (Mục 7)

Giả sử gold answer start position là token $i = 5$, và mô hình dự đoán phân phối xác suất $P_{\text{start}}(5) = 0.60$:

$$L_{\text{start}} = -\ln P_{\text{start}}(5) = -\ln(0.60) \approx 0.5108$$

Tương tự, với vị trí kết thúc thực tế $j$ và phân phối xác suất dự đoán $P_{\text{end}}$:

$$L_{\text{end}} = -\ln P_{\text{end}}(j)$$

*(Đề bài không cung cấp phân phối xác suất cụ thể cho `end` nên giữ công thức tổng quát theo $P_{\text{end}}(j)$.)*

---

## 3. QA Loss (Mục 8)

Hàm mất mát chuẩn cho bài toán Extractive QA là trung bình cộng của Cross-Entropy loss tại hai đầu:

$$L = \frac{L_{\text{start}} + L_{\text{end}}}{2} = \frac{-\ln P_{\text{start}}(i) - \ln P_{\text{end}}(j)}{2}$$

**Giải thích: Vì sao cần hai prediction thay vì chỉ một prediction?**
- Trong Extractive QA, câu trả lời là một **đoạn văn bản liên tiếp (span)** có độ dài thay đổi tùy theo từng câu hỏi (có thể 1 token như `Paris`, hoặc nhiều token như `capital of France`).
- Nếu chỉ dự đoán một vị trí `start`, mô hình chỉ biết đáp án bắt đầu ở đâu nhưng không biết đáp án kéo dài đến token nào. Ngược lại, dự đoán đồng thời hai vị trí `(start, end)` với ràng buộc $\text{start} \le \text{end}$ cho phép xác định trọn vẹn ranh giới của một span có độ dài bất kỳ trong context.

---

## 4. Calculation Exercise 3 — Character Span (Mục 12)

Cho context:

```text
The capital of France is Paris.
```

Đánh chỉ số ký tự từ `0`:
- `T` ở index `0`, ..., khoảng trắng sau `is` ở index `24`.
- **Index của ký tự đầu tiên của `Paris`** (`'P'`): **`25`**.
- **Index của ký tự cuối của `Paris`** (`'s'`): **`29`**.
- **Answer span theo character** (dạng half-open `[start, end)` dùng trong Python slicing): **`[25, 30)`**.
  Kiểm tra: `context[25:30] == "Paris"`.
- **Trong chuỗi token minh họa ở Bài 1** (`[The, capital, of, France, is, Paris, .]`):
  - Token tương ứng: `Paris`
  - `start_token = 5`, `end_token = 5`.
- **Với tokenizer thực tế (`albert/albert-base-v2`)** cho cặp:
  - Question: `What is the capital of France?`
  - Context: `The capital of France is Paris.`
  - Toàn bộ chuỗi đầu vào gồm `[CLS] + question tokens + [SEP] + context tokens + [SEP]`. Token `▁paris` nằm ở vị trí **`14`** trong toàn bộ `input_ids`, với `offset_mapping[14] = (25, 30)`. Do đó `start_token = 14`, `end_token = 14`. *(Được kiểm chứng trực tiếp trong `tokenizer_alignment.ipynb`.)*

**Lưu ý quan trọng:** `character index` (`25`) đếm theo từng ký tự trong context gốc, trong khi `token index` (`14`) đếm theo các subword token trong toàn bộ chuỗi đầu vào đã ghép `[CLS]`, question và `[SEP]`. Không bao giờ được dùng trực tiếp character offset làm token index.

---

## 5. Token-level F1 for QA (Mục 19)

Xét ví dụ:
- **Gold:** `Paris France` $\rightarrow$ sau chuẩn hóa: `["paris", "france"]` ($\text{\#gold} = 2$)
- **Prediction:** `France` $\rightarrow$ sau chuẩn hóa: `["france"]` ($\text{\#predicted} = 1$)
- **Số token trùng nhau (overlap):** $\text{\#overlap} = 1$ (`"france"`)

Tính các chỉ số:
- **Precision:**
  $$P = \frac{\text{\#overlap}}{\text{\#predicted}} = \frac{1}{1} = 1.0$$
- **Recall:**
  $$R = \frac{\text{\#overlap}}{\text{\#gold}} = \frac{1}{2} = 0.5$$
- **F1:**
  $$F1 = \frac{2PR}{P + R} = \frac{2 \times 1.0 \times 0.5}{1.0 + 0.5} = \frac{1.0}{1.5} = \frac{2}{3} \approx 0.6667 \quad (\text{tức } 66.67\%)$$
- **Exact Match (EM):** $0$ (vì `"france" != "paris france"` sau normalization).
