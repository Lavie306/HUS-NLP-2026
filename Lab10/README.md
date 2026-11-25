# LAB 10 — Extractive Question Answering
## From Token Classification to Span Prediction

- **Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP 2026)
- **Trường:** Đại học Khoa học Tự nhiên — ĐHQGHN (VNU-HUS)
- **Giảng viên / TA:** Phạm Ngọc Hải
- **Sinh viên thực hiện:** Nguyễn Trọng Thành

---

## 1. Danh mục sản phẩm (Deliverables Checklist — Mục 31)

```text
Lab10/
├── README.md                     # Báo cáo tổng quan, tái lập thực nghiệm và hướng dẫn sử dụng
├── W10.pdf                       # Đề bài và đặc tả thực hành LAB 10
├── calculations.md               # Lời giải chi tiết Bài tập 1, 2, 3, QA Loss và Token-level F1
├── prediction.md                 # 4 dự đoán lý thuyết trước khi chạy thực nghiệm
├── tokenizer_alignment.ipynb     # Kiểm chứng tokenization, offset_mapping và sliding window (đã chạy)
├── qa_model.py                   # Dữ liệu SQuAD 1.1, KeywordBaseline, span alignment, ALBERT QA & decoding
├── qa_finetuning.ipynb           # Huấn luyện, 5 thí nghiệm có kiểm soát, phân tích lỗi & Learning Check
├── results.csv                   # Bảng tổng hợp EM, F1, validation scores và training time
├── learning_curves.png           # Đồ thị Training/Validation Loss và Validation EM/F1
├── em_f1_analysis.png            # So sánh EM vs F1 giữa Baseline và ALBERT
├── answer_length_analysis.png    # Phân tích EM/F1 theo độ dài đáp án (1, 2–3, 4–6, 7+ tokens)
├── error_analysis.md             # Phân tích 15 mẫu lỗi (5 wrong start, 5 wrong end, 5 wrong span)
├── document_qa.py                # Ứng dụng Mini Document QA trên ngữ cảnh/tài liệu cung cấp trực tiếp
└── reflection.md                 # Bài thu hoạch 300–500 từ và cầu nối sang LAB 11–12 (Retrieval & RAG)
```

---

## 2. Thiết lập thực nghiệm & Tính tái lập (Reproducibility — Mục 30)

| Thành phần | Giá trị / Cấu hình |
| :--- | :--- |
| **Model** | `albert/albert-base-v2` (revision `8e2f239c5f8a2c0f253781ca60135db913e5c80c`) |
| **Tokenizer** | `AutoTokenizer.from_pretrained('albert/albert-base-v2', use_fast=True)` |
| **Dataset** | SQuAD 1.1 (`train-v1.1.json`, `dev-v1.1.json`) |
| **Train split** | 1.000 câu hỏi (lấy mẫu ngẫu nhiên có kiểm soát từ official train) |
| **Validation split** | 200 câu hỏi (từ một nửa số bài viết không giao nhau của official dev) |
| **Test split** | 200 câu hỏi (từ nửa bài viết còn lại của official dev, hoàn toàn tách biệt bài viết) |
| **Max sequence length** | 128, 256, 384 (Experiment 1) |
| **Stride** | 32 tokens (cho phép dùng chung trên cả 3 độ dài 128/256/384; minh họa thêm 384/stride 128 trong notebook) |
| **Batch size** | 4 (dynamic padding theo batch) |
| **Learning rate** | `3e-5` (linear warmup 10% steps) |
| **Epochs** | 2 (giữ checkpoint có Validation F1 cao nhất) |
| **Optimizer** | AdamW (`weight_decay = 0.01`, gradient clipping `1.0`) |
| **Random seed** | 42 |
| **Hardware** | CPU (x86_64) |
| **Training time** | `max_length=128`: 1894.0s \| `max_length=256`: 1286.0s \| `max_length=384`: 1029.4s |

---

## 3. Tóm tắt kết quả thực nghiệm (Mục 14, 17, 20, 22, 23)

### So sánh Baseline vs Fine-tuned ALBERT (Test split — `results.csv`)

| Model | Max Length | Validation EM | Validation F1 | Test EM | Test F1 | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Keyword Overlap)** | — | — | — | 7.00 | 13.33 | 0.07 |
| **ALBERT** | 128 | 58.00 | 72.45 | 64.50 | 73.95 | 1893.99 |
| **ALBERT** | 256 | **65.00** | **75.48** | 67.50 | 79.32 | 1285.98 |
| **ALBERT** | 384 | 61.50 | 73.01 | **69.50** | **81.66** | 1029.39 |

- **Chọn mô hình theo Validation F1:** Cấu hình `max_length = 256` đạt Validation F1 cao nhất (`75.48`), vượt trội hoàn toàn so với heuristic Keyword Baseline (`EM = 7.00`, `F1 = 13.33`).
- **Context Length & Training Time:** Khi `max_length = 128`, các context dài bị cắt thành nhiều sliding windows hơn (`stride = 32`), làm tăng số lượng feature windows cần duyệt qua mỗi epoch nên thời gian huấn luyện thực tế lâu hơn so với `256` và `384`.
- **Answer Length (Experiment 5):** Nhóm câu trả lời 1 từ đạt EM/F1 cao (`70.83 / 78.17`), trong khi nhóm dài `7+` từ giảm mạnh xuống `38.46 / 60.39`, xác nhận dự đoán về độ khó xác định ranh giới với span dài.
- **Wrong Start vs Wrong End (Mục 25):** Trên tập test có `48` lỗi wrong start và `50` lỗi wrong end (`71` span không khớp nguyên vẹn gold). Chi tiết 15 ca lỗi điển hình được phân tích trong [error_analysis.md](file:///d:/NLP/Lab10/error_analysis.md).

---

## 4. Hướng dẫn chạy

1. **Kiểm tra tokenization & span alignment:**
   Chạy `Lab10/tokenizer_alignment.ipynb` (đã thực thi và kiểm chứng trực tiếp trên `albert/albert-base-v2`).
2. **Xem và chạy lại toàn bộ thực nghiệm QA:**
   Mở `Lab10/qa_finetuning.ipynb`. Đặt `REUSE_CHECKPOINTS = True` để tái sử dụng checkpoint trong `checkpoints/lab10/` hoặc `False` để huấn luyện lại từ đầu.
3. **Chạy ứng dụng Document QA (Mục 27):**
   ```powershell
   python Lab10/document_qa.py "What is the capital of France?" --context "The capital of France is Paris."
   ```
   Hoặc đọc từ tệp văn bản UTF-8:
   ```powershell
   python Lab10/document_qa.py "Your question here" --file path/to/document.txt
   ```
   *(Lưu ý: Đây là hệ thống Extractive QA với context cung cấp sẵn, chưa có bước Retrieval/RAG.)*

---

## 5. AI Usage Statement (Mục 29)

- **What was generated / assisted:** Hỗ trợ kiểm tra cấu trúc code `qa_model.py`, `document_qa.py`, xử lý tương thích API `transformers 5.3.0` (`token_type_ids` không mặc định trả về), định dạng tài liệu báo cáo và đồ thị trực quan hóa.
- **What was modified:** Điều chỉnh đường dẫn thư mục dự án (`d:\NLP\Lab10`), cập nhật số luồng CPU (`torch.set_num_threads`) và hoàn thiện các phần giải thích, phân tích lỗi bằng tiếng Việt.
- **What was verified:** Đã tự kiểm tra toàn bộ phép tính tay trong `calculations.md`, chạy thực thi `tokenizer_alignment.ipynb` thành công với `offset_mapping` thực tế của `albert/albert-base-v2`, và đối chiếu các số liệu thực nghiệm với `results.csv`.
