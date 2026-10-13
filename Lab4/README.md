# LAB 04 - Phân loại văn bản

## 1. Mục tiêu

Lab này xây dựng một pipeline phân loại cảm xúc cho bộ dữ liệu IMDb. Mục tiêu là so sánh ảnh hưởng của biểu diễn văn bản và mô hình phân loại, không chỉ nhìn vào accuracy mà còn phân tích precision, recall, F1, confusion matrix và các lỗi thực tế.

Nhãn dữ liệu:

- `0`: đánh giá tiêu cực;
- `1`: đánh giá tích cực.

## 2. Cấu trúc thư mục

```text
Lab4/
├── W4.pdf
├── README.md
├── calculations.md
├── prediction.md
├── classification.ipynb
├── preprocessing.py
├── results.csv
├── class_distribution.png
├── confusion_matrix.png
├── correct_examples.csv
├── error_analysis.md
└── reflection.md
```

File dữ liệu `aclImdb_v1.tar.gz` được giữ ở local để chạy thí nghiệm nhưng không đưa lên GitHub vì kích thước lớn. Khi chạy trên máy khác, đặt file này vào cùng thư mục `Lab4` hoặc tải lại bộ IMDb tương ứng.

## 3. Cách chạy

Từ thư mục gốc của repository:

```powershell
cd Lab4
python preprocessing.py
```

Để xem notebook đã có kết quả:

```powershell
jupyter notebook classification.ipynb
```

Notebook đã được chạy và lưu output trực tiếp trong file `.ipynb`. Nếu muốn chạy lại từ đầu, chạy các cell theo thứ tự.

## 4. Kiểm tra dữ liệu và chống leakage

Pipeline thực hiện các bước sau:

1. Đọc dữ liệu IMDb và chuẩn hóa văn bản.
2. Kiểm tra số mẫu, số lớp, phân bố lớp, độ dài văn bản, missing values và duplicate texts.
3. Chia dữ liệu theo phương pháp phân tầng thành train/validation/test với tỷ lệ gần `60/20/20`.
4. Chỉ fit TF-IDF và Word2Vec trên tập train.
5. Dùng validation để so sánh/chọn mô hình.
6. Chỉ đánh giá test sau khi đã chọn mô hình.

Như vậy, vocabulary, IDF và embedding của validation/test không được dùng trong quá trình học. Đây là cách tránh data leakage.

## 5. Các thí nghiệm

- TF-IDF unigram + Logistic Regression.
- Average Word2Vec + Logistic Regression.
- TF-IDF unigram + bigram.
- TF-IDF + Linear SVM.
- So sánh threshold `0.5` và `0.8`.
- Confusion matrix và phân tích ví dụ dự đoán đúng/sai.
- Kiểm tra các feature có trọng số dương/âm lớn nhất.
- Thí nghiệm word order với `not good` và `good not`.

## 6. Kết quả thực nghiệm trên tập test

| Biểu diễn | Mô hình | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| TF-IDF unigram | Logistic Regression | 0.8866 | 0.8825 | 0.8928 | 0.8877 |
| TF-IDF unigram + bigram | Logistic Regression | 0.8954 | 0.8892 | 0.9042 | 0.8966 |
| TF-IDF unigram | Linear SVM | 0.8770 | 0.8827 | 0.8705 | 0.8765 |
| Average Word2Vec | Logistic Regression | 0.8499 | 0.8532 | 0.8464 | 0.8498 |

Trong lần chạy chính thức, TF-IDF unigram + bigram có kết quả cao nhất. Bigram giúp mô hình nhận diện các cụm có ý nghĩa như `not good`. Average Word2Vec thấp hơn vì phép lấy trung bình làm mất thứ tự từ và làm yếu thông tin phủ định.

## 7. Các file kết quả

- `classification.ipynb`: toàn bộ pipeline, bảng kết quả và output của từng cell.
- `preprocessing.py`: mã nguồn đọc dữ liệu, chia tập, huấn luyện và đánh giá.
- `results.csv`: kết quả các mô hình.
- `class_distribution.png`: biểu đồ phân bố lớp.
- `confusion_matrix.png`: ma trận nhầm lẫn của mô hình cơ sở.
- `correct_examples.csv`: các ví dụ dự đoán đúng.
- `error_analysis.md`: phân tích các ví dụ dự đoán sai.
- `calculations.md`: các phép tính sigmoid, threshold, precision, recall, F1 và accuracy trap.
- `prediction.md`: dự đoán trước khi chạy thí nghiệm.
- `reflection.md`: nhận xét về TF-IDF, Word2Vec, bigram và word order.

## 8. Tuyên bố sử dụng AI

- Công cụ: OpenAI Codex.
- Mục đích: hỗ trợ debug, viết tài liệu, trực quan hóa, tiền xử lý và sử dụng sklearn/gensim.
- Phần được sinh: khung pipeline và mã nguồn thí nghiệm ban đầu.
- Phần đã chỉnh sửa: đường dẫn, tham số, nội dung phân tích và kiểm tra kết quả.
- Kiểm chứng: notebook và script đã được chạy trên bộ dữ liệu IMDb local; các kết quả được lưu trong thư mục này.
