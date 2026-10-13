# Phân tích lỗi

Các ví dụ trong `correct_examples.csv` và `misclassified.csv` được lấy từ mô hình cơ sở TF-IDF + Logistic Regression trên tập test IMDb.

## Ví dụ dự đoán đúng

Ba ví dụ đầu trong `correct_examples.csv` là các trường hợp mô hình dự đoán đúng. Những review này thường chứa nhiều từ cảm xúc rõ ràng và nhất quán, chẳng hạn các từ khen mạnh hoặc chê mạnh.

## Ví dụ dự đoán sai

Ba ví dụ đầu trong `misclassified.csv` là các trường hợp mô hình dự đoán sai. Có thể phân loại nguyên nhân theo các dạng sau:

| Dạng lỗi | Nguyên nhân có thể | Nội dung cần kiểm tra |
|---|---|---|
| Review tiêu cực bị dự đoán tích cực | Từ tích cực xuất hiện nhiều hoặc dữ liệu nhiễu | Review khen diễn viên nhưng kết luận chung lại là chê |
| Review tích cực bị dự đoán tiêu cực | Phủ định hoặc mỉa mai | Các cụm như `not good`, hoặc phần mở đầu tiêu cực nhưng kết luận tích cực |
| Review ngắn bị dự đoán sai | Ít bằng chứng ngữ nghĩa | Văn bản quá ngắn, có từ hiếm hoặc thiếu ngữ cảnh |

Confusion matrix cho thấy false positive là `1478`, false negative là `1333`; vì vậy cả hai loại lỗi đều đáng kể.

## Diễn giải mô hình

20 đặc trưng có trọng số dương lớn nhất và 20 đặc trưng có trọng số âm lớn nhất được lưu trong `experiment_summary.json`. Trọng số dương là bằng chứng cho lớp positive; trọng số âm là bằng chứng cho lớp negative. Các tính từ đánh giá mạnh thường có ý nghĩa, trong khi tên riêng, từ chỉ thể loại và lỗi chính tả có thể là shortcut phụ thuộc dataset.
