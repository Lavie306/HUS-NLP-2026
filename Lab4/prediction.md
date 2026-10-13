# Dự đoán trước khi chạy thí nghiệm

| Dự đoán | Lý do | Mức độ tin cậy |
|---|---|---|
| TF-IDF + Logistic Regression sẽ là mô hình cơ sở tốt. | IMDb có nhiều từ thể hiện cảm xúc rõ như “excellent”, “boring”. | Cao |
| Average Word2Vec không chắc chắn tốt hơn TF-IDF. | Vector trung bình làm mất thứ tự từ và có thể làm yếu các từ mang cảm xúc. | Cao |
| Tăng vocabulary không chắc chắn làm accuracy tăng. | Từ hiếm và nhiễu có thể làm tăng phương sai và thời gian huấn luyện. | Cao |
| Bỏ stopwords không chắc chắn cải thiện kết quả. | Các từ như “not” và “never” quan trọng đối với cảm xúc. | Cao |
| Accuracy và F1 có thể đưa ra kết luận khác nhau. | Accuracy có thể che giấu mất cân bằng lớp, còn F1 cân bằng precision và recall. | Cao |

Các dự đoán trên được ghi lại trước khi xem kết quả thí nghiệm.
