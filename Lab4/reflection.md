# Nhận xét và phản ánh

TF-IDF và Average Word2Vec lưu giữ những loại thông tin khác nhau. TF-IDF là biểu diễn sparse và thiên về từ vựng: nó giữ các từ hoặc cụm từ xuất hiện và tăng trọng số cho term có khả năng phân biệt văn bản. Average Word2Vec là biểu diễn dense và thiên về ngữ nghĩa: các từ liên quan có thể có vector gần nhau, nhưng phép lấy trung bình làm mất thứ tự từ và nhiều thông tin về cách các từ kết hợp.

Vì vậy, cụm `not good` có thể bị biểu diễn không chính xác khi vector của `not` và `good` bị lấy trung bình. Kết quả IMDb cho thấy TF-IDF phù hợp vì các từ và cụm từ cảm xúc là tín hiệu mạnh. Bigram có thể nhận diện các biểu thức như `not good`, nhưng làm không gian đặc trưng lớn hơn và tăng thời gian tính toán. Embedding tĩnh có thể khái quát các từ liên quan, nhưng không hiểu nghĩa theo ngữ cảnh: từ `bank` chỉ có một vector.

Bài học kỹ thuật quan trọng nhất là phải chia dữ liệu trước khi fit. Trong bài làm này, TF-IDF chỉ được fit trên train; validation và test chỉ được transform sau đó. Nhờ vậy, vocabulary và IDF của test không bị rò rỉ vào quá trình huấn luyện. Bước tiếp theo là dùng RNN/LSTM hoặc Transformer để mô hình hóa thứ tự từ và ngữ cảnh.
