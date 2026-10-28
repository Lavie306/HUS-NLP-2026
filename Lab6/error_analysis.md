# Error Analysis — Phân tích chi tiết 6 mẫu kiểm thử (IMDb Test Set)

Bảng dưới đây phân tích 3 mẫu dự đoán đúng và 3 mẫu dự đoán sai từ tập kiểm thử IMDb.
Xác suất $P(\text{pos})$ là xác suất mô hình dự đoán nhãn **Tích cực**.

> [!IMPORTANT]
> **Lưu ý quan trọng về tính nhân quả:** Phân phối trọng số attention chỉ phản ánh tương quan toán học trong quá trình tổng hợp vector context, **không phải là bằng chứng nhân quả** hay lời giải thích tuyệt đối rằng mô hình 'hiểu ngôn ngữ'. Việc xác định yếu tố ảnh hưởng chỉ mang tính giả thuyết (hypothesis).

| STT | Kết quả | Nhãn thật | Dự đoán | $P(\text{pos})$ | Độ dài tokens | Top tokens nhận Attention cao nhất | Yếu tố có thể ảnh hưởng (Giả thuyết) | Trích đoạn bài đánh giá |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| 1 | **ĐÚNG** | Tích cực | Tích cực | 0.7366 | 175 | `in`, `in`, `in` | Chuyển ý / Tương phản (Contrast), Cắt ngắn chuỗi (> 100 tokens) | this is a wonderful film, makes the 1950'S look beautifully stylish. Kim Novak is intriguing and compelling as a modern-day witch with one foot in Manhattan and another in infinity. All the supporting performances are terrific, from Jack Lemmon as her bother Nicky to Ernie Kovacs... |
| 2 | **ĐÚNG** | Tích cực | Tích cực | 0.8498 | 719 | `most`, `melodrama`, `and` | Phủ định (Negation), Cắt ngắn chuỗi (> 100 tokens) | Lee Chang-dong's exceptional "Secret Sunshine" is the single most emotionally ravaging experience of the year. It is an instantly sobering, brutally honest character piece on the reverberations of loss and a graceful memento mori that resonates with a striking density of thought,... |
| 3 | **ĐÚNG** | Tích cực | Tích cực | 0.6717 | 909 | `in`, `in`, `in` | Phủ định (Negation), Cắt ngắn chuỗi (> 100 tokens) | It is always a well-known, and important directorial device to set up the atmosphere of a film within the first 5 minutes. In the crucial opening scenes, the film should assert itself and make the viewers take notice and get interested in the rest of the film. Here, in "Mute Witn... |
| 4 | **SAI** | Tích cực | Tiêu cực | 0.4976 | 303 | `and`, `make`, `statement` | Phủ định (Negation), Chuyển ý / Tương phản (Contrast), Cắt ngắn chuỗi (> 100 tokens) | I won't go to a generalization, and say it's the best love story of all time, as some have said. That's fine, people feel very deeply about this film, you either love it I believe...or you simply hate it. I don't want to say, the best of all,because that is simply too 'broad' for... |
| 5 | **SAI** | Tích cực | Tiêu cực | 0.3271 | 95 | `movie`, `movie`, `andrew` | Phủ định (Negation), Chuyển ý / Tương phản (Contrast) | I've seen this movie after watching Paltrow's version. I've found that one a very good one, and I thought this would not be as good... but I was wrong: British version was far better and enjoyable! I found Jeremy Northam more "agreeable" than Mark Strong, but I can say that Stron... |
| 6 | **SAI** | Tích cực | Tiêu cực | 0.4712 | 259 | `in`, `heard`, `make` | Phủ định (Negation), Cắt ngắn chuỗi (> 100 tokens) | I had never heard of this film and only got it because I am a Martin Sheen fan. Now I am stunned as to why it did not receive the praise and recognition that it truly deserves. The four characters all make you feel for them, the father trying to assert his authority, the mother s... |

## Nhận xét phân tích lỗi

1. **Ảnh hưởng của Truncation (Cắt ngắn chuỗi):** Khi giới hạn độ dài chuỗi tại 100 tokens, các đánh giá dài có thể chứa tín hiệu cảm xúc quyết định (ví dụ: cú lật cốt truyện ở cuối review) ở phần sau nhưng bị loại bỏ hoàn toàn.
2. **Cấu trúc phủ định và đảo ý:** Self-attention cơ bản khi không có Positional Encoding sẽ coi các từ như một túi đặc trưng tương tác lẫn nhau. Một từ tích cực mạnh như `great` có thể nhận trọng số attention cao ngay cả khi đứng sau `not`.
3. **Attention không đồng nghĩa với giải thích nhân quả (Reasoning):** Một token nhận attention cao chỉ có nghĩa là nó đóng góp nhiều vào context vector qua phép nhân $A V$, chứ không đồng nghĩa rằng mô hình 'hiểu' ngữ pháp của từ đó.