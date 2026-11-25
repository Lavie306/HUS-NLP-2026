# Error Analysis

Dùng ALBERT max length 256, được chọn bằng validation F1. Mỗi ví dụ dưới đây là một question test khác nhau. Counts dùng raw character boundaries; EM/F1 còn có bước normalization.

## Wrong start và Wrong end

Toàn bộ test có 48 wrong start, 50 wrong end và 71 span không khớp nguyên vẹn một gold span. Một prediction có thể sai cả hai đầu nên không cộng hai số đầu thành tổng số lỗi. Start/end được đối chiếu với các reference annotations, không suy ra trực tiếp từ EM.

## 5 wrong start

| Question | Gold | Prediction | Error type | Explanation |
| --- | --- | --- | --- | --- |
| What can increase the tension force on a load? | movable pulleys / connecting the same string multiple times to the same object through the use of a set-up that uses movable pulleys, / every string / connecting the same string multiple times to the same object through the use of a set-up that uses movable pulleys | By connecting the same string multiple times to the same object through the use of a set-up that uses movable pulleys | wrong_start | Prediction lấy thêm By ở đầu cụm giải thích. Biên cuối khớp một reference dài, nhưng không reference nào bắt đầu tại By nên EM bằng 0; phần lớn từ vẫn trùng nên F1 cao. |
| Besides the V&A's, whose collections are under the responsibility of conservators at the V&A? | V&A Museum of Childhood | Museum of Childhood | wrong_start | Model lấy đúng phần Museum of Childhood nhưng bỏ V&A ở đầu tên bảo tàng. Đây là partial answer do sai start; toàn bộ gold vẫn có trong window được chọn. |
| Where was the 1857 riot? | Buckland Valley / Buckland Valley near Bright | at Buckland Valley | wrong_start | Prediction thêm at trước Buckland Valley. Địa điểm đúng nhưng biên đầu rộng hơn reference, trong khi biên cuối vẫn khớp reference ngắn. |
| How long may the Amazon rainforest be threatened, according to some computer models? | though the 21st century | 21st century | wrong_start | Model chỉ lấy 21st century, bỏ though the ở đầu gold. Giữ nguyên cách viết though của dataset; F1 còn cao nhờ phần thời gian trùng nhau nhưng EM không đạt. |
| Where will the 2009 International Arts Fair be held? | Sage Gateshead Music and Arts Centre / Gateshead Music and Arts Centre | in the Norman Foster designed Sage Gateshead Music and Arts Centre | wrong_start | Prediction bao gồm cả in the Norman Foster designed trước tên địa điểm. Model tìm đúng cơ sở nhưng lấy thêm phần mô tả nên sai biên đầu. |

## 5 wrong end

| Question | Gold | Prediction | Error type | Explanation |
| --- | --- | --- | --- | --- |
| What was the Doritos customer Super Bowl ad campaign called? | Crash the Super Bowl | Crash the Super Bowl" contest | wrong_end | Prediction bắt đầu đúng Crash nhưng lấy thêm dấu nháy và từ contest sau tên chiến dịch. Dấu nháy được normalization bỏ đi, còn contest vẫn là từ thừa khiến EM bằng 0. |
| What was Frensh military presence at start of war? | no French regular army troops were stationed in North America / no French regular army troops were stationed in North America, / no French regular army | no French regular army troops were stationed in North America, and few British troops | wrong_end | Model lấy thêm and few British troops vào câu trả lời về quân Pháp. Start đúng nhưng end kéo dài sang ý nói về quân Anh; đây là lỗi lấy thừa phần cuối của câu. |
| The church states that, as Christians, they are aware that neither the way of what is righteous before God?" | the way of military action / persons who conscientiously oppose all war / military action | military action, nor the way of inaction | wrong_end | Prediction bắt đầu đúng military action nhưng ghép thêm nor the way of inaction. Câu có hai vế được so sánh, còn question và references khá mơ hồ; vẫn ghi lỗi end theo gold, không tự sửa annotation. |
| How much dust is blown out of the Sahara each year? | 182 million tons | 182 million | wrong_end | Model lấy đúng con số 182 million nhưng dừng trước đơn vị tons. Prediction còn trùng phần lớn nội dung nhưng chưa phải một lượng đầy đủ theo gold. |
| What is the name of the stadium in Miami that was considered? | Sun Life Stadium | Sun Life | wrong_end | Prediction giữ Sun Life nhưng bỏ Stadium ở cuối tên địa điểm. Gold đầy đủ trong cùng window nên không có bằng chứng đây là lỗi do truncation. |

## 5 wrong span

| Question | Gold | Prediction | Error type | Explanation |
| --- | --- | --- | --- | --- |
| By what name is that first Huguenot church known today? | L'Eglise du Saint-Esprit | L'Église française à la Nouvelle-Amsterdam | wrong_start + wrong_end | Context có tên lịch sử và tên hiện tại của cùng nhà thờ. Question hỏi known today nhưng model chọn tên lịch sử; đây là một span có vẻ hợp chủ đề nhưng không trả lời đúng mốc thời gian. |
| How many times did the Denver defense force Newton into turnovers? | three | seven | wrong_start + wrong_end | Trong cùng câu có seven times cho số lần sack và three turnovers cho số lần mất bóng. Model chọn seven, cho thấy prediction chưa phân biệt đúng hai quan hệ gắn với các con số. |
| What did CBS provide of the Super Bowl 50 broadcast for its website, some apps and media players? | digital streams of the game / digital streams | CBSSports.com | wrong_start + wrong_end | Question hỏi CBS đã cung cấp gì nhưng model chọn website CBSSports.com, thay vì digital streams. Website có liên quan trực tiếp đến broadcast nhưng không phải loại thông tin mà câu hỏi yêu cầu. |
| What team lost Super Bowl XXXIII? | Atlanta Falcons / the Atlanta Falcons / Falcons | Denver | wrong_start + wrong_end | Question hỏi đội thua Super Bowl XXXIII, gold là Atlanta Falcons. Model lấy Denver ở câu nói về một trận khác; nhiều tên đội và số trận gần nhau có thể là distractors, nhưng đây chỉ là giả thuyết từ context. |
| What was the score of the last game the Carolina Panthers played prior to the NFC Championship? | 31–24 / 31–24, | 49–15 | wrong_start + wrong_end | Context có tỉ số 31–24 của trận trước và 49–15 của NFC Championship. Question hỏi trận trước championship nhưng model chọn tỉ số championship, nên sai cả hai biên và nội dung. |

## Kết luận

Cần tách lỗi lấy sai nội dung khỏi lỗi chỉ thiếu/thừa một phần answer. F1 có thể còn cao khi prediction trùng nhiều từ, nhưng EM vẫn bằng 0. Các ví dụ ưu tiên chỉ sai start hoặc end; cột Error type ghi rõ nếu cả hai cùng sai.

Context, cách chia windows, độ dài answer và lexical overlap là các thông tin để phân tích. Không quy mọi lỗi cho truncation khi gold vẫn có mặt đầy đủ trong một window, và không coi một giả thuyết là nguyên nhân đã được chứng minh.
