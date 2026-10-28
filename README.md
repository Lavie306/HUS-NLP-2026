# Natural Language Processing & Applications (NLP 2026)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Coursework-success)
![Institution](https://img.shields.io/badge/Institution-VNU--HUS-informational)

---

## Tổng quan khóa học (Overview)

Kho lưu trữ này chứa toàn bộ mã nguồn, tài liệu thực hành, bài tập tính toán và báo cáo phân tích thực nghiệm của học phần **Xử lý ngôn ngữ tự nhiên và ứng dụng** (*Natural Language Processing & Applications* — Học kỳ I, Năm học 2026) tại **Trường Đại học Khoa học Tự nhiên — ĐHQGHN (VNU-HUS)**.

- **Giảng viên / TA:** Phạm Ngọc Hải  
- **Sinh viên thực hiện:** Nguyễn Trọng Thành  

Chương trình học được thiết kế theo lộ trình tiến hóa tự nhiên của ngành NLP: từ các biểu diễn từ vựng rời rạc/thưa cổ điển (Sparse Representations: Bag-of-Words, TF-IDF, N-gram) đến các biểu diễn phân bố dày (Dense Embeddings: Word2Vec), bài toán phân loại văn bản tĩnh (Text Classification), và chuyển dịch sang mô hình hóa chuỗi tuần tự theo thời gian (Sequential Modeling: RNN, LSTM, GRU), tạo tiền đề trực tiếp mở đường sang cơ chế Chú ý (**Attention Mechanism — LAB 06**) và kiến trúc **Transformer (LAB 07)**.

---

## Cấu trúc tổng thể của kho lưu trữ (Repository Structure)

```text
NLP/
├── README.MD                   # Tài liệu tổng quan toàn bộ kho lưu trữ
├── requirements.txt            # Danh sách thư viện phụ thuộc của dự án
├── .gitignore                  # Cấu hình bỏ qua file tạm, cache và dữ liệu lớn
│
├── Lab1/                       # Lab 01: From Text Processing to Search
│   ├── W1.pdf                  # Đề bài và đặc tả thực hành Lab 1
│   ├── c4-train...-30K.json    # Tập ngữ liệu 30.000 documents trích từ C4
│   ├── implementation.py       # Thuật toán cốt lõi: Vocabulary, TF, IDF, TF-IDF, Cosine
│   ├── experiments.ipynb       # Thực nghiệm phân tích ma trận thưa trên 30K văn bản
│   ├── Calculations.pdf        # Lời giải chi tiết bài toán tính tay
│   ├── Prediction.pdf          # Dự đoán giả thuyết trước khi chạy thực nghiệm
│   ├── results.csv             # Kết quả đánh giá retrieval (Precision@5, Recall@5, MRR)
│   └── Reflection.pdf          # Báo cáo phản tư, phân tích Vocabulary Mismatch Problem
│
├── Lab2/                       # Lab 02: Language Models (N-gram, Smoothing & Perplexity)
│   ├── W2.pdf                  # Đề bài và đặc tả thực hành Lab 2
│   ├── README.md               # Báo cáo chi tiết riêng cho Lab 2
│   ├── calculations.pdf        # Tính toán xác suất (Unigram, Bigram, PPL, Laplace)
│   ├── prediction.pdf          # 5 dự đoán trước thực nghiệm
│   ├── ngram_lm.py             # Cốt lõi: NGramLanguageModel, MLE, Laplace, Log-prob
│   ├── experiments.ipynb       # Thực nghiệm C4: Thống kê Zipf, PPL, Next-Word, Ranking
│   ├── results.csv             # Bảng tổng hợp Perplexity trên Train, Validation, Test
│   ├── error_analysis.pdf      # Phân tích các ca dự đoán đúng & sai
│   └── reflection.pdf          # Báo cáo phản ánh học tập và giới hạn của N-gram
│
├── Lab3/                       # Lab 03: Word Representations and Dense Embeddings
│   ├── W3.pdf                  # Đề bài và đặc tả thực hành Lab 3
│   ├── README.md               # Báo cáo chi tiết riêng cho Lab 3
│   ├── Calculations.pdf        # Lời giải tính tay Co-occurrence matrix, Cosine, Analogy
│   ├── prediction.md           # Dự đoán thực nghiệm theo Distributional Hypothesis
│   ├── cooccurrence.py         # Triển khai Co-occurrence matrix, Cosine Similarity
│   ├── experiments.ipynb       # Thực nghiệm Word2Vec hyperparameter sweep & Semantic Search
│   ├── word_embedding.ipynb    # Phân tích chuyển dịch từ Sparse sang Dense Embeddings
│   ├── results.csv             # Bảng tổng hợp số liệu đo đạc thực nghiệm
│   ├── error_analysis.md       # Phân tích ca đúng/sai kèm corpus evidence
│   └── reflection.md           # So sánh 4 dạng biểu diễn và phân tích hiện tượng đa nghĩa
│
├── Lab4/                       # Lab 04: Text Classification & Sentiment Analysis
│   ├── W4.pdf                  # Đề bài và đặc tả thực hành Lab 4
│   ├── README.md               # Báo cáo chi tiết riêng cho Lab 4
│   ├── calculations.md         # Tính toán Sigmoid, Decision Threshold, Precision, Recall, F1
│   ├── prediction.md           # Dự đoán trước thực nghiệm phân loại cảm xúc
│   ├── preprocessing.py        # Pipeline chuẩn bị dữ liệu IMDb, chống Data Leakage
│   ├── classification.ipynb    # Thực nghiệm phân loại: TF-IDF vs Avg Word2Vec + LR/SVM
│   ├── results.csv             # Kết quả định lượng các mô hình phân loại
│   ├── class_distribution.png  # Biểu đồ phân bố lớp cân bằng của IMDb
│   ├── confusion_matrix.png    # Ma trận nhầm lẫn của mô hình baseline
│   ├── correct_examples.csv    # Tập mẫu các đánh giá dự đoán đúng
│   ├── error_analysis.md       # Phân tích chuyên sâu các ca phân loại sai
│   └── reflection.md           # Nhận xét trade-off biểu diễn tĩnh và hiện tượng mất thứ tự từ
│
├── Lab5/                       # Lab 05: Sequence Models: RNN, LSTM and GRU
│   ├── W5.pdf                  # Đề bài và đặc tả thực hành Lab 5
│   ├── README.md               # Báo cáo tổng quan, rubric checklist và hướng dẫn kiểm tra 3 phút
│   ├── calculations.md         # Tính toán từng bước Forward RNN, Information Flow, LSTM Gates
│   ├── prediction.md           # 5 dự đoán độc lập trước khi huấn luyện (Mục 10)
│   ├── rnn_cell.py             # Cài đặt SimpleRNNCell từ đầu, đối chiếu PyTorch nn.RNNCell
│   ├── run_experiments.py      # Pipeline thực nghiệm hoàn chỉnh, tự động và tái lập
│   ├── sequence_classification.ipynb # Notebook thực nghiệm lưu sẵn output 8/8 code cells
│   ├── results.csv             # Bảng kết quả định lượng Baseline vs RNN vs LSTM vs GRU
│   ├── experiment_details.json # Metadata thống kê 50k IMDb, learning history, probes
│   ├── learning_curves.png     # Biểu đồ đường cong học tập Train/Val Loss qua các epoch
│   ├── confusion_matrix.png    # Ma trận nhầm lẫn của mô hình GRU trên held-out test set
│   ├── sequence_lengths.png    # Histogram phân bố độ dài 50.000 bài đánh giá IMDb
│   ├── length_tradeoff.png     # Biểu đồ đánh đổi giữa độ dài chuỗi, F1 và thời gian huấn luyện
│   ├── error_analysis.md       # Phân tích lỗi (negation, contrast, truncation, ambiguity)
│   ├── reflection.md           # Báo cáo phản ánh chuyên sâu và mạch nối sang Attention
│   ├── requirements.txt        # Danh sách thư viện cần thiết cho Lab 5
│   └── gru_model.pt            # Trọng số mô hình GRU tốt nhất phục vụ suy luận trực tiếp
│
└── Lab6/                       # Lab 06: Attention Mechanism: From Recurrent State to Context-Aware
    ├── W6.pdf                  # Đề bài và đặc tả thực hành Lab 6
    ├── README.md               # Báo cáo tổng quan, rubric checklist và hướng dẫn chạy
    ├── calculations.md         # Lời giải tính tay: Scores, Softmax, Context, Scaling, Temperature
    ├── prediction.md           # 5 dự đoán trước thực nghiệm kèm lý giải và đối chiếu thực nghiệm
    ├── attention_numpy.py      # Cài đặt Attention từ đầu bằng NumPy (stable softmax, scaled attn)
    ├── self_attention.ipynb    # Notebook thực nghiệm hoàn chỉnh lưu sẵn output tất cả cells
    ├── results.csv             # Bảng kết quả định lượng so sánh LSTM (Lab 05) vs Self-Attention (Lab 06)
    ├── attention_model.pt      # Trọng số mô hình PyTorch đã huấn luyện
    ├── experiment_details.json # Metadata chi tiết về kết quả thí nghiệm và lịch sử loss
    ├── attention_heatmap.png   # Trực quan hóa ma trận Attention (Heatmap) câu ví dụ 'cat'
    ├── confusion_matrix.png    # Ma trận nhầm lẫn của mô hình Self-Attention trên tập test IMDb
    ├── query_attention.png     # Biểu đồ phân phối attention theo Query (Sentiment vs Topic)
    ├── temperature_experiment.png # Biểu đồ khảo sát phân phối và entropy theo nhiệt độ T
    ├── sequence_length_cost.png# Biểu đồ phân tích chi phí tính toán bậc hai O(T^2)
    ├── error_analysis.md       # Phân tích lỗi (3 mẫu đúng, 3 mẫu sai trên tập test IMDb)
    ├── reflection.md           # Báo cáo phản ánh chuyên sâu và cầu nối sang Transformer (Lab 07)
    └── requirements.txt        # Danh sách thư viện cần thiết cho Lab 6
```

---

## Tóm tắt nội dung các bài thực hành

### Lab 1 — Tách từ và biểu diễn văn bản bằng vector thưa (Sparse Representations)
- **Chủ đề:** Text Processing, Sparse Vector Representations, TF-IDF và Document Search Engine.
- **Nội dung chính:**
  - Thực hiện chuẩn hóa văn bản, tokenization, loại bỏ stopwords.
  - Tự cài đặt từ đầu (*from scratch*) các giải thuật cốt lõi: `Vocabulary Builder`, `Count Vector`, `Term Frequency (TF)`, `Inverse Document Frequency (IDF)`, `TF-IDF`, và `Cosine Similarity`.
  - Khảo sát thực nghiệm trên kho ngữ liệu 30.000 văn bản tiếng Anh trích từ tập C4.
  - Xây dựng Document Search Engine, đánh giá các chỉ số `Precision@5`, `Recall@5`, `MRR` và phân tích hiện tượng **Vocabulary Mismatch Problem**.

### Lab 2 — Mô hình ngôn ngữ N-gram, Làm mịn và Perplexity (Language Modeling)
- **Chủ đề:** N-gram Language Models, Zero-Frequency Problem, Laplace Smoothing, Perplexity.
- **Nội dung chính:**
  - Triển khai mô hình Unigram, Bigram, Trigram theo thiết kế hướng đối tượng OOP (`NGramLanguageModel`).
  - Giải quyết bài toán xác suất bằng không với kỹ thuật **Laplace (Add-1) Smoothing** và xử lý chống tràn số dưới (*numerical underflow*) bằng thang Log-probability.
  - Khảo sát quy luật Zipf trên 50.000 câu văn bản C4.
  - Đánh giá định lượng mô hình ngôn ngữ thông qua chỉ số **Perplexity (PPL)** trên 3 tập dữ liệu độc lập (Train, Validation, Test).
  - Triển khai 2 ứng dụng thực tế: **Dự đoán từ tiếp theo (Next-Word Prediction)** và **Xếp hạng câu (Sentence Ranking)**.

### Lab 3 — Biểu diễn từ và Không gian vector dày (Dense Word Embeddings)
- **Chủ đề:** Distributional Hypothesis, Co-occurrence Matrix, Word2Vec (Skip-gram / CBOW), GloVe, Vector Arithmetic.
- **Nội dung chính:**
  - Hiện thực hóa giả thuyết phân bố (*Distributional Hypothesis*): *"You shall know a word by the company it keeps"*.
  - Tự xây dựng ma trận đồng xuất hiện ngữ cảnh (Co-occurrence Matrix) với sliding window $k=1$.
  - Huấn luyện mô hình **Word2Vec** qua thư viện Gensim với sweep siêu tham số: Context Window $\in \{2, 5, 10\}$ và Vector Dimension $\in \{50, 100, 300\}$.
  - Khảo sát các phép toán vector ngữ nghĩa (Word Analogy: $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$).
  - Xây dựng hệ thống tìm kiếm ngữ nghĩa (**Semantic Search**) vượt qua giới hạn so khớp từ khóa chính xác.

### Lab 4 — Phân loại văn bản và Cảm xúc (Text Classification & Sentiment Analysis)
- **Chủ đề:** Sentiment Classification, IMDb Movie Reviews, Representation Comparison, Data Leakage Prevention.
- **Nội dung chính:**
  - Xây dựng pipeline phân loại cảm xúc nhị phân trên bộ dữ liệu 50.000 bài đánh giá phim IMDb.
  - Tuân thủ nguyên tắc chống rò rỉ dữ liệu (*Split-before-fit*): chỉ trích xuất từ vựng và IDF trên tập Train, sau đó mới transform Validation và Test.
  - So sánh đối đầu giữa biểu diễn thưa (TF-IDF Unigram, Bigram) và biểu diễn dày (Average Word2Vec) khi kết hợp với Logistic Regression và Linear SVM.
  - Phân tích hiện tượng **mất thứ tự từ** của Average Word2Vec: `"not good"` và `"good not"` bị tính gộp như nhau do tính giao hoán của phép cộng vector.

### Lab 5 — Mô hình chuỗi tuần tự: RNN, LSTM và GRU (Sequential Modeling)
- **Chủ đề:** Sequence Classification, Recurrent Computation, Vanishing Gradient, Gated Architectures, Length Trade-offs.
- **Nội dung chính:**
  - Tự cài đặt **`SimpleRNNCell`** từ đầu ($h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$) và đối chiếu trùng khớp tuyệt đối với `torch.nn.RNNCell`.
  - Khảo sát toàn diện 50.000 bài đánh giá IMDb, phân tích phân bố độ dài lệch phải và vẽ biểu đồ histogram.
  - So sánh công bằng 3 kiến trúc mạng nơ-ron hồi quy: **RNN, LSTM, GRU** (cùng embedding 64, hidden 64, Adam lr=0.001, 2 epochs) với Baseline TF-IDF + Logistic Regression.
  - Khảo sát sự đánh đổi tính toán (*Length Trade-off*): tăng `max_length` từ 50 lên 200 giúp F1 tăng từ 0.56 lên 0.65 nhưng thời gian huấn luyện tăng gấp 15 lần.
  - Đo lường định tính bằng các probe: Order Sensitivity Probe (chứng minh GRU nhạy cảm với thứ tự từ hơn Bag-of-Words), Application Probe, Long-Range Position Probe.
  - Phân tích nguyên nhân lỗi (cắt cụt truncation, từ phủ định negation, cấu trúc tương phản contrast).
  - Làm rõ nút thắt cổ chai của mạng hồi quy (nén chuỗi dài vào 1 vector ẩn và tính toán tuần tự không thể song song hóa), tạo tiền đề chuyển dịch sang **Attention (LAB 06)** và **Transformer (LAB 07)**.

### Lab 6 — Cơ chế Attention: Từ Hidden State tới Biểu diễn nhạy ngữ cảnh (Attention Mechanism)
- **Chủ đề:** Attention Mechanism, Query/Key/Value, Scaled Dot-Product, Self-Attention, Temperature, O(T^2) Complexity.
- **Nội dung chính:**
  - Giải quyết bài toán **Information Bottleneck** của mạng hồi quy bằng cách cho phép mô hình kết nối trực tiếp mọi cặp vị trí $(i, j)$ qua cơ chế Attention.
  - Tự cài đặt từ đầu bằng **NumPy thuần** (`attention_numpy.py`): `stable_softmax` (trừ $\max(x)$ chống tràn số overflow), `softmax_with_temperature`, `entropy`, `scaled_dot_product_attention`, và `self_attention`.
  - Thực hiện bài tập tính tay toàn bộ quy trình: Attention Scores, Softmax weights, Context vector, Scaling chia $\sqrt{d_k}$, và tính toán ma trận Self-Attention với `['I', 'like', 'cats']`.
  - Khảo sát sự thay đổi phân phối Attention theo Query: Query Sentiment tập trung vào từ cảm xúc (`'good'`), Query Topic tập trung vào từ chủ đề (`'movie'`).
  - Trực quan hóa ma trận trọng số bằng Heatmap và làm rõ luận điểm quan trọng: **Attention weights là phân phối xác suất toán học, không phải là bằng chứng suy luận ngôn ngữ (Attention is Not Explanation).**
  - Huấn luyện bộ phân loại **Self-Attention Classifier** trên 50.000 bài đánh giá IMDb, so sánh trực tiếp với **LSTM (LAB 05)** trên cùng tập test cân bằng: Self-Attention đạt F1 **$0.6660$** (vượt trội so với LSTM $0.5406$) và tốc độ huấn luyện nhanh gấp **$8.1$ lần** ($3.42$s vs $27.93$s).
  - Phân tích chi phí tính toán và bộ nhớ bậc hai **$O(T^2)$ ('Attention không miễn phí')**: chỉ ra nguy cơ tràn bộ nhớ khi $T \ge 10,000$ và sự mất mát thông tin thứ tự từ nếu không có Positional Encoding, mở đường trực tiếp sang kiến trúc **Transformer (LAB 07)**.


---

## Hướng dẫn cài đặt và Chạy thực nghiệm (Quickstart)

### 1. Khởi tạo môi trường ảo
Khuyến nghị sử dụng Python 3.9 trở lên (đặc biệt Python 3.10 hoặc 3.11):
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt trên Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Hoặc kích hoạt trên Linux/macOS:
source venv/bin/activate
```

### 2. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### 3. Thực thi nhanh từng bài thực hành

#### Chạy Lab 1:
```bash
cd Lab1
python implementation.py
jupyter notebook experiments.ipynb
```

#### Chạy Lab 2:
```bash
cd Lab2
python ngram_lm.py
jupyter notebook experiments.ipynb
```

#### Chạy Lab 3:
```bash
cd Lab3
python cooccurrence.py
jupyter notebook experiments.ipynb
```

#### Chạy Lab 4:
```bash
cd Lab4
python preprocessing.py
jupyter notebook classification.ipynb
```

#### Chạy Lab 5:
```bash
cd Lab5
# 1. Kiểm tra SimpleRNNCell tự code với PyTorch:
python rnn_cell.py

# 2. Chạy toàn bộ pipeline huấn luyện, đánh giá và xuất kết quả:
python run_experiments.py

# 3. Mở Jupyter Notebook xem trực quan hóa kết quả đã nạp sẵn:
jupyter notebook sequence_classification.ipynb
```

#### Chạy Lab 6:
```bash
cd Lab6
# 1. Kiểm tra thuật toán Attention từ đầu bằng NumPy:
python attention_numpy.py

# 2. Chạy toàn bộ pipeline huấn luyện, đánh giá và xuất kết quả:
python run_lab6.py

# 3. Mở Jupyter Notebook xem trực quan hóa kết quả đã nạp sẵn:
jupyter notebook self_attention.ipynb
```


---

## Công nghệ & Thư viện sử dụng
- **Ngôn ngữ lập trình:** Python 3.9+ / 3.10 / 3.11
- **Deep Learning Framework:** PyTorch 2.0+ (Torch, Torch.nn, PackedSequence)
- **Học máy & NLP cổ điển:** scikit-learn, Gensim (Word2Vec), NLTK
- **Tính toán khoa học & Xử lý dữ liệu:** NumPy, SciPy, Pandas
- **Trực quan hóa:** Matplotlib, Seaborn
- **Môi trường thực nghiệm:** Jupyter Notebook, IPython

---
*Kho lưu trữ được cập nhật và kiểm thử tự động, tuân thủ các chuẩn mực khoa học và chính sách học thuật của Trường Đại học Khoa học Tự nhiên — ĐHQGHN.*
