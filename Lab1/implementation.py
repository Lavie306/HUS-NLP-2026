"""
LAB 01 — Phần 8: Core Implementation
Tự xây dựng phiên bản TF-IDF tối giản trên corpus nhỏ (không sử dụng trực tiếp TfidfVectorizer).
"""

import math
import numpy as np


# 8.2 Các hàm cần xây dựng
def build_vocabulary(corpus):
    """Xây dựng vocabulary sắp xếp theo thứ tự bảng chữ cái."""
    vocab = set()
    for doc in corpus:
        for word in doc.lower().split():
            vocab.add(word)
    return sorted(list(vocab))


def compute_counts(corpus, vocab):
    """Tính ma trận count vector c(t, d) kích thước (N, V)."""
    vocab_to_idx = {word: i for i, word in enumerate(vocab)}
    counts = np.zeros((len(corpus), len(vocab)))
    for i, doc in enumerate(corpus):
        for word in doc.lower().split():
            if word in vocab_to_idx:
                counts[i, vocab_to_idx[word]] += 1
    return counts


def compute_tf(counts):
    """
    Tính term frequency theo công thức Section 4.2:
    tf(t, d) = c(t, d) / tổng số term trong doc
    """
    doc_lengths = counts.sum(axis=1, keepdims=True)
    return counts / doc_lengths


def compute_idf(counts):
    """
    Tính inverse document frequency theo công thức Section 4.4:
    idf(t) = log(N / df(t))
    """
    N = counts.shape[0]
    df = np.count_nonzero(counts > 0, axis=0)
    return np.log(N / df)


def compute_tfidf(tf, idf):
    """Tính tfidf(t, d) = tf(t, d) * idf(t)."""
    return tf * idf


def cosine_similarity(x, y):
    """Tính cos(x, y) = (x . y) / (||x||_2 * ||y||_2)."""
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return float(np.dot(x, y) / (norm_x * norm_y))


# 8.3 & 8.4 Corpus kiểm thử và Unit tests
if __name__ == "__main__":
    # Corpus kiểm thử từ đề bài
    corpus = [
        "cat eats fish",
        "dog eats fish",
        "cat likes fish"
    ]

    # 1. Test build_vocabulary
    vocab = build_vocabulary(corpus)
    expected_vocab = ["cat", "dog", "eats", "fish", "likes"]
    assert vocab == expected_vocab, f"Sai vocabulary: {vocab}"

    # 2. Test compute_counts
    counts = compute_counts(corpus, vocab)
    assert counts[0, vocab.index("cat")] == 1.0
    assert counts[0, vocab.index("dog")] == 0.0

    # 3. Test compute_tf
    tf = compute_tf(counts)
    tf_cat = tf[0, vocab.index("cat")]
    assert abs(tf_cat - 1 / 3) < 1e-9, f"Sai TF của cat: {tf_cat}"
    assert abs(tf[0].sum() - 1.0) < 1e-9, "Tổng TF một doc phải bằng 1.0"

    # 4. Test compute_idf
    idf = compute_idf(counts)
    # df(fish) = 3 -> log(3/3) = 0.0
    # df(cat) = 2 -> log(3/2)
    assert abs(idf[vocab.index("fish")] - 0.0) < 1e-9, "IDF của fish phải bằng 0"
    assert abs(idf[vocab.index("cat")] - math.log(3 / 2)) < 1e-9

    # 5. Test compute_tfidf
    tfidf = compute_tfidf(tf, idf)
    assert abs(tfidf[0, vocab.index("fish")] - 0.0) < 1e-9
    assert abs(tfidf[0, vocab.index("cat")] - (1 / 3) * math.log(3 / 2)) < 1e-9

    # 6. Test cosine_similarity (Exercise 5)
    x = np.array([1, 1, 1])
    y = np.array([1, 1, 0])
    expected_cos = 2 / math.sqrt(6)
    assert abs(cosine_similarity(x, y) - expected_cos) < 1e-9

    print("All unit tests in Part 8 passed successfully!")
