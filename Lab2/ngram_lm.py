# -*- coding: utf-8 -*-
"""
LAB 02 — Language Models (N-gram Language Models, Smoothing and Perplexity)
Module: ngram_lm.py

Triển khai n-gram language model từ đầu (không sử dụng thư viện n-gram có sẵn):
- build_vocabulary
- count_ngrams
- NGramLanguageModel (hỗ trợ n=1, 2, 3)
- MLE & Laplace (Add-1) Smoothing
- Log probability để chống Numerical Underflow
- Perplexity computation
- Next-word prediction & Sentence ranking
"""

import math
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Union


def simple_tokenize(text: str) -> List[str]:
    """
    Tiền xử lý và tách từ cơ bản:
    Chuyển chữ thường, giữ các từ cấu tạo từ chữ cái và số.
    """
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def build_vocabulary(tokenized_sentences: List[Union[str, List[str]]], min_freq: int = 1) -> List[str]:
    """
    Xây dựng tập từ vựng từ danh sách các câu (hỗ trợ cả chuỗi hoặc danh sách tokens).
    Trả về danh sách từ đã sắp xếp theo thứ tự bảng chữ cái.
    """
    counts = Counter()
    for sent in tokenized_sentences:
        tokens = simple_tokenize(sent) if isinstance(sent, str) else sent
        counts.update(tokens)
    return sorted([w for w, c in counts.items() if c >= min_freq])


def count_ngrams(tokenized_sentences: List[List[str]], n: int, pad: bool = True) -> Tuple[Counter, Counter]:
    """
    Đếm số lần xuất hiện của n-grams và (n-1)-grams (context).
    Nếu pad=True, thêm ký tự đặc biệt <s> và </s> để xử lý ranh giới câu.
    """
    ngram_counts = Counter()
    context_counts = Counter()

    for sent in tokenized_sentences:
        if pad:
            tokens = ["<s>"] * (n - 1) + sent + ["</s>"]
        else:
            tokens = sent

        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            context = tuple(tokens[i : i + n - 1])
            ngram_counts[ngram] += 1
            context_counts[context] += 1

    return ngram_counts, context_counts


def train_unigram(corpus, smoothing=None):
    return NGramLanguageModel(n=1, smoothing=smoothing).fit(corpus)


def train_bigram(corpus, smoothing=None):
    return NGramLanguageModel(n=2, smoothing=smoothing).fit(corpus)


def train_trigram(corpus, smoothing=None):
    return NGramLanguageModel(n=3, smoothing=smoothing).fit(corpus)


def probability(model, context, word):
    return model.probability(context, word)


def sentence_probability(model, sentence):
    return model.sentence_probability(sentence)


def sentence_log_probability(model, sentence):
    return model.sentence_log_probability(sentence)


class NGramLanguageModel:
    """
    Mô hình ngôn ngữ N-gram tổng quát hỗ trợ n=1 (Unigram), n=2 (Bigram), n=3 (Trigram).
    Hỗ trợ 2 chế độ:
    - smoothing=None (Maximum Likelihood Estimation - MLE)
    - smoothing='laplace' (Laplace Add-1 Smoothing)
    """

    def __init__(self, n: int = 2, smoothing: Optional[str] = "laplace"):
        if n not in [1, 2, 3]:
            raise ValueError("Chỉ hỗ trợ n = 1 (Unigram), 2 (Bigram), 3 (Trigram)")
        self.n = n
        self.smoothing = smoothing
        self.vocabulary: set = set()
        self.V: int = 0
        self.total_tokens: int = 0
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()

    def fit(self, tokenized_sentences: List[Union[str, List[str]]]):
        """
        Huấn luyện mô hình N-gram trên tập câu (hỗ trợ cả câu dạng chuỗi hoặc list tokens).
        """
        processed_sentences = []
        for s in tokenized_sentences:
            if isinstance(s, str):
                processed_sentences.append(simple_tokenize(s))
            else:
                processed_sentences.append(s)

        self.vocabulary = set(build_vocabulary(processed_sentences))
        self.vocabulary.add("</s>")
        self.V = len(self.vocabulary)

        if self.n == 1:
            self.ngram_counts = Counter()
            self.total_tokens = 0
            for sent in processed_sentences:
                tokens = sent + ["</s>"]
                self.ngram_counts.update(tokens)
                self.total_tokens += len(tokens)
        else:
            self.ngram_counts, self.context_counts = count_ngrams(processed_sentences, self.n, pad=True)

        return self

    def probability(self, context: Union[str, Tuple[str, ...]], word: str) -> float:
        """
        Tính xác suất có điều kiện P(word | context).
        """
        if isinstance(context, str):
            context = (context,) if context else ()
        elif isinstance(context, list):
            context = tuple(context)

        # Chuẩn hóa context về độ dài đúng n-1
        expected_len = self.n - 1
        if len(context) > expected_len:
            context = context[-expected_len:]
        elif len(context) < expected_len:
            context = ("<s>",) * (expected_len - len(context)) + context

        if self.n == 1:
            count_w = self.ngram_counts[word]
            if self.smoothing == "laplace":
                return (count_w + 1) / (self.total_tokens + self.V)
            else:
                return count_w / self.total_tokens if self.total_tokens > 0 else 0.0

        ngram = context + (word,)
        count_ngram = self.ngram_counts[ngram]
        count_ctx = self.context_counts[context]

        if self.smoothing == "laplace":
            return (count_ngram + 1) / (count_ctx + self.V)
        else:
            if count_ctx == 0:
                return 0.0
            return count_ngram / count_ctx

    def sentence_log_probability(self, sentence: Union[str, List[str]]) -> float:
        """
        Tính log xác suất của câu bằng cách cộng dồn log P(w | context)
        để tránh hiện tượng Numerical Underflow.
        """
        if isinstance(sentence, str):
            tokens = simple_tokenize(sentence)
        else:
            tokens = sentence

        if not tokens:
            return -math.inf

        tokens = tokens + ["</s>"]
        pad_len = self.n - 1
        padded_tokens = ["<s>"] * pad_len + tokens

        log_prob = 0.0
        for i in range(pad_len, len(padded_tokens)):
            w = padded_tokens[i]
            ctx = tuple(padded_tokens[i - pad_len : i])
            p = self.probability(ctx, w)
            if p <= 0.0:
                return -math.inf
            log_prob += math.log(p)

        return log_prob

    def sentence_probability(self, sentence: Union[str, List[str]]) -> float:
        """
        Tính xác suất P(S) = exp(log P(S)).
        """
        log_p = self.sentence_log_probability(sentence)
        if math.isinf(log_p) and log_p < 0:
            return 0.0
        return math.exp(log_p)

    def perplexity(self, sentences: List[Union[str, List[str]]]) -> float:
        """
        Tính Perplexity trên danh sách các câu:
        PPL = exp( - (1 / N) * sum(log P(w | context)) )
        Trong đó N là tổng số token (bao gồm cả token kết thúc </s>).
        """
        total_log_prob = 0.0
        total_words = 0

        for sent in sentences:
            if isinstance(sent, str):
                tokens = simple_tokenize(sent)
            else:
                tokens = sent

            if not tokens:
                continue

            tokens = tokens + ["</s>"]
            total_words += len(tokens)

            pad_len = self.n - 1
            padded_tokens = ["<s>"] * pad_len + tokens

            for i in range(pad_len, len(padded_tokens)):
                w = padded_tokens[i]
                ctx = tuple(padded_tokens[i - pad_len : i])
                p = self.probability(ctx, w)
                if p <= 0.0:
                    return math.inf
                total_log_prob += math.log(p)

        if total_words == 0:
            return math.inf

        return math.exp(-total_log_prob / total_words)

    def next_word_distribution(self, context: Union[str, Tuple[str, ...]], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Trả về top-K từ có xác suất cao nhất đứng sau context cho trước.
        Tối ưu hóa: chỉ duyệt qua các từ thực sự xuất hiện sau context (hoặc backoff nếu context rỗng/hiếm).
        """
        if isinstance(context, str):
            context = tuple(context.lower().split())
        elif isinstance(context, list):
            context = tuple(w.lower() for w in context)

        expected_len = self.n - 1
        if len(context) > expected_len:
            context = context[-expected_len:]
        elif len(context) < expected_len:
            context = ("<s>",) * (expected_len - len(context)) + context

        candidates = set()
        for ngram in self.ngram_counts.keys():
            if ngram[:-1] == context:
                candidates.add(ngram[-1])

        # Nếu không có từ nào xuất hiện sau context và context dài > 1, thử backoff 1 từ
        if not candidates and len(context) > 1:
            backoff_context = (context[-1],)
            for ngram in self.ngram_counts.keys():
                if len(ngram) >= 2 and ngram[-2:-1] == backoff_context:
                    candidates.add(ngram[-1])

        scores = []
        for w in candidates:
            p = self.probability(context, w)
            scores.append((w, p))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def rank_candidates(self, context: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Xếp hạng các câu/đoạn văn bản nối tiếp (continuation candidates) dựa trên xác suất câu.
        """
        results = []
        for cand in candidates:
            full_sentence = f"{context.strip()} {cand.strip()}"
            score = self.sentence_log_probability(full_sentence)
            results.append((cand, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ==========================================
# UNIT TESTS (Kiểm chứng theo Mục 7 của W2.pdf)
# ==========================================
def run_unit_tests():
    print("--- RUNNING UNIT TESTS FOR NGRAM_LM.PY ---")
    raw_corpus = [
        "the cat eats fish",
        "the cat likes fish",
        "the dog eats meat"
    ]
    tokenized = [simple_tokenize(s) for s in raw_corpus]

    # 1. Test build_vocabulary
    vocab = build_vocabulary(tokenized)
    expected_vocab = {"the", "cat", "eats", "fish", "likes", "dog", "meat"}
    assert set(vocab) == expected_vocab, f"Sai vocabulary: {vocab}"
    print("[PASS] build_vocabulary")

    # 2. Test Unigram Model (MLE)
    unigram_mle = NGramLanguageModel(n=1, smoothing=None)
    unigram_mle.fit(tokenized)
    # Tổng token tính cả </s> (3 token </s>): 12 + 3 = 15
    assert unigram_mle.ngram_counts["the"] == 3
    assert unigram_mle.ngram_counts["cat"] == 2
    assert unigram_mle.total_tokens == 15
    print("[PASS] Unigram MLE counts")

    # 3. Test Bigram Model (MLE)
    bigram_mle = NGramLanguageModel(n=2, smoothing=None)
    bigram_mle.fit(tokenized)
    p_cat_the = bigram_mle.probability("the", "cat")
    p_dog_the = bigram_mle.probability("the", "dog")
    assert abs(p_cat_the - 2/3) < 1e-6, f"P(cat|the) sai: {p_cat_the}"
    assert abs(p_dog_the - 1/3) < 1e-6, f"P(dog|the) sai: {p_dog_the}"
    print("[PASS] Bigram MLE probabilities")

    # 4. Test Zero Probability
    p_zero = bigram_mle.probability("dog", "likes")
    assert p_zero == 0.0
    prob_unseen_sent = bigram_mle.sentence_probability("the dog likes fish")
    assert prob_unseen_sent == 0.0
    print("[PASS] Bigram Zero Probability")

    # 5. Test Laplace Smoothing
    bigram_laplace = NGramLanguageModel(n=2, smoothing="laplace")
    bigram_laplace.fit(tokenized)
    # V gồm 7 từ + '</s>' = 8
    # P_laplace(likes|dog) = (0 + 1) / (1 + 8) = 1/9
    p_laplace_zero = bigram_laplace.probability("dog", "likes")
    assert abs(p_laplace_zero - 1/9) < 1e-6, f"Laplace sai: {p_laplace_zero}"
    print("[PASS] Laplace Smoothing")

    # 6. Test Sentence Log Probability & Perplexity
    ppl = bigram_laplace.perplexity(["the cat eats fish"])
    assert ppl > 0 and not math.isinf(ppl)
    print(f"[PASS] Perplexity: {ppl:.4f}")

    print(">>> ALL UNIT TESTS PASSED! <<<\n")


if __name__ == "__main__":
    run_unit_tests()
