"""Core co-occurrence embedding implementation for LAB 03.

This module intentionally contains only the reusable implementation. Student
calculations, predictions, and interpretations belong in the report files.
"""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Iterable, List, Mapping, Sequence, Tuple


def build_vocabulary(sentences: Iterable[Sequence[str]], min_count: int = 1) -> Tuple[List[str], Mapping[str, int]]:
    """Return deterministic vocabulary and token-to-column index mapping."""
    counts = Counter(token for sentence in sentences for token in sentence)
    vocab = sorted(token for token, count in counts.items() if count >= min_count)
    return vocab, {token: index for index, token in enumerate(vocab)}


def tokenize(text: str) -> List[str]:
    """Provide a deliberately small tokenizer for the toy-corpus checks."""
    return text.lower().split()


def build_cooccurrence_matrix(
    sentences: Iterable[Sequence[str]],
    vocabulary: Sequence[str],
    window: int = 1,
) -> List[List[int]]:
    """Build a word-context count matrix with symmetric left/right context."""
    if window < 1:
        raise ValueError("window must be >= 1")
    index = {token: i for i, token in enumerate(vocabulary)}
    matrix = [[0 for _ in vocabulary] for _ in vocabulary]
    for sentence in sentences:
        for center, token in enumerate(sentence):
            if token not in index:
                continue
            target_i = index[token]
            start = max(0, center - window)
            stop = min(len(sentence), center + window + 1)
            for context in sentence[start:stop]:
                if context != token and context in index:
                    matrix[target_i][index[context]] += 1
    return matrix


def cosine_similarity(x: Sequence[float], y: Sequence[float]) -> float:
    """Return cosine similarity; zero is returned for a zero vector."""
    if len(x) != len(y):
        raise ValueError("vectors must have the same length")
    dot = sum(a * b for a, b in zip(x, y))
    norm_x = sqrt(sum(a * a for a in x))
    norm_y = sqrt(sum(b * b for b in y))
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return dot / (norm_x * norm_y)


def most_similar(
    word: str,
    matrix: Sequence[Sequence[float]],
    vocabulary: Sequence[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return the top-k nearest rows, excluding the query word itself."""
    if word not in vocabulary:
        raise KeyError(f"unknown word: {word}")
    if top_k < 1:
        return []
    query_i = vocabulary.index(word)
    scored = [
        (candidate, cosine_similarity(matrix[query_i], row))
        for candidate, row in zip(vocabulary, matrix)
        if candidate != word
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


def demo() -> None:
    sentences = [
        "the cat eats fish".split(),
        "the cat likes milk".split(),
        "the dog eats meat".split(),
        "the dog likes fish".split(),
    ]
    vocab, _ = build_vocabulary(sentences)
    matrix = build_cooccurrence_matrix(sentences, vocab, window=1)
    print(f"vocabulary_size={len(vocab)}")
    print(f"matrix_shape=({len(matrix)}, {len(matrix[0])})")
    print(f"non_zero_values={sum(value != 0 for row in matrix for value in row)}")
    print("Use most_similar(...) in the notebook after completing prediction.md.")


def run_self_checks() -> None:
    """Run implementation checks without producing report conclusions."""
    sentences = [tokenize("the cat eats fish"), tokenize("the dog eats fish")]
    vocab, index = build_vocabulary(sentences)
    assert vocab == sorted(set(token for sentence in sentences for token in sentence))
    assert index["cat"] < len(vocab)
    matrix = build_cooccurrence_matrix(sentences, vocab, window=1)
    assert len(matrix) == len(vocab) and all(len(row) == len(vocab) for row in matrix)
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    neighbours = most_similar("cat", matrix, vocab, top_k=2)
    assert len(neighbours) == 2 and all(word != "cat" for word, _ in neighbours)
    print("[ok] vocabulary, co-occurrence, cosine, and nearest-neighbour checks passed")


if __name__ == "__main__":
    run_self_checks()
    demo()
