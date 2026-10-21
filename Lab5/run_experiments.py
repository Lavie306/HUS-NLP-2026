"""Reproduce the IMDb sequence classification experiments for LAB 05.

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng - Học kỳ I, 2026.
Bao gồm:
- Dataset inspection (50,000 samples) và vẽ histogram sequence lengths.
- Baseline TF-IDF + Logistic Regression (LAB 04 benchmark).
- So sánh kiến trúc: RNN vs LSTM vs GRU (cùng siêu tham số, 100 tokens).
- Thử nghiệm độ dài chuỗi: max_length in {50, 100, 200} (GRU).
- Đánh giá định lượng: Accuracy, Precision, Recall, F1, Loss, Training time, Parameters.
- Learning curves (train/val loss qua epoch).
- Confusion matrix cho test set.
- Probes: Order sensitivity probe, Application probe, Long-range position probe.
- Xuất bản: results.csv, experiment_details.json, gru_model.pt, biểu đồ PNG, error_analysis.md.
"""

from __future__ import annotations

import csv
import html
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from rnn_cell import verify_against_torch


def find_data_path() -> Path:
    """Tìm đường dẫn tới file IMDB Dataset.csv."""
    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "imdb" / "IMDB Dataset.csv",
        Path(__file__).resolve().parent / "data" / "imdb" / "IMDB Dataset.csv",
        Path("D:/NLP/data/imdb/IMDB Dataset.csv"),
        Path(__file__).resolve().parent / "IMDB Dataset.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


@dataclass(frozen=True)
class Config:
    """Lưu trữ cấu hình toàn bộ thí nghiệm."""

    data_path: Path = find_data_path()
    output_path: Path = Path(__file__).resolve().parent
    seed: int = 42
    train_count: int = 5000
    validation_count: int = 1000
    test_count: int = 1000
    vocabulary_size: int = 12000
    embedding_size: int = 64
    hidden_size: int = 64
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 0.001
    lengths: tuple[int, ...] = (50, 100, 200)
    primary_length: int = 100
    pad_id: int = 0
    unknown_id: int = 1
    test_fraction: float = 0.2
    validation_fraction: float = 0.25
    max_workers: int = 0
    torch_threads: int = 4
    histogram_bins: int = 60
    plot_dpi: int = 140
    examples_per_outcome: int = 3
    error_snippet_length: int = 420


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MODEL_NAMES = ("RNN", "LSTM", "GRU")
LABEL_NAMES = ("negative", "positive")
LONG_RANGE_TEMPLATES = (
    "{cue} The movie had several scenes and a long story with many characters and turns.",
    "The movie had several scenes and a long story. {cue} It also had many characters and turns.",
    "The movie had several scenes and a long story with many characters and turns. {cue}",
)
LONG_RANGE_CUES = ("Ultimately it was excellent.", "Ultimately it was terrible.")
APPLICATION_REVIEWS = {
    "short": "I really enjoyed this movie.",
    "long": "The film begins slowly with several unnecessary scenes, but the acting becomes powerful and the ending is wonderful. I enjoyed the movie overall.",
    "negation": "I did not like this movie.",
    "contrast": "I thought the movie would be terrible, but it was surprisingly good.",
}
ORDER_PROBE_REVIEWS = (
    "this movie is not good",
    "this movie is good not",
    "the film is not bad",
    "the film is bad not",
)


def tokenize(text: str) -> list[str]:
    """Chuyển về chữ thường, loại bỏ HTML tags và tokenize bảo toàn từ phủ định."""
    normalized = HTML_TAG_PATTERN.sub(" ", html.unescape(str(text))).lower()
    return TOKEN_PATTERN.findall(normalized)


def inspect_dataset(frame: pd.DataFrame, config: Config) -> dict:
    """Thống kê toàn bộ dataset 50,000 dòng trước khi trích chọn tập mẫu."""
    print("-> Đang kiểm tra thống kê toàn bộ dataset IMDb...")
    lengths = frame["review"].fillna("").map(lambda value: len(tokenize(value)))
    summary = {
        "samples": int(len(frame)),
        "classes": frame["sentiment"].value_counts(dropna=False).to_dict(),
        "minimum_tokens": int(lengths.min()),
        "median_tokens": float(lengths.median()),
        "mean_tokens": float(lengths.mean()),
        "maximum_tokens": int(lengths.max()),
        "quantiles": {
            str(quantile): float(lengths.quantile(quantile))
            for quantile in (0.5, 0.75, 0.9, 0.95)
        },
        "duplicate_documents": int(frame["review"].duplicated().sum()),
        "missing_text": int(frame["review"].isna().sum()),
        "missing_labels": int(frame["sentiment"].isna().sum()),
    }

    # Vẽ histogram độ dài văn bản
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(
        lengths.clip(upper=lengths.quantile(0.99)),
        bins=config.histogram_bins,
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.8,
    )
    axis.set_xlabel("Số tokens trên mỗi review (clip tại phân vị 99%)", fontsize=11)
    axis.set_ylabel("Số lượng reviews", fontsize=11)
    axis.set_title("Phân bố độ dài chuỗi IMDb (Sequence Lengths Histogram)", fontsize=12)
    axis.grid(True, linestyle="--", alpha=0.5)
    figure.tight_layout()
    figure.savefig(config.output_path / "sequence_lengths.png", dpi=config.plot_dpi)
    plt.close(figure)
    print("   Đã lưu biểu đồ sequence_lengths.png")
    return summary


def prepare_splits(frame: pd.DataFrame, config: Config) -> dict[str, pd.DataFrame]:
    """Loại bỏ trùng lặp và chia tách tập dữ liệu train/val/test phân tầng cân bằng."""
    print("-> Đang chia tập dữ liệu train/val/test...")
    clean = frame.dropna(subset=["review", "sentiment"]).drop_duplicates(subset="review")
    train_validation, test = train_test_split(
        clean,
        test_size=config.test_fraction,
        stratify=clean["sentiment"],
        random_state=config.seed,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=config.validation_fraction,
        stratify=train_validation["sentiment"],
        random_state=config.seed,
    )
    counts = {
        "train": config.train_count,
        "validation": config.validation_count,
        "test": config.test_count,
    }
    sources = {"train": train, "validation": validation, "test": test}
    return {
        name: source.groupby("sentiment", group_keys=False)
        .sample(n=count // len(LABEL_NAMES), random_state=config.seed)
        .sample(frac=1, random_state=config.seed)
        .reset_index(drop=True)
        for name, source in sources.items()
        for count in (counts[name],)
    }


def build_vocabulary(texts: pd.Series, config: Config) -> dict[str, int]:
    """Xây dựng từ vựng CHỈ từ tập train để chống rò rỉ dữ liệu (data leakage)."""
    counts = Counter(token for text in texts for token in tokenize(text))
    # Dành 0 cho PAD, 1 cho UNK
    return {
        token: index
        for index, (token, _) in enumerate(
            counts.most_common(config.vocabulary_size - 2), start=2
        )
    }


def encode(
    text: str, vocabulary: dict[str, int], max_length: int, config: Config
) -> tuple[list[int], int]:
    """Mã hóa chuỗi token IDs, áp dụng padding hoặc truncation."""
    tokens = tokenize(text)
    ids = [vocabulary.get(token, config.unknown_id) for token in tokens[:max_length]]
    length = max(1, len(ids))
    padded_ids = ids + [config.pad_id] * (max_length - len(ids))
    return padded_ids, length


class ReviewDataset(Dataset):
    """PyTorch Dataset cho IMDb reviews."""

    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: dict[str, int],
        max_length: int,
        config: Config,
    ) -> None:
        encoded = [encode(text, vocabulary, max_length, config) for text in frame["review"]]
        self.tokens = torch.tensor([item[0] for item in encoded], dtype=torch.long)
        self.lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long)
        self.labels = torch.tensor(
            (frame["sentiment"] == LABEL_NAMES[1]).astype(int).to_numpy(),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.tokens[index], self.lengths[index], self.labels[index]


class SequenceClassifier(nn.Module):
    """Mô hình phân loại chuỗi tuần tự: Embedding -> Recurrent Layer -> Linear."""

    def __init__(self, architecture: str, vocabulary_size: int, config: Config) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size, config.embedding_size, padding_idx=config.pad_id
        )
        recurrent_type = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[architecture]
        self.recurrent = recurrent_type(
            config.embedding_size, config.hidden_size, batch_first=True
        )
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, tokens: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.recurrent(packed)
        # Với LSTM, state là (h_n, c_n); với RNN/GRU, state là h_n
        hidden = state[0] if isinstance(state, tuple) else state
        # Lấy hidden state lớp cuối cùng
        return self.classifier(hidden[-1]).squeeze(-1)


def score(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Tính các metric phân loại nhị phân đối với nhãn tích cực (positive)."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    model: SequenceClassifier, loader: DataLoader, criterion: nn.Module
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Đánh giá mô hình trên DataLoader không tính gradient."""
    model.eval()
    losses = []
    labels = []
    probabilities = []
    with torch.no_grad():
        for tokens, lengths, target in loader:
            logits = model(tokens, lengths)
            losses.append(float(criterion(logits, target)))
            labels.extend(target.numpy())
            probabilities.extend(torch.sigmoid(logits).numpy())
    labels_array = np.asarray(labels, dtype=int)
    probability_array = np.asarray(probabilities)
    predictions = (probability_array >= 0.5).astype(int)
    metrics = score(labels_array, predictions)
    metrics["loss"] = float(np.mean(losses))
    return metrics, labels_array, predictions, probability_array


def train_model(
    architecture: str,
    max_length: int,
    splits: dict[str, pd.DataFrame],
    vocabulary: dict[str, int],
    config: Config,
) -> tuple[SequenceClassifier, dict, list[dict]]:
    """Huấn luyện mô hình tuần tự và ghi lại lịch sử huấn luyện."""
    torch.manual_seed(config.seed)
    datasets = {
        name: ReviewDataset(frame, vocabulary, max_length, config)
        for name, frame in splits.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=name == "train",
            num_workers=config.max_workers,
        )
        for name, dataset in datasets.items()
    }
    model = SequenceClassifier(architecture, len(vocabulary) + 2, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for tokens, lengths, targets in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(tokens, lengths), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        validation_metrics, _, _, _ = evaluate(model, loaders["validation"], criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_metrics["loss"],
                "validation_f1": validation_metrics["f1"],
            }
        )
    duration = time.perf_counter() - start
    test_metrics, labels, predictions, probabilities = evaluate(
        model, loaders["test"], criterion
    )
    tokens_total = sum(len(tokenize(text)) for text in splits["train"]["review"])
    tokens_retained = sum(
        min(len(tokenize(text)), max_length) for text in splits["train"]["review"]
    )
    result = {
        "model": architecture,
        "max_length": max_length,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_time_seconds": duration,
        "epochs": config.epochs,
        "vocabulary_size": len(vocabulary) + 2,
        "tokens_retained_percent": 100 * tokens_retained / tokens_total,
        "validation_loss": history[-1]["validation_loss"],
        **test_metrics,
    }
    return (
        model,
        {
            "result": result,
            "labels": labels,
            "predictions": predictions,
            "probabilities": probabilities,
        },
        history,
    )


def run_baseline(
    splits: dict[str, pd.DataFrame], config: Config
) -> tuple[dict, np.ndarray, TfidfVectorizer, LogisticRegression]:
    """Mô hình cơ sở TF-IDF + Logistic Regression từ LAB 04."""
    print("-> Đang chạy baseline: TF-IDF + Logistic Regression...")
    vectorizer = TfidfVectorizer(
        max_features=config.vocabulary_size,
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
    )
    start = time.perf_counter()
    train_features = vectorizer.fit_transform(splits["train"]["review"])
    classifier = LogisticRegression(max_iter=1000, random_state=config.seed)
    classifier.fit(train_features, splits["train"]["sentiment"])
    duration = time.perf_counter() - start
    test_features = vectorizer.transform(splits["test"]["review"])
    predictions = classifier.predict(test_features)
    labels = (splits["test"]["sentiment"] == LABEL_NAMES[1]).astype(int).to_numpy()
    predicted_binary = (predictions == LABEL_NAMES[1]).astype(int)
    result = {
        "model": "TF-IDF + LR",
        "max_length": "",
        "parameters": int(train_features.shape[1] + 1),
        "training_time_seconds": duration,
        "epochs": "",
        "vocabulary_size": len(vectorizer.vocabulary_),
        "tokens_retained_percent": 100.0,
        "validation_loss": "",
        "loss": "",
        **score(labels, predicted_binary),
    }
    return result, predicted_binary, vectorizer, classifier


def predict_text(
    model: SequenceClassifier,
    text: str,
    vocabulary: dict[str, int],
    config: Config,
    max_len: int | None = None,
) -> float:
    """Dự đoán xác suất nhãn tích cực (positive) cho một câu đầu vào bất kỳ."""
    length_to_use = max_len or config.primary_length
    ids, length = encode(text, vocabulary, length_to_use, config)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([ids]), torch.tensor([length]))
        return float(torch.sigmoid(logits)[0])


def probe_order(
    model: SequenceClassifier,
    vocabulary: dict[str, int],
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    config: Config,
) -> list[dict]:
    """Kiểm chứng độ nhạy với thứ tự từ (word order) giữa TF-IDF và mô hình chuỗi."""
    rows = []
    for review in ORDER_PROBE_REVIEWS:
        baseline_prob = classifier.predict_proba(vectorizer.transform([review]))[
            0, list(classifier.classes_).index(LABEL_NAMES[1])
        ]
        gru_prob = predict_text(model, review, vocabulary, config)
        rows.append(
            {
                "review": review,
                "tfidf_positive_probability": float(baseline_prob),
                "gru_positive_probability": float(gru_prob),
            }
        )
    return rows


def write_figures(
    results: list[dict],
    histories: dict[str, list[dict]],
    labels: np.ndarray,
    predictions: np.ndarray,
    config: Config,
) -> None:
    """Lưu các biểu đồ learning curves, ma trận nhầm lẫn và trade-off độ dài."""
    # 1. Learning curves
    figure, axis = plt.subplots(figsize=(8, 4.5))
    markers = {"train": "o", "validation": "s"}
    colors = {"RNN": "#e41a1c", "LSTM": "#377eb8", "GRU": "#4daf4a"}
    for name, history in histories.items():
        if name in MODEL_NAMES:
            c = colors.get(name, "gray")
            axis.plot(
                [row["epoch"] for row in history],
                [row["train_loss"] for row in history],
                marker=markers["train"],
                linestyle="-",
                color=c,
                label=f"{name} Train Loss",
            )
            axis.plot(
                [row["epoch"] for row in history],
                [row["validation_loss"] for row in history],
                marker=markers["validation"],
                linestyle="--",
                color=c,
                label=f"{name} Val Loss",
            )
    axis.set_xlabel("Epoch", fontsize=11)
    axis.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    axis.set_title("Đường cong huấn luyện (Learning Curves) tại max_length=100", fontsize=12)
    axis.legend(ncol=3, fontsize=9)
    axis.grid(True, linestyle="--", alpha=0.5)
    figure.tight_layout()
    figure.savefig(config.output_path / "learning_curves.png", dpi=config.plot_dpi)
    plt.close(figure)
    print("   Đã lưu biểu đồ learning_curves.png")

    # 2. Confusion matrix
    matrix = confusion_matrix(labels, predictions)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(matrix, display_labels=LABEL_NAMES)
    display.plot(cmap="Blues", values_format="d", ax=ax_cm)
    ax_cm.set_title("Ma trận nhầm lẫn mô hình GRU (Held-out Test Set)", fontsize=12)
    fig_cm.tight_layout()
    fig_cm.savefig(config.output_path / "confusion_matrix.png", dpi=config.plot_dpi)
    plt.close(fig_cm)
    print("   Đã lưu biểu đồ confusion_matrix.png")

    # 3. Length trade-off
    length_rows = sorted(
        [row for row in results if row["model"] == "GRU"],
        key=lambda r: int(r["max_length"]),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(
        [row["max_length"] for row in length_rows],
        [row["f1"] for row in length_rows],
        marker="o",
        color="#2b5c8f",
        linewidth=2,
    )
    axes[0].set_xlabel("Max length (tokens)", fontsize=11)
    axes[0].set_ylabel("Test F1 Score", fontsize=11)
    axes[0].set_title("Độ dài chuỗi vs Test F1", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(
        [row["max_length"] for row in length_rows],
        [row["training_time_seconds"] for row in length_rows],
        marker="s",
        color="#d95f02",
        linewidth=2,
    )
    axes[1].set_xlabel("Max length (tokens)", fontsize=11)
    axes[1].set_ylabel("Thời gian huấn luyện (giây)", fontsize=11)
    axes[1].set_title("Độ dài chuỗi vs Thời gian huấn luyện", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    figure.tight_layout()
    figure.savefig(config.output_path / "length_tradeoff.png", dpi=config.plot_dpi)
    plt.close(figure)
    print("   Đã lưu biểu đồ length_tradeoff.png")


def write_analysis(
    splits: dict[str, pd.DataFrame],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    config: Config,
) -> None:
    """Tạo bảng phân tích ít nhất 3 mẫu đúng và 3 mẫu sai từ held-out test set."""
    test = splits["test"]
    labels = (test["sentiment"] == LABEL_NAMES[1]).astype(int).to_numpy()
    rows = []
    for outcome, indices in (
        ("Dự đoán ĐÚNG", np.flatnonzero(labels == predictions)),
        ("Dự đoán SAI", np.flatnonzero(labels != predictions)),
    ):
        for index in indices[: config.examples_per_outcome]:
            text = test.iloc[index]["review"]
            tokens = tokenize(text)
            reasons = []
            if any(token in tokens for token in ("not", "never", "no", "neither", "nor")):
                reasons.append("Chứa từ phủ định (negation)")
            if any(token in tokens for token in ("but", "however", "although", "though", "despite")):
                reasons.append("Chuyển ý / tương phản (contrast)")
            if len(tokens) > config.primary_length:
                reasons.append(f"Cắt cụt độ dài (tokens={len(tokens)} > {config.primary_length})")
            if not reasons:
                reasons.append("Từ vựng hoặc ngữ cảnh mập mờ (lexical/context ambiguity)")
            rows.append(
                (
                    outcome,
                    LABEL_NAMES[labels[index]],
                    LABEL_NAMES[predictions[index]],
                    float(probabilities[index]),
                    len(tokens),
                    "; ".join(reasons),
                    text[: config.error_snippet_length].replace("|", "\\|").replace("\n", " "),
                )
            )

    lines = [
        "# Phân tích lỗi mô hình (Error Analysis) — LAB 05",
        "",
        "Bảng tổng hợp chi tiết ít nhất 3 mẫu dự đoán đúng và 3 mẫu dự đoán sai từ tập kiểm thử held-out test set của mô hình GRU (max_length=100).",
        "",
        "| Phân loại | Nhãn thực tế | Dự đoán | P(positive) | Số tokens | Yếu tố ngôn ngữ / Kỹ thuật ảnh hưởng | Đoạn trích văn bản review |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for outcome, truth, pred, prob, length, reason, excerpt in rows:
        lines.append(
            f"| {outcome} | **{truth}** | {pred} | {prob:.3f} | {length} | {reason} | {excerpt}... |"
        )
    lines.extend(
        [
            "",
            "## Nhận xét chuyên sâu về các nguyên nhân lỗi",
            "",
            "1. **Ảnh hưởng của việc cắt cụt chuỗi (Truncation):** Đối với các bài đánh giá dài (>100 tokens), phần kết luận hoặc câu chốt cảm xúc ('ultimately brilliant', 'overall a waste of time') thường nằm ở cuối bài. Khi bị cắt tại token thứ 100, mô hình chỉ đọc được phần tóm tắt cốt truyện mở đầu mà mất đi nhận định cảm xúc quyết định.",
            "2. **Cấu trúc tương phản và đảo ý (Contrast):** Nhiều đánh giá mở đầu bằng việc khen ngợi một khía cạnh kỹ thuật ('great cinematography, wonderful acting') nhưng sau đó dùng 'however', 'but' để chê cốt truyện tổng thể. Nếu mô hình chưa học đủ trọng số chú ý để vượt qua các từ tích cực mạnh ban đầu, nó sẽ phân loại nhầm.",
            "3. **Từ phủ định (Negation):** Dù GRU có cơ chế cập nhật trạng thái tốt hơn Bag-of-Words, khi từ phủ định đứng cách xa vị ngữ hoặc mang tính phức hợp ('not as good as I hoped', 'hardly worth watching'), mô hình vẫn có thể bị chi phối bởi các từ đơn lẻ có trọng số dương mạnh.",
            "4. **Không phải mọi lỗi đều do sequence modeling:** Các yếu tố như từ hiếm nằm ngoài vocabulary (bị thay bằng UNK), độ dài ngữ cảnh bị giới hạn bởi tài nguyên tính toán (100 tokens), và ngân sách huấn luyện ngắn (2 epochs trên CPU) cũng đóng vai trò quyết định.",
        ]
    )
    (config.output_path / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("   Đã lưu phân tích lỗi vào error_analysis.md")


def main() -> None:
    """Hàm điều khiển toàn bộ pipeline thí nghiệm."""
    print("=" * 70)
    print("BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE THÍ NGHIỆM LAB 05")
    print("=" * 70)
    config = Config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)

    print(f"Data path: {config.data_path}")
    frame = pd.read_csv(config.data_path)
    summary = inspect_dataset(frame, config)
    splits = prepare_splits(frame, config)
    vocabulary = build_vocabulary(splits["train"]["review"], config)
    print(f"Vocabulary size fit on train: {len(vocabulary) + 2}")

    baseline, _, vectorizer, baseline_classifier = run_baseline(splits, config)
    results = [baseline]
    histories = {}
    selected_model = None
    selected_evaluation = None

    # Thử nghiệm 3 kiến trúc recurrent tại max_length=100
    for architecture in MODEL_NAMES:
        print(f"-> Đang huấn luyện {architecture} (max_length={config.primary_length})...")
        model, evaluation, history = train_model(
            architecture, config.primary_length, splits, vocabulary, config
        )
        results.append(evaluation["result"])
        histories[architecture] = history
        print(
            f"   {architecture:5s} | Test Acc: {evaluation['result']['accuracy']:.4f} | "
            f"F1: {evaluation['result']['f1']:.4f} | Time: {evaluation['result']['training_time_seconds']:.2f}s"
        )
        if architecture == "GRU":
            selected_model = model
            selected_evaluation = evaluation

    # Thử nghiệm thay đổi độ dài sequence length trên GRU
    for max_length in config.lengths:
        if max_length != config.primary_length:
            print(f"-> Đang huấn luyện GRU (max_length={max_length})...")
            _, evaluation, history = train_model(
                "GRU", max_length, splits, vocabulary, config
            )
            results.append(evaluation["result"])
            histories[f"GRU_{max_length}"] = history
            print(
                f"   GRU ({max_length:3d}) | Test Acc: {evaluation['result']['accuracy']:.4f} | "
                f"F1: {evaluation['result']['f1']:.4f} | Time: {evaluation['result']['training_time_seconds']:.2f}s"
            )

    assert selected_model is not None and selected_evaluation is not None
    print("-> Đang xuất biểu đồ và phân tích lỗi...")
    write_figures(
        results,
        histories,
        selected_evaluation["labels"],
        selected_evaluation["predictions"],
        config,
    )
    write_analysis(
        splits,
        selected_evaluation["predictions"],
        selected_evaluation["probabilities"],
        config,
    )

    # Lưu results.csv
    with (config.output_path / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("   Đã lưu kết quả vào results.csv")

    # Các phép đo định tính (Probes)
    print("-> Đang thực hiện các phép đo định tính (Probes)...")
    probes = {
        "application": {
            name: predict_text(selected_model, review, vocabulary, config)
            for name, review in APPLICATION_REVIEWS.items()
        },
        "positions": {
            str(position): {
                cue: predict_text(
                    selected_model,
                    template.format(cue=cue),
                    vocabulary,
                    config,
                )
                for cue in LONG_RANGE_CUES
            }
            for position, template in enumerate(LONG_RANGE_TEMPLATES)
        },
        "order": probe_order(
            selected_model, vocabulary, vectorizer, baseline_classifier, config
        ),
    }

    cell_match = verify_against_torch()[0]
    metadata = {
        "dataset": summary,
        "split_sizes": {name: len(split) for name, split in splits.items()},
        "config": {
            "seed": config.seed,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "embedding_size": config.embedding_size,
            "hidden_size": config.hidden_size,
            "learning_rate": config.learning_rate,
            "primary_length": config.primary_length,
        },
        "cell_matches_torch": cell_match,
        "history": histories,
        "probes": probes,
        "confusion_matrix": confusion_matrix(
            selected_evaluation["labels"], selected_evaluation["predictions"]
        ).tolist(),
    }
    (config.output_path / "experiment_details.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("   Đã lưu metadata vào experiment_details.json")

    # Lưu model checkpoint
    torch.save(
        {
            "model": selected_model.state_dict(),
            "vocabulary": vocabulary,
            "config": metadata["config"],
        },
        config.output_path / "gru_model.pt",
    )
    print("   Đã lưu trọng số mô hình vào gru_model.pt")
    print("=" * 70)
    print("HOÀN TẤT PIPELINE THÍ NGHIỆM LAB 05 THÀNH CÔNG!")
    print("=" * 70)


if __name__ == "__main__":
    main()
