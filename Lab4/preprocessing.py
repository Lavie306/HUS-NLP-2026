"""Reproducible IMDb sentiment classification experiments for Lab 4.

The script keeps the split-before-fit rule explicit: all vectorizers are fit on
the training split only, then used to transform validation and test data.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tarfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "aclImdb_v1.tar.gz"
OUT = ROOT


def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text.lower())
    return re.sub(r"[^a-z0-9'!? ]+", " ", text).replace("  ", " ").strip()


def load_imdb(archive: Path = ARCHIVE) -> pd.DataFrame:
    rows = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if len(parts) != 4 or parts[0] != "aclImdb" or parts[1] not in {"train", "test"}:
                continue
            split, label_dir, filename = parts[1], parts[2], parts[3]
            if label_dir not in {"pos", "neg"} or not member.isfile():
                continue
            match = re.match(r"(\d+)_", filename)
            rating = int(match.group(1)) if match else None
            raw = tar.extractfile(member).read().decode("utf-8", errors="replace")
            rows.append({"split": split, "label": int(label_dir == "pos"),
                         "rating": rating, "text": clean_text(raw)})
    df = pd.DataFrame(rows).dropna(subset=["text", "label"])
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


def metrics(y_true, y_pred, seconds, model, representation, vocabulary_size):
    return {"representation": representation, "classifier": model,
            "training_time_s": round(seconds, 3),
            "vocabulary_size": int(vocabulary_size),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def fit_tfidf(train_x, train_y, test_x, ngram_range=(1, 1), classifier="lr"):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=2,
                                 max_df=0.95, sublinear_tf=True,
                                 max_features=80000)
    model = LogisticRegression(max_iter=1000, random_state=SEED, solver="liblinear")
    if classifier == "svm":
        model = LinearSVC(random_state=SEED)
    pipe = Pipeline([("tfidf", vectorizer), ("classifier", model)])
    start = time.perf_counter()
    pipe.fit(train_x, train_y)
    elapsed = time.perf_counter() - start
    return pipe, pipe.predict(test_x), elapsed, len(pipe.named_steps["tfidf"].vocabulary_)


def train_word2vec(train_x):
    tokenized = [x.split() for x in train_x]
    return Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2,
                    workers=1, sg=1, epochs=5, seed=SEED)


def average_vectors(model, texts):
    dim = model.vector_size
    output = []
    for text in texts:
        vectors = [model.wv[word] for word in text.split() if word in model.wv]
        output.append(np.mean(vectors, axis=0) if vectors else np.zeros(dim))
    return np.asarray(output)


def fit_avg_w2v(train_x, train_y, test_x):
    start = time.perf_counter()
    w2v = train_word2vec(train_x)
    train_vec = average_vectors(w2v, train_x)
    test_vec = average_vectors(w2v, test_x)
    classifier = LogisticRegression(max_iter=1000, random_state=SEED, solver="liblinear")
    classifier.fit(train_vec, train_y)
    return classifier.predict(test_vec), time.perf_counter() - start, len(w2v.wv)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["negative", "positive"],
           yticklabels=["negative", "positive"], xlabel="Predicted label",
           ylabel="True label", title="IMDb test confusion matrix (TF-IDF + LR)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use 4,000 train and 4,000 test docs")
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    df = load_imdb()
    train_df = df[df.split == "train"].copy()
    test_df = df[df.split == "test"].copy()
    if args.quick:
        train_df = train_df.groupby("label", group_keys=False).sample(n=2000, random_state=SEED)
        test_df = test_df.groupby("label", group_keys=False).sample(n=2000, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df.label,
                                        random_state=SEED)
    train_x, train_y = train_df.text.tolist(), train_df.label.to_numpy()
    val_x, val_y = val_df.text.tolist(), val_df.label.to_numpy()
    test_x, test_y = test_df.text.tolist(), test_df.label.to_numpy()

    results = []
    fitted = {}
    for name, ngrams, clf in [("TF-IDF unigram", (1, 1), "lr"),
                              ("TF-IDF unigram+bigram", (1, 2), "lr"),
                              ("TF-IDF unigram", (1, 1), "svm")]:
        pipe, pred, elapsed, vocab = fit_tfidf(train_x, train_y, test_x, ngrams, clf)
        results.append(metrics(test_y, pred, elapsed, "Linear SVM" if clf == "svm" else "Logistic Regression",
                                name, vocab))
        if name == "TF-IDF unigram" and clf == "lr":
            fitted["tfidf_lr"] = pipe
            fitted["tfidf_pred"] = pred
    pred_w2v, elapsed, vocab = fit_avg_w2v(train_x, train_y, test_x)
    results.append(metrics(test_y, pred_w2v, elapsed, "Logistic Regression", "Average Word2Vec", vocab))

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / "results.csv", index=False)
    cm = confusion_matrix(test_y, fitted["tfidf_pred"])
    plot_confusion(cm, OUT / "confusion_matrix.png")

    pipe = fitted["tfidf_lr"]
    pred = fitted["tfidf_pred"]
    report = classification_report(test_y, pred, target_names=["negative", "positive"], output_dict=True)
    top = []
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["classifier"]
    features = np.asarray(vec.get_feature_names_out())
    weights = clf.coef_[0]
    for i in np.argsort(weights)[-20:][::-1]:
        top.append({"feature": features[i], "weight": float(weights[i]), "direction": "positive"})
    for i in np.argsort(weights)[:20]:
        top.append({"feature": features[i], "weight": float(weights[i]), "direction": "negative"})
    summary = {"dataset_rows_after_dedup": len(df), "train": len(train_df), "validation": len(val_df),
               "test": len(test_df), "classes": sorted(df.label.unique().tolist()),
               "class_counts": {str(k): int(v) for k, v in df.label.value_counts().sort_index().items()},
               "duplicate_count": int(df.duplicated("text").sum()),
               "missing_text": int(df.text.isna().sum()), "missing_labels": int(df.label.isna().sum()),
               "document_length_words": {"mean": float(df.text.str.split().str.len().mean()),
                                         "median": float(df.text.str.split().str.len().median()),
                                         "p95": float(df.text.str.split().str.len().quantile(.95))},
               "confusion_matrix": cm.tolist(), "classification_report": report, "top_features": top}
    (OUT / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    examples = pd.DataFrame({"text": test_x, "true": test_y, "pred": pred})
    examples[examples.true == examples.pred].head(20).to_csv(OUT / "correct_examples.csv", index=False)
    examples[examples.true != examples.pred].head(20).to_csv(OUT / "misclassified.csv", index=False)
    print(result_df.to_string(index=False))
    print("\nConfusion matrix:\n", cm)


if __name__ == "__main__":
    main()
