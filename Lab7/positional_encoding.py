"""
positional_encoding.py
=======================
Chương trình minh họa và thực nghiệm về Positional Encoding trong kiến trúc Transformer.

Nội dung bao gồm:
1. Định nghĩa lớp SinusoidalPositionalEncoding theo công thức gốc của Vaswani et al. (2017).
2. Thực nghiệm 1 (Mục 11): Khảo sát ảnh hưởng của thứ tự từ trên câu ngắn ["I", "love", "NLP"]
   so với ["NLP", "love", "I"].
3. Permutation Experiment (Mục 12): Kiểm chứng thực nghiệm trên cặp câu hoán vị:
   S1 = "dog bites man" và S2 = "man bites dog".
   - Kịch bản A: Self-Attention thuần túy (không có Positional Encoding).
   - Kịch bản B: Self-Attention tích hợp Positional Encoding.
   - Định lượng qua Khoảng cách Euclid và Độ tương đồng Cosine (Cosine Similarity).
4. Xuất biểu đồ phân tích trực quan lưu tại positional_encoding.png.
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def compute_softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Tính hàm Softmax với kỹ thuật dịch trừ max để tránh tràn số (numerical stability)."""
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)


def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Tính hệ số tương đồng cosin giữa hai vector đặc trưng."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class SinusoidalPositionalEncoding:
    """
    Sinh ma trận Positional Encoding dạng hình sin (Vaswani et al., 2017).

    Công thức:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, max_sequence_length: int = 5000):
        self.d_model = d_model
        self.max_len = max_sequence_length
        self.encoding_matrix = self._construct_encoding()

    def _construct_encoding(self) -> np.ndarray:
        matrix = np.zeros((self.max_len, self.d_model), dtype=np.float64)
        positions = np.arange(0, self.max_len, dtype=np.float64)[:, np.newaxis]

        # Tần số góc theo từng cặp chiều chẵn lẻ: 10000^(-2i / d_model)
        dim_indices = np.arange(0, self.d_model, 2, dtype=np.float64)
        divisors = np.exp(dim_indices * -(np.log(10000.0) / self.d_model))

        matrix[:, 0::2] = np.sin(positions * divisors)
        matrix[:, 1::2] = np.cos(positions * divisors)
        return matrix

    def get_pe(self, sequence_length: int) -> np.ndarray:
        """Trích xuất ma trận vị trí cho độ dài sequence_length."""
        if sequence_length > self.max_len:
            raise ValueError(f"Độ dài {sequence_length} vượt quá giới hạn tối đa {self.max_len}")
        return self.encoding_matrix[:sequence_length, :].copy()


def execute_self_attention(
    inputs: np.ndarray,
    proj_q: np.ndarray | None = None,
    proj_k: np.ndarray | None = None,
    proj_v: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thực hiện phép tính Scaled Dot-Product Self-Attention.
    Nếu không truyền ma trận chiếu, mặc định dùng phép chiếu đồng nhất (Identity).
    """
    queries = inputs @ proj_q if proj_q is not None else inputs
    keys = inputs @ proj_k if proj_k is not None else inputs
    values = inputs @ proj_v if proj_v is not None else inputs

    d_k = queries.shape[-1]
    raw_scores = (queries @ keys.T) / np.sqrt(d_k)
    attention_weights = compute_softmax(raw_scores, axis=-1)
    context_vectors = attention_weights @ values
    return raw_scores, attention_weights, context_vectors


def run_experiment_1() -> None:
    """
    Thực nghiệm 1 (Mục 11):
    So sánh chuỗi ["I", "love", "NLP"] và chuỗi đảo vị trí ["NLP", "love", "I"].
    """
    print("=" * 70)
    print("THỰC NGHIỆM 1 (Mục 11): KHẢO SÁT CHUỖI NHỎ VÀ THỨ TỰ TỪ")
    print("=" * 70)

    # Embedding giả lập với d_model = 4
    vocab_emb = {
        "I": np.array([0.8, -0.2, 0.4, 0.1]),
        "love": np.array([0.1, 0.9, -0.3, 0.5]),
        "NLP": np.array([-0.6, 0.3, 0.8, 0.2]),
    }

    seq_original = ["I", "love", "NLP"]
    seq_reversed = ["NLP", "love", "I"]

    x_orig = np.stack([vocab_emb[w] for w in seq_original])
    x_rev = np.stack([vocab_emb[w] for w in seq_reversed])

    pe_generator = SinusoidalPositionalEncoding(d_model=4, max_sequence_length=10)
    pe_3 = pe_generator.get_pe(3)

    # Không PE
    _, attn_orig_no_pe, ctx_orig_no_pe = execute_self_attention(x_orig)
    _, attn_rev_no_pe, ctx_rev_no_pe = execute_self_attention(x_rev)
    rep_orig_no_pe = np.mean(ctx_orig_no_pe, axis=0)
    rep_rev_no_pe = np.mean(ctx_rev_no_pe, axis=0)

    # Có PE
    _, attn_orig_pe, ctx_orig_pe = execute_self_attention(x_orig + pe_3)
    _, attn_rev_pe, ctx_rev_pe = execute_self_attention(x_rev + pe_3)
    rep_orig_pe = np.mean(ctx_orig_pe, axis=0)
    rep_rev_pe = np.mean(ctx_rev_pe, axis=0)

    print("\n[Không dùng Positional Encoding]")
    print(f"  Sentence 1 representation : {rep_orig_no_pe}")
    print(f"  Sentence 2 representation : {rep_rev_no_pe}")
    print(f"  Khoảng cách Euclid        : {np.linalg.norm(rep_orig_no_pe - rep_rev_no_pe):.6e}")
    print(f"  Cosine Similarity         : {calculate_cosine_similarity(rep_orig_no_pe, rep_rev_no_pe):.6f}")

    print("\n[Có dùng Positional Encoding]")
    print(f"  Sentence 1 representation : {rep_orig_pe}")
    print(f"  Sentence 2 representation : {rep_rev_pe}")
    print(f"  Khoảng cách Euclid        : {np.linalg.norm(rep_orig_pe - rep_rev_pe):.6f}")
    print(f"  Cosine Similarity         : {calculate_cosine_similarity(rep_orig_pe, rep_rev_pe):.6f}")


def run_permutation_experiment(output_figure_path: str = "positional_encoding.png") -> None:
    """
    Thực nghiệm Hoán vị (Mục 12):
    S1 = 'dog bites man' (Chủ ngữ: dog, Tân ngữ: man)
    S2 = 'man bites dog' (Chủ ngữ: man, Tân ngữ: dog)
    """
    print("\n" + "=" * 70)
    print("THỰC NGHIỆM 2 (Mục 12): PERMUTATION EXPERIMENT ('dog bites man' vs 'man bites dog')")
    print("=" * 70)

    embeddings: Dict[str, np.ndarray] = {
        "dog": np.array([1.1, 0.4, -0.6, 0.7]),
        "bites": np.array([0.2, 1.4, 0.1, -0.5]),
        "man": np.array([-0.9, 0.1, 1.3, 0.4]),
    }

    tokens_s1 = ["dog", "bites", "man"]
    tokens_s2 = ["man", "bites", "dog"]

    x_s1 = np.stack([embeddings[w] for w in tokens_s1])
    x_s2 = np.stack([embeddings[w] for w in tokens_s2])

    pe_engine = SinusoidalPositionalEncoding(d_model=4, max_sequence_length=10)
    pe_values = pe_engine.get_pe(3)

    # 1. Kịch bản A: Không Positional Encoding
    _, a_s1_no_pe, z_s1_no_pe = execute_self_attention(x_s1)
    _, a_s2_no_pe, z_s2_no_pe = execute_self_attention(x_s2)
    pool_s1_no_pe = np.mean(z_s1_no_pe, axis=0)
    pool_s2_no_pe = np.mean(z_s2_no_pe, axis=0)
    dist_no_pe = float(np.linalg.norm(pool_s1_no_pe - pool_s2_no_pe))
    cos_no_pe = calculate_cosine_similarity(pool_s1_no_pe, pool_s2_no_pe)

    # 2. Kịch bản B: Tích hợp Positional Encoding
    _, a_s1_pe, z_s1_pe = execute_self_attention(x_s1 + pe_values)
    _, a_s2_pe, z_s2_pe = execute_self_attention(x_s2 + pe_values)
    pool_s1_pe = np.mean(z_s1_pe, axis=0)
    pool_s2_pe = np.mean(z_s2_pe, axis=0)
    dist_pe = float(np.linalg.norm(pool_s1_pe - pool_s2_pe))
    cos_pe = calculate_cosine_similarity(pool_s1_pe, pool_s2_pe)

    print("\n[Kịch bản A: Self-Attention KHÔNG Positional Encoding]")
    print(f"  Biểu diễn toàn chuỗi S1 : {np.round(pool_s1_no_pe, 4)}")
    print(f"  Biểu diễn toàn chuỗi S2 : {np.round(pool_s2_no_pe, 4)}")
    print(f"  Khoảng cách Euclid      : {dist_no_pe:.6e}")
    print(f"  Độ tương đồng Cosine    : {cos_no_pe:.6f}")
    print("  => KẾT LUẬN: Thiếu PE, hai câu có ý nghĩa trái ngược cho biểu diễn Y HỆT NHAU!")

    print("\n[Kịch bản B: Self-Attention CÓ Positional Encoding]")
    print(f"  Biểu diễn toàn chuỗi S1 : {np.round(pool_s1_pe, 4)}")
    print(f"  Biểu diễn toàn chuỗi S2 : {np.round(pool_s2_pe, 4)}")
    print(f"  Khoảng cách Euclid      : {dist_pe:.6f}")
    print(f"  Độ tương đồng Cosine    : {cos_pe:.6f}")
    print("  => KẾT LUẬN: Có PE, mô hình phân biệt rạch ròi thứ tự từ và cấu trúc ngữ pháp!")

    # 3. Trực quan hóa kết quả
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # Plot 1: Heatmap Positional Encoding diện rộng
    pe_large = SinusoidalPositionalEncoding(d_model=16, max_sequence_length=30).get_pe(30)
    im0 = axes[0, 0].imshow(pe_large, cmap="coolwarm", aspect="auto")
    axes[0, 0].set_title("(a) Sinusoidal PE Matrix (30 pos x 16 dim)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Feature Dimension (i)")
    axes[0, 0].set_ylabel("Sequence Position (pos)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    def draw_heatmap(ax, matrix, labels_x, labels_y, title_text, colormap="Blues"):
        im = ax.imshow(matrix, cmap=colormap, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels_x)))
        ax.set_yticks(range(len(labels_y)))
        ax.set_xticklabels(labels_x, fontweight="bold")
        ax.set_yticklabels(labels_y, fontweight="bold")
        ax.set_title(title_text, fontsize=11, fontweight="bold")
        for r in range(len(labels_y)):
            for c in range(len(labels_x)):
                val = matrix[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.5 else "black", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot 2 & 3: Attention Matrices không PE
    draw_heatmap(axes[0, 1], a_s1_no_pe, tokens_s1, tokens_s1, "(b) S1 Attention [No PE]", "Blues")
    draw_heatmap(axes[0, 2], a_s2_no_pe, tokens_s2, tokens_s2, "(c) S2 Attention [No PE]", "Blues")

    # Plot 4 & 5: Attention Matrices có PE
    draw_heatmap(axes[1, 0], a_s1_pe, tokens_s1, tokens_s1, "(d) S1 Attention [With PE]", "Purples")
    draw_heatmap(axes[1, 1], a_s2_pe, tokens_s2, tokens_s2, "(e) S2 Attention [With PE]", "Purples")

    # Plot 6: Biểu đồ cột định lượng sự khác biệt
    labels = ["Euclidean Dist", "Cosine Dist (1 - Sim)"]
    vals_no_pe = [dist_no_pe, max(0.0, 1.0 - cos_no_pe)]
    vals_pe = [dist_pe, max(0.0, 1.0 - cos_pe)]

    bar_width = 0.35
    indices = np.arange(len(labels))
    axes[1, 2].bar(indices - bar_width / 2, vals_no_pe, bar_width, label="Không có PE", color="#4a7bb0")
    axes[1, 2].bar(indices + bar_width / 2, vals_pe, bar_width, label="Có PE", color="#e06666")
    axes[1, 2].set_xticks(indices)
    axes[1, 2].set_xticklabels(labels, fontweight="bold")
    axes[1, 2].set_ylabel("Mức độ phân biệt giữa S1 và S2")
    axes[1, 2].set_title("(f) Phân biệt S1 ('dog bites man') vs S2", fontsize=11, fontweight="bold")
    axes[1, 2].legend()
    axes[1, 2].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_figure_path, dpi=300)
    plt.close()
    print(f"\n[THÔNG BÁO] Đã lưu đồ thị phân tích thực nghiệm tại: {output_figure_path}")


if __name__ == "__main__":
    run_experiment_1()
    figure_file = os.path.join(os.path.dirname(__file__), "positional_encoding.png")
    run_permutation_experiment(figure_file)
