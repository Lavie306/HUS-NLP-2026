"""Triển khai Attention Mechanism và Self-Attention bằng NumPy từ đầu (from scratch).

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng - Học kỳ I, 2026.
Bao gồm:
- stable_softmax: Softmax ổn định số bằng cách trừ max(x).
- softmax_with_temperature: Softmax với tham số nhiệt độ T.
- entropy: Đo độ hỗn loạn (entropy) của phân phối xác suất.
- scaled_dot_product_attention: Tính attention theo công thức Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.
- self_attention: Chiếu ma trận đầu vào X qua W_Q, W_K, W_V rồi tính scaled dot-product attention.
- verify_examples: Hàm kiểm tra tự động các bài toán tính tay và tính ổn định số.
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


def naive_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Tính softmax trực tiếp (dễ gây overflow với giá trị lớn)."""
    values = np.asarray(x, dtype=np.float64)
    exp_x = np.exp(values)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def stable_softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Tính softmax ổn định số bằng cách trừ giá trị lớn nhất theo trục.

    Công thức: softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores phải khác rỗng và chỉ chứa các giá trị số hữu hạn.")
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def softmax_with_temperature(scores: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Tính softmax với tham số nhiệt độ (temperature) T > 0.

    - T thấp (< 1.0): phân phối tập trung (gần argmax).
    - T cao (> 1.0): phân phối dàn đều (tiến tới uniform).
    """
    if temperature <= 0:
        raise ValueError("Temperature T phải là số thực dương (T > 0).")
    values = np.asarray(scores, dtype=np.float64)
    return stable_softmax(values / temperature, axis=axis)


def entropy(probabilities: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray | float:
    """Tính Shannon entropy theo đơn vị nat: H(p) = - sum(p * ln(p))."""
    p = np.asarray(probabilities, dtype=np.float64)
    p_clipped = np.clip(p, eps, 1.0)
    ent = -np.sum(p * np.log(p_clipped), axis=axis)
    if ent.ndim == 0:
        return float(ent)
    return ent


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tính scaled dot-product attention chuẩn bằng NumPy.

    Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + M) V

    Tham số:
        Q: Ma trận Query (..., T_q, d_k)
        K: Ma trận Key (..., T_k, d_k)
        V: Ma trận Value (..., T_k, d_v)
        mask: Ma trận nhị phân hoặc boolean, True ở các vị trí hợp lệ cần giữ, False ở vị trí cần che.

    Trả về:
        scores: Ma trận attention scores trước softmax (..., T_q, T_k)
        attention_weights: Ma trận trọng số attention sau softmax (..., T_q, T_k)
        context: Context vector sau khi nhân có trọng số với Value (..., T_q, d_v)
    """
    queries = np.asarray(Q, dtype=np.float64)
    keys = np.asarray(K, dtype=np.float64)
    values = np.asarray(V, dtype=np.float64)

    if queries.ndim < 2 or keys.ndim < 2 or values.ndim < 2:
        raise ValueError("Q, K và V phải có ít nhất 2 chiều.")
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError(
            f"Kích thước chiều đặc trưng của Q ({queries.shape[-1]}) và K ({keys.shape[-1]}) phải bằng nhau."
        )
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError(
            f"Độ dài chuỗi key ({keys.shape[-2]}) và value ({values.shape[-2]}) phải bằng nhau."
        )

    d_k = keys.shape[-1]
    # Tính dot product giữa Q và K^T
    # scores: (..., T_q, T_k)
    scores = (queries @ np.swapaxes(keys, -1, -2)) / np.sqrt(d_k)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        # Gán -1e9 cho những vị trí mask == False để exp(-1e9) -> 0
        scores = np.where(mask_arr, scores, -1e9)

    attention_weights = stable_softmax(scores, axis=-1)
    context = attention_weights @ values

    return scores, attention_weights, context


def self_attention(
    X: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triển khai Self-Attention bằng cách chiếu X qua các ma trận trọng số.

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    Context = Attention(Q, K, V)

    Trả về:
        (Q, K, V, scores, attention_weights, context)
    """
    inputs = np.asarray(X, dtype=np.float64)
    w_q = np.asarray(W_Q, dtype=np.float64)
    w_k = np.asarray(W_K, dtype=np.float64)
    w_v = np.asarray(W_V, dtype=np.float64)

    queries = inputs @ w_q
    keys = inputs @ w_k
    values = inputs @ w_v

    scores, attention_weights, context = scaled_dot_product_attention(
        queries, keys, values, mask=mask
    )
    return queries, keys, values, scores, attention_weights, context


def verify_examples() -> bool:
    """Xác minh các tính chất toán học và các bài toán mẫu trong W6.pdf."""
    # 1. Kiểm tra Numerical Stability với số rất lớn
    x_large = np.array([1000.0, 1001.0, 1002.0])
    softmax_large = stable_softmax(x_large)
    softmax_origin = stable_softmax(np.array([0.0, 1.0, 2.0]))
    assert np.allclose(softmax_large, softmax_origin, atol=1e-7), "Lỗi numerical stability!"
    assert np.isclose(np.sum(softmax_large), 1.0), "Tổng softmax phải bằng 1.0"

    # 2. Kiểm tra Bài 1: Attention scores
    q_b1 = np.array([1.0, 2.0])
    K_b1 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    s_b1 = q_b1 @ K_b1.T
    assert np.allclose(s_b1, [1.0, 2.0, 3.0]), "Lỗi tính score Bài 1!"

    # 3. Kiểm tra Bài 2: Softmax
    alpha_b2 = stable_softmax(s_b1)
    exp_sum = np.exp(1.0) + np.exp(2.0) + np.exp(3.0)
    expected_alpha = np.array([np.exp(1.0), np.exp(2.0), np.exp(3.0)]) / exp_sum
    assert np.allclose(alpha_b2, expected_alpha), "Lỗi softmax Bài 2!"
    assert np.isclose(np.sum(alpha_b2), 1.0), "Tổng weights Bài 2 phải bằng 1.0"

    # 4. Kiểm tra Bài 3: Context vector
    alpha_b3 = np.array([0.1, 0.2, 0.7])
    V_b3 = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    c_b3 = alpha_b3 @ V_b3
    assert np.allclose(c_b3, [1.5, 1.6]), "Lỗi tính context vector Bài 3!"

    # 5. Kiểm tra Bài 4: Scaled dot-product attention
    q_b4 = np.array([[1.0, 0.0]])
    K_b4 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V_b4 = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    scores_b4, weights_b4, context_b4 = scaled_dot_product_attention(q_b4, K_b4, V_b4)
    d_k_b4 = 2.0
    expected_scores = np.array([[1.0, 0.0, 1.0]]) / np.sqrt(d_k_b4)
    assert np.allclose(scores_b4, expected_scores), "Lỗi scaled scores Bài 4!"
    assert np.allclose(weights_b4, stable_softmax(expected_scores, axis=-1)), "Lỗi weights Bài 4!"
    assert np.isclose(np.sum(weights_b4), 1.0), "Tổng weights phải bằng 1.0"

    # 6. Kiểm tra Self-Attention
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    W = np.eye(2)
    Q, K, V, s, w, c = self_attention(X, W, W, W)
    assert w.shape == (3, 3), "Ma trận trọng số self-attention phải có kích thước (3, 3)"
    assert np.allclose(np.sum(w, axis=1), [1.0, 1.0, 1.0]), "Mỗi hàng của ma trận attention phải có tổng bằng 1.0"

    return True


if __name__ == "__main__":
    print("-> Đang kiểm tra attention_numpy.py...")
    x = np.array([1000.0, 1001.0, 1002.0])
    print("   Test x_large:", x)
    try:
        print("   Naive softmax:", naive_softmax(x))
    except Exception as e:
        print("   Naive softmax failed as expected:", e)
    print("   Stable softmax:", np.round(stable_softmax(x), 6))

    if verify_examples():
        print("-> TẤT CẢ CÁC BÀI TEST KIỂM TRA ĐỀU THÀNH CÔNG VÀ CHÍNH XÁC 100%!")
