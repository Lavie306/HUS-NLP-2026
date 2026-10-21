"""Simple RNN Cell implementation from scratch and verification for LAB 05.

Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng - Học kỳ I, 2026.
Yêu cầu theo Section 14 trong W5.pdf:
- Xây dựng SimpleRNNCell với __init__(input_size, hidden_size) và forward(x_t, h_prev).
- Tính toán: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h).
- Kiểm tra tính toán trên dữ liệu nhỏ từ Bài 1 (Section 6 trong W5.pdf).
- So sánh kiểm chứng tương đương với torch.nn.RNNCell.
"""

from __future__ import annotations

import sys
import torch
from torch import Tensor, nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")



class SimpleRNNCell(nn.Module):
    """RNN cell cơ bản thực hiện một bước recurrent theo công thức:
    h_t = tanh(x_t @ W_xh^T + h_{t-1} @ W_hh^T + b_h)
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Trọng số W_xh: (hidden_size, input_size)
        self.weight_input = nn.Parameter(torch.empty(hidden_size, input_size))
        # Trọng số W_hh: (hidden_size, hidden_size)
        self.weight_hidden = nn.Parameter(torch.empty(hidden_size, hidden_size))
        # Bias b_h: (hidden_size,)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Khởi tạo trọng số bằng phân phối Xavier/Orthogonal."""
        nn.init.xavier_uniform_(self.weight_input)
        nn.init.orthogonal_(self.weight_hidden)
        nn.init.zeros_(self.bias)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """Thực hiện một bước forward:
        Args:
            x_t: Tensor kích thước (batch_size, input_size)
            h_prev: Tensor kích thước (batch_size, hidden_size)
        Returns:
            h_t: Tensor kích thước (batch_size, hidden_size)
        """
        linear_input = x_t @ self.weight_input.T
        linear_hidden = h_prev @ self.weight_hidden.T
        pre_activation = linear_input + linear_hidden + self.bias
        return torch.tanh(pre_activation)


def verify_against_torch() -> tuple[bool, Tensor, Tensor, Tensor]:
    """Kiểm tra SimpleRNNCell với ví dụ tính toán thủ công Bài 1 (Section 6)
    và so sánh trực tiếp với torch.nn.RNNCell.
    """
    input_size = 2
    hidden_size = 2

    # Dữ liệu từ Bài 1 (W5.pdf)
    # x_t = [1, 0]^T
    x_t = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    # h_{t-1} = [0.5, 0.2]^T
    h_prev = torch.tensor([[0.5, 0.2]], dtype=torch.float32)

    # Trọng số từ Bài 1 (W5.pdf)
    # W_xh = [[0.5, 0.2], [0.1, 0.4]]
    W_xh = torch.tensor([[0.5, 0.2], [0.1, 0.4]], dtype=torch.float32)
    # W_hh = [[0.3, 0.1], [0.2, 0.5]]
    W_hh = torch.tensor([[0.3, 0.1], [0.2, 0.5]], dtype=torch.float32)
    # b_h = [0, 0]^T
    b_h = torch.tensor([0.0, 0.0], dtype=torch.float32)

    # 1. Custom cell
    custom_cell = SimpleRNNCell(input_size, hidden_size)
    with torch.no_grad():
        custom_cell.weight_input.copy_(W_xh)
        custom_cell.weight_hidden.copy_(W_hh)
        custom_cell.bias.copy_(b_h)
    with torch.no_grad():
        h_custom = custom_cell(x_t, h_prev)


    # 2. PyTorch reference cell
    torch_cell = nn.RNNCell(input_size, hidden_size, nonlinearity="tanh")
    with torch.no_grad():
        torch_cell.weight_ih.copy_(W_xh)
        torch_cell.weight_hh.copy_(W_hh)
        torch_cell.bias_ih.zero_()
        torch_cell.bias_hh.zero_()
        h_torch = torch_cell(x_t, h_prev)

    # 3. Giá trị giải tích từ Bài 1: a_t = [0.67, 0.30]^T -> h_t = tanh(a_t) ~ [0.584983, 0.291313]
    h_analytical = torch.tanh(torch.tensor([[0.67, 0.30]], dtype=torch.float32))

    matches = bool(
        torch.allclose(h_custom, h_torch, atol=1e-7)
        and torch.allclose(h_custom, h_analytical, atol=1e-5)
    )
    return matches, h_custom, h_torch, h_analytical


if __name__ == "__main__":
    is_match, h_custom, h_torch, h_analytical = verify_against_torch()
    print("=" * 60)
    print("KIỂM TRA TÍNH TOÁN BÀI 1 & ĐỐI CHIẾU PYTORCH")
    print("=" * 60)
    print("Custom SimpleRNNCell output :", h_custom.numpy())
    print("PyTorch nn.RNNCell output   :", h_torch.numpy())
    print("Analytical expected output  :", h_analytical.numpy())
    print(f"Khớp tuyệt đối (allclose)   : {is_match}")
    print("=" * 60)
