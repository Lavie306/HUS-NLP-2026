"""
transformer_block.py
====================
Cài đặt kiến trúc khối Transformer Encoder Block tối giản từ các tầng nguyên thủy (primitives)
phục vụ LAB 07 môn Xử lý ngôn ngữ tự nhiên và ứng dụng (Vaswani et al., 2017).

Quy tắc bắt buộc:
- Tuyệt đối không dùng sẵn nn.TransformerEncoderLayer hay nn.MultiheadAttention.
- Tự ghép nối từ: nn.Linear, nn.LayerNorm, nn.Dropout, F.softmax.
- Cấu trúc tiêu chuẩn:
    Input X
      ↓
    Multi-Head Self-Attention
      ↓
    Residual Connection + LayerNorm
      ↓
    Position-wise Feed-Forward Network (FFN)
      ↓
    Residual Connection + LayerNorm
      ↓
    Output
- Hỗ trợ kiểm thử shape tự động theo Mục 18 của Lab:
    B = 8, T = 100, d_model = 128, n_heads = 4 => d_k = 32.
"""

from __future__ import annotations

import math
import sys
from typing import Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncodingTorch(nn.Module):
    """
    Tầng mã hóa vị trí Sinusoidal Positional Encoding kế thừa nn.Module.
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        encoding_table = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        freq_div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )

        encoding_table[:, 0::2] = torch.sin(pos * freq_div)
        encoding_table[:, 1::2] = torch.cos(pos * freq_div)

        # Lưu dưới dạng buffer không tham số (không cập nhật gradient)
        self.register_buffer("pe", encoding_table.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Seq_Len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttentionModule(nn.Module):
    """
    Cơ chế Multi-Head Attention tự triển khai.
    - Chiếu Q, K, V thông qua 3 phép biến đổi tuyến tính riêng rẽ.
    - Chia thành n_heads không gian con có số chiều d_k = d_model // n_heads.
    - Tính Scaled Dot-Product Attention song song trên từng head.
    - Ghép nối (Concat) các head và chiếu qua ma trận đầu ra W^O.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) phải chia hết cho số heads ({n_heads})!"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Ma trận chiếu tuyến tính cho Queries, Keys, Values
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Ma trận chiếu tổng hợp sau khi ghép nối các heads
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q_in, k_in, v_in: Tensor đầu vào shape (B, T, d_model)
            attention_mask: Tensor mặt nạ shape (B, 1, 1, T) hoặc (B, 1, T, T)
        Returns:
            context_output: Tensor đầu ra shape (B, T, d_model)
            attention_weights: Tensor trọng số shape (B, n_heads, T, T)
        """
        batch_size, seq_len, _ = q_in.size()

        # Bước 1: Chiếu tuyến tính và tách thành n_heads:
        # (B, T, d_model) -> (B, T, n_heads, d_k) -> (B, n_heads, T, d_k)
        Q = self.w_q(q_in).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k_in).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v_in).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Bước 2: Tính Scaled Dot-Product Attention:
        # Scores: (B, n_heads, T, d_k) @ (B, n_heads, d_k, T_k) -> (B, n_heads, T, T_k)
        raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attention_mask is not None:
            raw_scores = raw_scores.masked_fill(attention_mask == 0, -1e9)

        attention_weights = F.softmax(raw_scores, dim=-1)
        weights_dropped = self.dropout(attention_weights)

        # Bước 3: Nhân với Value V:
        # (B, n_heads, T, T) @ (B, n_heads, T, d_k) -> (B, n_heads, T, d_k)
        head_outputs = torch.matmul(weights_dropped, V)

        # Bước 4: Ghép nối (Concat) các heads trở lại không gian d_model:
        # (B, n_heads, T, d_k) -> (B, T, n_heads, d_k) -> (B, T, d_model)
        concatenated = (
            head_outputs.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Bước 5: Chiếu qua ma trận đầu ra W^O
        output = self.w_o(concatenated)

        return output, attention_weights


class PositionwiseFeedForwardModule(nn.Module):
    """
    Position-wise Feed-Forward Network:
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    Hoạt động độc lập trên từng vị trí token.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model) -> (B, T, d_ff) -> (B, T, d_model)
        """
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Khối Transformer Encoder Block hoàn chỉnh.
    Kiến trúc Post-LayerNorm cổ điển (Vaswani et al., 2017):
        1. Sub-layer 1: Multi-Head Attention với Residual Connection và LayerNorm.
        2. Sub-layer 2: Feed-Forward Network với Residual Connection và LayerNorm.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttentionModule(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        self.ffn = PositionwiseFeedForwardModule(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor đầu vào shape (B, T, d_model)
            mask: Tensor mặt nạ padding shape (B, 1, 1, T)
        Returns:
            out: Biểu diễn đầu ra shape (B, T, d_model)
            attn_weights: Phân phối trọng số attention shape (B, n_heads, T, T)
        """
        # --- Sub-layer 1: Self-Attention + Residual + LayerNorm ---
        attn_out, attn_weights = self.attention(x, x, x, attention_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # --- Sub-layer 2: Position-wise FFN + Residual + LayerNorm ---
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x, attn_weights


class TransformerClassifier(nn.Module):
    """
    Mô hình phân loại chuỗi văn bản sử dụng Transformer Encoder Block.
    Hỗ trợ 2 chiến lược Pooling:
    - 'mean': Masked Mean Pooling (lấy trung bình các token có nghĩa, bỏ qua PAD)
    - 'first' / 'cls': Lấy token đầu tiên làm biểu diễn toàn câu
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        num_classes: int = 1,
        pooling: str = "mean",
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling.lower()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncodingTorch(d_model, max_seq_len=max_seq_len, dropout=dropout)
        self.transformer_block = TransformerBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self, input_ids: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Tạo mask: 1 với token thật, 0 với token PAD (B, 1, 1, T)
        attention_mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # Embedding kết hợp hệ số tỷ lệ sqrt(d_model) theo chuẩn Vaswani
        embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(embeddings)

        # Truyền qua Transformer Encoder Block
        encoded_tokens, attn_weights = self.transformer_block(x, mask=attention_mask)

        # Cơ chế gộp chuỗi (Sequence Pooling)
        if self.pooling == "mean":
            # Masked Mean Pooling: chỉ tính trung bình các token không phải PAD
            valid_mask = (input_ids != self.pad_idx).unsqueeze(-1).float()  # (B, T, 1)
            summed = torch.sum(encoded_tokens * valid_mask, dim=1)           # (B, d_model)
            counts = torch.clamp(valid_mask.sum(dim=1), min=1e-9)            # (B, 1)
            pooled = summed / counts
        elif self.pooling in ("first", "cls"):
            # Lấy biểu diễn tại token đầu tiên (index 0)
            pooled = encoded_tokens[:, 0, :]
        else:
            raise ValueError(f"Chiến lược pooling không hỗ trợ: {self.pooling}")

        logits = self.classifier(pooled).squeeze(-1)

        if return_attention:
            return logits, attn_weights
        return logits


class LSTMClassifier(nn.Module):
    """
    Mô hình đối chứng Recurrent (LSTM) chuẩn từ LAB 05 để so sánh công bằng ở Experiment 4.
    Cùng vocabulary, cùng embedding dimension, cùng hidden dimension và linear classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.lstm = nn.LSTM(d_model, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _, (hidden_last, _) = self.lstm(embedded)
        # hidden_last: (num_layers, B, hidden_dim) -> lấy layer cuối
        logits = self.classifier(hidden_last[-1]).squeeze(-1)
        return logits


def verify_transformer_shapes() -> None:
    """
    Kiểm tra shape bắt buộc theo đúng quy chuẩn Mục 18 của LAB 07:
    Batch size: B = 8
    Sequence length: T = 100
    Model dimension: d_model = 128
    Number of heads: n_heads = 4
    => d_k = 128 / 4 = 32
    """
    print("=" * 70)
    print("KIỂM TRA SHAPE BẮT BUỘC (MỤC 18 - SHAPE VERIFICATION)")
    print("=" * 70)

    B = 8
    T = 100
    d_model = 128
    n_heads = 4
    d_k = d_model // n_heads
    d_ff = 256

    print(f"Cấu hình thử nghiệm:")
    print(f"  Batch size (B)           = {B}")
    print(f"  Sequence length (T)      = {T}")
    print(f"  Model dimension (d_model)= {d_model}")
    print(f"  Attention heads (n_heads)= {n_heads}")
    print(f"  Head dimension (d_k)     = {d_model} / {n_heads} = {d_k}")
    print(f"  Feed-Forward dim (d_ff)  = {d_ff}")

    # Giả lập đầu vào X
    X = torch.randn(B, T, d_model)
    print(f"\n1. Input X shape                        : {list(X.shape)}  [Mong đợi: [8, 100, 128]]")
    assert X.shape == (8, 100, 128)

    block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
    block.eval()

    # Kiểm tra phép chiếu Multi-Head Attention
    w_q = block.attention.w_q(X).view(B, T, n_heads, d_k).transpose(1, 2)
    w_k = block.attention.w_k(X).view(B, T, n_heads, d_k).transpose(1, 2)
    w_v = block.attention.w_v(X).view(B, T, n_heads, d_k).transpose(1, 2)

    print(f"2. Q, K, V projection shapes            : {list(w_q.shape)}  [Mong đợi: [8, 4, 100, 32]]")
    assert w_q.shape == (8, 4, 100, 32)
    assert w_k.shape == (8, 4, 100, 32)
    assert w_v.shape == (8, 4, 100, 32)

    scores = torch.matmul(w_q, w_k.transpose(-2, -1)) / math.sqrt(d_k)
    print(f"3. Attention Scores shape (Q @ K^T / √d): {list(scores.shape)}  [Mong đợi: [8, 4, 100, 100]]")
    assert scores.shape == (8, 4, 100, 100)

    attn_weights = F.softmax(scores, dim=-1)
    print(f"4. Attention Weights shape (Softmax)    : {list(attn_weights.shape)}  [Mong đợi: [8, 4, 100, 100]]")
    assert attn_weights.shape == (8, 4, 100, 100)

    head_out = torch.matmul(attn_weights, w_v)
    print(f"5. Head context outputs (Weights @ V)   : {list(head_out.shape)}  [Mong đợi: [8, 4, 100, 32]]")
    assert head_out.shape == (8, 4, 100, 32)

    concat_out = head_out.transpose(1, 2).contiguous().view(B, T, d_model)
    print(f"6. Concat heads shape                   : {list(concat_out.shape)}  [Mong đợi: [8, 100, 128]]")
    assert concat_out.shape == (8, 100, 128)

    attn_linear_out = block.attention.w_o(concat_out)
    print(f"7. Attention Output shape (sau W^O)     : {list(attn_linear_out.shape)}  [Mong đợi: [8, 100, 128]]")
    assert attn_linear_out.shape == (8, 100, 128)

    # Khối FFN
    ffn_hidden = block.ffn.activation(block.ffn.fc1(attn_linear_out))
    print(f"8. FFN Hidden layer shape               : {list(ffn_hidden.shape)}  [Mong đợi: [8, 100, 256]]")
    assert ffn_hidden.shape == (8, 100, 256)

    ffn_output = block.ffn.fc2(ffn_hidden)
    print(f"9. FFN Output shape                     : {list(ffn_output.shape)}  [Mong đợi: [8, 100, 128]]")
    assert ffn_output.shape == (8, 100, 128)

    # Chạy trọn vẹn forward của TransformerBlock
    full_out, weights = block(X)
    print(f"10. Final TransformerBlock Output shape  : {list(full_out.shape)}  [Mong đợi: [8, 100, 128]]")
    assert full_out.shape == (8, 100, 128)
    assert weights.shape == (8, 4, 100, 100)

    print("\n=> TẤT CẢ CÁC TẦNG ĐỀU TRÙNG KHỚP SHAPE 100% VỚI ĐẶC TẢ CỦA BÀI TOÁN!")


if __name__ == "__main__":
    verify_transformer_shapes()
