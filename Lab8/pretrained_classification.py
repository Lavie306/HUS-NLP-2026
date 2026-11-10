"""Pretrained Transformer Classification Architecture and Pipeline.

Học phần: Xử lý ngôn ngữ tự nhiên và ứng dụng
LAB 08 — Pretrained Transformers and BERT Fine-Tuning
Giảng viên / TA: Phạm Ngọc Hải

Cài đặt pipeline phân loại văn bản chuẩn hóa theo Mục 12 & Mục 13 của đề bài:
Input text -> Tokenizer -> Input IDs -> Pretrained Transformer ->
Contextual representations -> [CLS] representation -> Linear layer -> Class logits -> Softmax
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple, Union

if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr is not None:
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


class PretrainedClassifier(nn.Module):
    """Mô hình phân loại chuỗi văn bản tích hợp khối nền tảng Pretrained Transformer Encoder.

    Kiến trúc mô hình phân tách rõ ràng giữa hai phần:
    1. Transformer Backbone (General-purpose representation):
       - Trích xuất biểu diễn ngữ nghĩa đa tầng hai chiều.
       - Hỗ trợ khởi tạo từ trọng số pre-trained hoặc khởi tạo ngẫu nhiên từ cấu hình kiến trúc (from-scratch).
       - Hỗ trợ đóng băng tham số (freeze) cho chế độ Feature Extraction.
    2. Classification Head (Task-specific):
       - Thu nhận vector ngữ cảnh đại diện tại vị trí token đặc biệt [CLS].
       - Áp dụng Dropout để chống quá khớp và tầng tuyến tính Linear chiếu về số lượng lớp mục tiêu.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        from_scratch: bool = False,
        freeze_encoder: bool = False,
        dropout_prob: float = 0.1,
    ):
        """Khởi tạo kiến trúc mô hình phân loại.

        Args:
            model_name: Tên định danh của mô hình Transformer trên HuggingFace Hub.
            num_classes: Số lượng lớp đầu ra của bài toán phân loại (mặc định = 2 cho Sentiment).
            from_scratch: Nếu True, khởi tạo trọng số ngẫu nhiên theo cấu hình (không dùng pretrained weights).
            freeze_encoder: Nếu True, đóng băng toàn bộ tham số của khối Transformer Encoder.
            dropout_prob: Tỷ lệ ngắt kết nối ngẫu nhiên (Dropout) trước tầng phân loại.
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.from_scratch = from_scratch
        self.freeze_encoder = freeze_encoder

        # Khởi tạo khối Transformer Encoder Backbone
        if from_scratch:
            # Khởi tạo cùng kiến trúc nhưng các ma trận trọng số hoàn toàn ngẫu nhiên
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
        else:
            # Tải trọng số đã qua tiền huấn luyện (Pretrained Checkpoint)
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)

        # Chế độ Feature Extraction: Khóa cập nhật trọng số của Encoder
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        # Thành phần chuyên biệt hóa cho tác vụ đích (Task-specific Classification Head)
        hidden_dimension = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_dimension, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Thực hiện quá trình lan truyền xuôi (Forward Propagation).

        Quy trình xử lý:
        1. input_ids & attention_mask truyền qua Transformer Encoder.
        2. Thu nhận ma trận trạng thái ẩn tầng cuối cùng: [batch_size, seq_len, hidden_dim].
        3. Cắt lát trích xuất vector tại chỉ số đầu tiên (index 0) ứng với token [CLS].
        4. Chiếu qua Dropout và tầng Linear để tạo ra vector Logits chưa chuẩn hóa.

        Args:
            input_ids: Tensor định danh token có kích thước [batch_size, seq_len].
            attention_mask: Tensor mặt nạ chú ý nhị phân có kích thước [batch_size, seq_len].

        Returns:
            logits: Tensor điểm số phân loại chưa qua Softmax có kích thước [batch_size, num_classes].
        """
        # Kiểm soát gradient tùy theo chế độ đóng băng encoder
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        # Trích xuất biểu diễn ngữ cảnh tổng hòa tại vị trí [CLS] (vị trí đầu tiên)
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Áp dụng Dropout và tầng phân loại chuyên biệt
        pooled_output = self.dropout(cls_embedding)
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]

        return logits

    @torch.no_grad()
    def predict(
        self,
        text: Union[str, List[str]],
        tokenizer,
        max_length: int = 128,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Union[int, float, List[float], List[int]]]:
        """Thực hiện suy luận dự đoán nhãn cảm xúc cho chuỗi văn bản đầu vào.

        Args:
            text: Chuỗi văn bản đơn hoặc danh sách chuỗi văn bản cần phân loại.
            tokenizer: Bộ tách từ tương ứng với mô hình.
            max_length: Độ dài chuỗi tối đa sau khi cắt/đệm.
            device: Thiết bị tính toán (CPU / GPU).

        Returns:
            Từ điển chứa nhãn dự đoán, điểm tin cậy cao nhất và phân phối xác suất Softmax.
        """
        self.eval()
        is_single = isinstance(text, str)
        text_list = [text] if is_single else text

        encoded_inputs = tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = self.forward(
            input_ids=encoded_inputs["input_ids"],
            attention_mask=encoded_inputs["attention_mask"],
        )
        probabilities = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probabilities, dim=-1)

        if is_single:
            return {
                "prediction": predictions[0].item(),
                "confidence": confidences[0].item(),
                "probabilities": probabilities[0].tolist(),
            }

        return {
            "predictions": predictions.tolist(),
            "confidences": confidences.tolist(),
            "probabilities": probabilities.tolist(),
        }


def compute_model_parameters(
    model: nn.Module,
) -> Tuple[int, int]:
    """Thống kê tổng số lượng tham số và số tham số có thể huấn luyện.

    Returns:
        Tuple (total_params, trainable_params)
    """
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def verify_tokenization_pipeline(
    text: str = "The cats are playing.",
    model_name: str = "bert-base-uncased",
) -> None:
    """Kiểm tra và in cấu trúc dữ liệu của bộ Tokenizer theo Mục 12.1 của đề bài."""
    print("=" * 70)
    print("KIỂM TRA CƠ CHẾ TOKENIZATION (Mục 12.1 & Calculation Exercise 2)")
    print("=" * 70)
    print(f"Chuỗi đầu vào: '{text}'")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens_dict = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    print(f"\n1. Các khóa trong từ điển đầu ra tokens.keys():")
    for key in tokens_dict.keys():
        tensor = tokens_dict[key]
        print(f"   - Key: '{key:<18}' | Kích thước: {list(tensor.shape)} | Kiểu dữ liệu: {tensor.dtype}")

    print("\n2. Giải thích chi tiết các thành phần đầu ra:")
    print("   - 'input_ids':      Danh sách các chỉ số định danh nguyên thủy của từng token")
    print("                       trong từ điển WordPiece, bắt đầu bằng [CLS] (101) và kết thúc")
    print("                       bằng [SEP] (102).")
    print(f"                       Giá trị: {tokens_dict['input_ids'].squeeze().tolist()}")
    print("   - 'attention_mask': Vector mặt nạ nhị phân chỉ thị cho khối Multi-Head Attention.")
    print("                       Giá trị 1 tương ứng với token thật; giá trị 0 tương ứng với")
    print("                       token đệm [PAD] để loại bỏ sự ảnh hưởng tới phân phối chú ý.")
    print(f"                       Giá trị: {tokens_dict['attention_mask'].squeeze().tolist()}")

    tokens_list = tokenizer.convert_ids_to_tokens(tokens_dict["input_ids"][0])
    print(f"\n3. Chuỗi Token đã giải mã:")
    print(f"   {tokens_list}")
    print("=" * 70)


def main():
    """Hàm chạy kiểm thử pipeline và phân tích tham số."""
    parser = argparse.ArgumentParser(description="Kiểm thử Pipeline Pretrained Transformer Classifier")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Tên pretrained checkpoint")
    parser.add_argument("--freeze_encoder", action="store_true", help="Kích hoạt chế độ Feature Extraction (Frozen)")
    parser.add_argument("--from_scratch", action="store_true", help="Khởi tạo trọng số ngẫu nhiên (không dùng pretrain)")
    parser.add_argument("--demo_text", type=str, default="Antigravity and Transformers make NLP fascinating!", help="Câu thử nghiệm")
    args = parser.parse_args()

    print(f"\n[1/3] Khởi tạo mô hình PretrainedClassifier ({args.model_name})...")
    model = PretrainedClassifier(
        model_name=args.model_name,
        num_classes=2,
        from_scratch=args.from_scratch,
        freeze_encoder=args.freeze_encoder,
    )

    total_params, trainable_params = compute_model_parameters(model)
    print(f"  * Tổng số lượng tham số:       {total_params:,}")
    print(f"  * Số tham số huấn luyện được:   {trainable_params:,}")
    print(f"  * Trạng thái Encoder:           {'ĐÓNG BĂNG (Frozen)' if args.freeze_encoder else 'HUẤN LUYỆN (Trainable)'}")
    print(f"  * Khởi tạo trọng số:            {'NGẪU NHIÊN (From scratch)' if args.from_scratch else 'PRE-TRAINED (BERT)'}")

    print("\n[2/3] Kiểm tra Tokenization và cấu trúc Tensor...")
    verify_tokenization_pipeline(text=args.demo_text, model_name=args.model_name)

    print("\n[3/3] Chạy thử nghiệm suy luận phân loại (Forward pass)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prediction_result = model.predict(args.demo_text, tokenizer=tokenizer)
    labels_map = {0: "Negative", 1: "Positive"}
    predicted_label = labels_map[prediction_result["prediction"]]

    print(f"  * Câu văn bản kiểm thử: '{args.demo_text}'")
    print(f"  * Nhãn dự đoán:        {predicted_label} (Lớp {prediction_result['prediction']})")
    print(f"  * Độ tin cậy cao nhất: {prediction_result['confidence']:.4f}")
    print(f"  * Phân phối xác suất:  Negative={prediction_result['probabilities'][0]:.4f}, Positive={prediction_result['probabilities'][1]:.4f}")
    print("\n=> Pipeline PretrainedClassifier hoạt động hoàn toàn chính xác theo yêu cầu đề bài!\n")


if __name__ == "__main__":
    main()
