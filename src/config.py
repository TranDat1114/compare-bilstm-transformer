"""Cấu hình tập trung cho toàn bộ dự án ViHSD: LSTM vs Transformer.

Mọi siêu tham số (hyper-parameter) đều khai báo ở đây để đảm bảo:
  - Tính tái lập (reproducibility): cùng seed -> cùng kết quả.
  - So sánh CÔNG BẰNG: hai mô hình dùng chung tiền xử lý, từ điển, embedding-dim,
    optimizer, lịch học (schedule), batch-size, số epoch... Chỉ khác KIẾN TRÚC.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# ------------------------------------------------------------------ đường dẫn
ROOT = Path(__file__).resolve().parent.parent
DATA = {"train": ROOT / "train.csv", "dev": ROOT / "dev.csv", "test": ROOT / "test.csv"}
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
MET_DIR = RESULTS / "metrics"
CKPT_DIR = RESULTS / "checkpoints"
for _d in (FIG_DIR, MET_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]  # 0, 1, 2  (theo ViHSD)
NUM_CLASSES = 3
SEED = 42


@dataclass
class DataCfg:
    text_col: str = "free_text"
    label_col: str = "label_id"
    max_len: int = 64          # ~ phân vị 99 của độ dài câu (p99 = 60 token)
    min_freq: int = 2          # giữ token xuất hiện >= 2 lần -> phủ 95.9% số token
    lowercase: bool = True
    clean: bool = True         # bật pipeline tiền xử lý tiếng Việt (ablation: tắt)
    word_segment: bool = False # tách từ bằng pyvi (ablation: bật để so sánh)


@dataclass
class ModelCfg:
    embed_dim: int = 200
    # --- LSTM ---
    lstm_hidden: int = 128
    lstm_layers: int = 2
    # --- Transformer ---
    d_model: int = 200         # = embed_dim để dùng chung lớp embedding
    n_heads: int = 4           # 200 / 4 = 50 -> head_dim hợp lệ
    n_layers: int = 2
    dim_ff: int = 512
    use_pos_enc: bool = True   # ablation quan trọng: tắt -> attention bất biến hoán vị
    dropout: float = 0.3


@dataclass
class TrainCfg:
    batch_size: int = 64
    epochs: int = 30
    patience: int = 6          # early stopping theo macro-F1 trên tập dev
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1  # 10% số bước đầu để warmup (quan trọng cho Transformer)
    grad_clip: float = 1.0     # cắt gradient -> ổn định, chống bùng nổ gradient
    use_class_weight: bool = True   # ablation: tắt -> minority class bị bỏ quên
    loss: str = "weighted_ce"  # {"ce", "weighted_ce", "focal"}
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    scheduler: str = "warmup_linear"  # {"none", "warmup_linear"}


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    seed: int = SEED

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")


def set_seed(seed: int = SEED):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Cho kết quả ổn định hơn (đánh đổi một chút tốc độ)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
