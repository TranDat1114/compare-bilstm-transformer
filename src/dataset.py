"""Nạp dữ liệu, áp tiền xử lý, mã hoá số, tạo DataLoader với padding + mask.

Padding & masking (lý thuyết):
  Các câu dài ngắn khác nhau, nhưng tensor cần hình chữ nhật (B x L). Ta độn <pad>
  cho bằng max-len trong batch. Để mô hình KHÔNG học từ vị trí độn, ta truyền
  attention/padding mask:  mask[i,j] = 1 nếu token thật, = 0 nếu là <pad>.
  - LSTM dùng mask để pooling có trọng số (bỏ qua pad).
  - Transformer dùng key_padding_mask để self-attention gán -inf cho vị trí pad
    => softmax ~ 0 => pad không đóng góp vào biểu diễn.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .vocab import Vocab, PAD_IDX, UNK_IDX
from .preprocessing import build_preprocessor
from .config import DATA


def load_split(path, text_col, label_col, preprocess):
    df = pd.read_csv(path)
    df[text_col] = df[text_col].fillna("").astype(str)
    df = df[df[label_col].notna()].copy()
    df[label_col] = df[label_col].astype(int)
    texts = [preprocess(t) for t in df[text_col].tolist()]
    tokens = [t.split() for t in texts]
    labels = df[label_col].tolist()
    return texts, tokens, labels


class TextDataset(Dataset):
    def __init__(self, token_lists, labels, vocab: Vocab, max_len: int):
        self.ids = [vocab.encode(toks)[:max_len] for toks in token_lists]
        self.labels = list(labels)
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch, pad_idx=PAD_IDX):
    """Độn batch về cùng độ dài + tạo mask + lengths (cho pack LSTM)."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([max(len(s), 1) for s in seqs], dtype=torch.long)
    maxlen = int(lengths.max())
    B = len(seqs)
    x = torch.full((B, maxlen), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) == 0:
            s = [UNK_IDX]                       # câu rỗng -> 1 token <unk> thật
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    mask = (x != pad_idx)                      # True = token thật
    y = torch.tensor(labels, dtype=torch.long)
    return x, mask, lengths, y


def build_dataloaders(cfg):
    """Trả về (loaders, vocab, meta). Vocab CHỈ xây trên train (tránh rò rỉ dữ liệu)."""
    pre = build_preprocessor(clean=cfg.data.clean, lowercase=cfg.data.lowercase,
                             word_seg=cfg.data.word_segment)
    data = {}
    for split, path in [("train", DATA["train"]), ("dev", DATA["dev"]),
                        ("test", DATA["test"])]:
        data[split] = load_split(path, cfg.data.text_col, cfg.data.label_col, pre)

    vocab = Vocab.build(data["train"][1], min_freq=cfg.data.min_freq)
    oov = {s: vocab.oov_rate(data[s][1]) for s in data}

    datasets, loaders = {}, {}
    for split in ("train", "dev", "test"):
        ds = TextDataset(data[split][1], data[split][2], vocab, cfg.data.max_len)
        datasets[split] = ds
        loaders[split] = DataLoader(
            ds, batch_size=cfg.train.batch_size, shuffle=(split == "train"),
            collate_fn=collate, num_workers=0, drop_last=False)

    # phân phối lớp trên train -> dùng tính class weight
    y_train = np.array(data["train"][2])
    class_counts = np.bincount(y_train, minlength=3)
    meta = {"oov": oov, "class_counts": class_counts.tolist(),
            "vocab_size": len(vocab), "data": data}
    return loaders, vocab, meta


def class_weights_from_counts(counts, num_classes=3):
    """w_c = N / (K * n_c)  (cân bằng nghịch tần suất). Trả về tensor float."""
    counts = np.asarray(counts, dtype=np.float64)
    N = counts.sum()
    w = N / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)
