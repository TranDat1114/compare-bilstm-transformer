"""Xây dựng từ điển (vocabulary) ánh xạ token -> chỉ số (index).

Lý thuyết: một câu là chuỗi token (t_1,...,t_L). Mô hình neural không nhận chữ,
mà nhận chỉ số nguyên -> sau đó tra bảng nhúng (embedding lookup) E ∈ R^{V×d}.
Ta cần:
  - <pad> (idx 0): token độn cho đủ độ dài, BỊ CHE (mask) khi tính toán.
  - <unk> (idx 1): đại diện mọi token hiếm / ngoài từ điển (Out-Of-Vocabulary).

Ngưỡng min_freq: chỉ giữ token xuất hiện >= min_freq lần. Đây là đánh đổi:
  - min_freq thấp  -> vocab lớn, nhiều tham số, dễ overfit token hiếm.
  - min_freq cao   -> vocab nhỏ, nhiều OOV -> mất thông tin.
"""
from collections import Counter
import json

PAD, UNK = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        s = self.stoi
        return [s.get(t, UNK_IDX) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] if 0 <= i < len(self.itos) else UNK for i in ids]

    @classmethod
    def build(cls, token_lists, min_freq=2, max_size=None):
        counter = Counter()
        for toks in token_lists:
            counter.update(toks)
        itos = [PAD, UNK]
        # sắp xếp theo tần suất giảm dần rồi alphabet (ổn định, tái lập)
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        for tok, freq in ordered:
            if freq < min_freq:
                break
            itos.append(tok)
            if max_size and len(itos) >= max_size:
                break
        v = cls(itos)
        v.counter = counter
        return v

    def oov_rate(self, token_lists):
        """Tỉ lệ token rơi vào <unk> (đo mức mất thông tin của từ điển)."""
        total = unk = 0
        for toks in token_lists:
            for t in toks:
                total += 1
                if t not in self.stoi:
                    unk += 1
        return unk / max(total, 1)

    def save(self, path):
        json.dump(self.itos, open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path, encoding="utf-8")))
