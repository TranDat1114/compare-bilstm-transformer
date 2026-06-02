"""Hai kiến trúc so sánh: BiLSTM và Transformer-Encoder (cài từ đầu).

Cả hai DÙNG CHUNG:
  - lớp Embedding (V x d), cùng d = embed_dim
  - cùng cách gộp chuỗi (masked mean-pooling) -> vector câu -> MLP phân loại
Khác biệt DUY NHẤT là cách trộn thông tin theo thời gian:
  - LSTM: hồi quy (recurrence), trạng thái h_t phụ thuộc h_{t-1} -> tuần tự.
  - Transformer: self-attention, mọi token "nhìn" mọi token song song.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- utils
def masked_mean(x, mask):
    """x: (B,L,D), mask: (B,L) bool. Trung bình theo thời gian, bỏ qua pad."""
    m = mask.unsqueeze(-1).float()
    s = (x * m).sum(1)
    d = m.sum(1).clamp(min=1.0)
    return s / d


def masked_max(x, mask):
    neg = torch.finfo(x.dtype).min
    xm = x.masked_fill(~mask.unsqueeze(-1), neg)
    out = xm.max(1).values
    # câu rỗng (không token thật nào) -> trả 0 thay vì finfo.min (tránh tràn số -> NaN)
    valid = mask.any(1, keepdim=True)
    return torch.where(valid, out, torch.zeros_like(out))


# ============================================================================ BiLSTM
class BiLSTMClassifier(nn.Module):
    r"""Long Short-Term Memory (hai chiều).

    Mỗi bước thời gian cập nhật bằng các cổng (gates):
        f_t = σ(W_f[h_{t-1},x_t]+b_f)         (forget)
        i_t = σ(W_i[h_{t-1},x_t]+b_i)         (input)
        \tilde c_t = tanh(W_c[h_{t-1},x_t]+b_c)
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ \tilde c_t (cell state)
        o_t = σ(W_o[h_{t-1},x_t]+b_o)         (output)
        h_t = o_t ⊙ tanh(c_t)
    Cổng quên f_t cho phép giữ/loại thông tin -> giảm vanishing gradient so với RNN.
    Hai chiều: nối h_t^→ và h_t^← để mỗi vị trí thấy cả ngữ cảnh trái và phải.
    """

    def __init__(self, vocab_size, cfg):
        super().__init__()
        d = cfg.model.embed_dim
        H = cfg.model.lstm_hidden
        self.embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.emb_drop = nn.Dropout(cfg.model.dropout)
        self.lstm = nn.LSTM(d, H, num_layers=cfg.model.lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=cfg.model.dropout if cfg.model.lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(4 * H, 2 * H), nn.ReLU(), nn.Dropout(cfg.model.dropout),
            nn.Linear(2 * H, 3))

    def forward(self, x, mask, lengths=None, return_repr=False):
        e = self.emb_drop(self.embed(x))                       # (B,L,d)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                e, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.size(1))
        else:
            out, _ = self.lstm(e)                              # (B,L,2H)
        pooled = torch.cat([masked_mean(out, mask), masked_max(out, mask)], dim=-1)
        logits = self.head(pooled)
        return (logits, pooled) if return_repr else logits


# ===================================================================== Transformer
class PositionalEncoding(nn.Module):
    r"""Mã hoá vị trí dạng sin/cos (Vaswani et al., 2017).
        PE(pos,2i)   = sin(pos / 10000^{2i/d})
        PE(pos,2i+1) = cos(pos / 10000^{2i/d})
    Self-attention BẤT BIẾN với hoán vị (permutation-invariant): nếu không cộng
    thông tin vị trí, "tôi ghét bạn" và "bạn ghét tôi" có cùng biểu diễn tập hợp.
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))            # (1,max_len,d)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    r"""Scaled Dot-Product Attention nhiều đầu (cài tay để minh hoạ toán học).

        Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V
    Chia d_model thành h đầu, mỗi đầu d_k = d_model/h, học quan hệ khác nhau,
    rồi nối lại và chiếu tuyến tính. Chia sqrt(d_k) để ổn định gradient của softmax.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B,h,L,dk)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B,h,L,L)
        if key_padding_mask is not None:
            # key_padding_mask: (B,L) True=pad -> đặt -inf để softmax ~ 0
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = scores.softmax(dim=-1)
        attn = self.drop(attn)
        ctx = (attn @ v).transpose(1, 2).reshape(B, L, D)     # nối các đầu
        out = self.proj(ctx)
        return (out, attn) if need_weights else (out, None)


class EncoderLayer(nn.Module):
    """Lớp encoder kiểu Pre-LN: x + Attn(LN(x)); x + FFN(LN(x)). Pre-LN ổn định hơn."""

    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        a, attn = self.attn(self.ln1(x), key_padding_mask, need_weights)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, attn


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        d = cfg.model.d_model
        self.d = d
        self.use_pos = cfg.model.use_pos_enc
        self.embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = PositionalEncoding(d, max_len=cfg.data.max_len + 2)
        self.emb_drop = nn.Dropout(cfg.model.dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d, cfg.model.n_heads, cfg.model.dim_ff, cfg.model.dropout)
            for _ in range(cfg.model.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(cfg.model.dropout),
            nn.Linear(d, 3))

    def forward(self, x, mask, lengths=None, return_repr=False, return_attn=False):
        key_padding = ~mask                                    # True = pad
        e = self.embed(x) * math.sqrt(self.d)                  # scale embedding
        if self.use_pos:
            e = self.pos(e)
        h = self.emb_drop(e)
        attns = []
        for i, layer in enumerate(self.layers):
            need = return_attn and (i == len(self.layers) - 1)
            h, a = layer(h, key_padding_mask=key_padding, need_weights=need)
            if need:
                attns.append(a)
        h = self.norm(h)
        pooled = torch.cat([masked_mean(h, mask), masked_max(h, mask)], dim=-1)
        logits = self.head(pooled)
        out = [logits]
        if return_repr:
            out.append(pooled)
        if return_attn:
            out.append(attns[-1] if attns else None)
        return logits if len(out) == 1 else tuple(out)


def build_model(name, vocab_size, cfg):
    name = name.lower()
    if name == "lstm":
        return BiLSTMClassifier(vocab_size, cfg)
    if name == "transformer":
        return TransformerClassifier(vocab_size, cfg)
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
