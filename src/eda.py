"""Phân tích khám phá dữ liệu (EDA) -> trả về số liệu + sinh biểu đồ."""
from collections import Counter
import numpy as np
import pandas as pd

from .config import DATA, LABEL_NAMES
from .preprocessing import build_preprocessor, clean_text
from .vocab import Vocab
from . import plots


def run_eda(cfg, fig_dir):
    pre_clean = build_preprocessor(clean=True, lowercase=True)
    pre_raw = build_preprocessor(clean=False, lowercase=True)

    raw_counts, lengths, tokens_clean = {}, {}, {}
    dfs = {}
    for split in ("train", "dev", "test"):
        df = pd.read_csv(DATA[split])
        df[cfg.data.text_col] = df[cfg.data.text_col].fillna("").astype(str)
        df = df[df[cfg.data.label_col].notna()].copy()
        df[cfg.data.label_col] = df[cfg.data.label_col].astype(int)
        dfs[split] = df
        raw_counts[split] = np.bincount(df[cfg.data.label_col], minlength=3).tolist()
        cl = [pre_clean(t) for t in df[cfg.data.text_col]]
        tokens_clean[split] = [c.split() for c in cl]
        lengths[split] = np.array([len(c.split()) for c in cl])

    # từ điển trên train (đã làm sạch) để tính OOV
    vocab_clean = Vocab.build(tokens_clean["train"], min_freq=cfg.data.min_freq)
    tokens_raw = {s: [pre_raw(t).split() for t in dfs[s][cfg.data.text_col]]
                  for s in dfs}
    vocab_raw = Vocab.build(tokens_raw["train"], min_freq=cfg.data.min_freq)
    oov_clean = {s: vocab_clean.oov_rate(tokens_clean[s]) for s in dfs}
    oov_raw = {s: vocab_raw.oov_rate(tokens_raw[s]) for s in dfs}

    # top token theo lớp (loại token đặc biệt và 1 ký tự)
    top_by_class = {}
    df = dfs["train"]
    stop = {"<", ">", "_", ".", ",", "!", "?", "là", "có", "và", "của", "cho",
            "thì", "mà", "này", "các", "một", "đã", "không", "với", "được",
            "ở", "ra", "đi", "cũng", "như", "lại", "nó", "khi", "nên", "rồi"}
    for lab_id, lab in enumerate(LABEL_NAMES):
        c = Counter()
        sub = df[df[cfg.data.label_col] == lab_id][cfg.data.text_col]
        for t in sub:
            for w in clean_text(t).split():
                if w not in stop and len(w) > 1 and not w.startswith("<"):
                    c[w] += 1
        top_by_class[lab] = c.most_common(12)

    figs = {}
    figs["label_dist"] = plots.plot_label_distribution(raw_counts, fig_dir / "01_label_distribution.png")
    figs["length"] = plots.plot_length_hist(lengths, cfg.data.max_len, fig_dir / "02_length_hist.png")
    figs["oov"] = plots.plot_oov_clean_vs_raw(oov_clean, oov_raw, fig_dir / "03_oov_clean_vs_raw.png")
    figs["top_tokens"] = plots.plot_top_tokens_per_class(top_by_class, fig_dir / "04_top_tokens.png")

    stats = {
        "label_counts": raw_counts,
        "label_pct": {s: (np.array(c) / sum(c) * 100).round(2).tolist()
                      for s, c in raw_counts.items()},
        "length": {s: {"mean": float(L.mean()), "median": float(np.median(L)),
                       "p90": float(np.percentile(L, 90)),
                       "p99": float(np.percentile(L, 99)), "max": int(L.max())}
                   for s, L in lengths.items()},
        "vocab_clean": len(vocab_clean), "vocab_raw": len(vocab_raw),
        "oov_clean": {s: round(v, 4) for s, v in oov_clean.items()},
        "oov_raw": {s: round(v, 4) for s, v in oov_raw.items()},
        "top_by_class": {k: v for k, v in top_by_class.items()},
    }
    return stats, figs
