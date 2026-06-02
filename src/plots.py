"""Tất cả biểu đồ giám sát & so sánh. Mỗi hàm lưu file PNG vào results/figures/."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"lstm": "#1f77b4", "transformer": "#d62728"}
LABELS = ["CLEAN", "OFFENSIVE", "HATE"]


def _save(fig, path, dpi=130):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------- EDA
def plot_label_distribution(counts_dict, path):
    fig, axes = plt.subplots(1, len(counts_dict), figsize=(4.2 * len(counts_dict), 3.6))
    if len(counts_dict) == 1:
        axes = [axes]
    for ax, (split, counts) in zip(axes, counts_dict.items()):
        total = sum(counts)
        bars = ax.bar(LABELS, counts, color=["#2ca02c", "#ff7f0e", "#d62728"])
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, c, f"{c}\n{c/total*100:.1f}%",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{split} (n={total})"); ax.set_ylabel("Số mẫu")
        ax.set_ylim(0, max(counts) * 1.18)
    fig.suptitle("Phân phối nhãn — dữ liệu MẤT CÂN BẰNG mạnh", fontweight="bold")
    return _save(fig, path)


def plot_length_hist(lengths_by_split, max_len, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, L in lengths_by_split.items():
        ax.hist(np.clip(L, 0, 120), bins=60, alpha=0.5, label=f"{split} (median={np.median(L):.0f})")
    ax.axvline(max_len, color="red", ls="--", lw=2, label=f"max_len={max_len} (cắt ở đây)")
    ax.set_xlabel("Độ dài câu (số token)"); ax.set_ylabel("Tần suất")
    ax.set_title("Phân phối độ dài câu — chọn max_len ~ phân vị 99"); ax.legend()
    return _save(fig, path)


def plot_oov_clean_vs_raw(oov_clean, oov_raw, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    splits = list(oov_clean.keys())
    x = np.arange(len(splits)); w = 0.35
    ax.bar(x - w/2, [oov_raw[s]*100 for s in splits], w, label="KHÔNG làm sạch", color="#bbbbbb")
    ax.bar(x + w/2, [oov_clean[s]*100 for s in splits], w, label="CÓ làm sạch", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(splits); ax.set_ylabel("Tỉ lệ OOV (%)")
    ax.set_title("Tiền xử lý làm GIẢM OOV (token rơi vào <unk>)"); ax.legend()
    for i, s in enumerate(splits):
        ax.text(i - w/2, oov_raw[s]*100, f"{oov_raw[s]*100:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, oov_clean[s]*100, f"{oov_clean[s]*100:.1f}", ha="center", va="bottom", fontsize=8)
    return _save(fig, path)


def plot_top_tokens_per_class(top_dict, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for ax, lab, col in zip(axes, LABELS, colors):
        toks, freqs = zip(*top_dict[lab])
        ax.barh(range(len(toks)), freqs, color=col)
        ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks, fontsize=8)
        ax.invert_yaxis(); ax.set_title(f"Top token — {lab}")
    fig.suptitle("Token đặc trưng theo lớp (sau tiền xử lý)", fontweight="bold")
    return _save(fig, path)


# ----------------------------------------------------------- giám sát training
def plot_training_curves(histories, path):
    """histories: {name: history_dict}. Vẽ loss, acc, macro-F1, LR theo epoch."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, h in histories.items():
        c = PALETTE.get(name, None); ep = h["epoch"]
        axes[0, 0].plot(ep, h["train_loss"], c=c, ls="-", label=f"{name} train")
        axes[0, 0].plot(ep, h["dev_loss"], c=c, ls="--", label=f"{name} dev")
        axes[0, 1].plot(ep, h["train_acc"], c=c, ls="-", label=f"{name} train")
        axes[0, 1].plot(ep, h["dev_acc"], c=c, ls="--", label=f"{name} dev")
        axes[1, 0].plot(ep, h["dev_macro_f1"], c=c, marker="o", ms=3, label=f"{name}")
        axes[1, 1].plot(ep, h["lr"], c=c, label=f"{name}")
    axes[0, 0].set_title("Loss (— train, -- dev)"); axes[0, 0].set_xlabel("epoch")
    axes[0, 1].set_title("Accuracy"); axes[0, 1].set_xlabel("epoch")
    axes[1, 0].set_title("Dev macro-F1 (chỉ số chính)"); axes[1, 0].set_xlabel("epoch")
    axes[1, 1].set_title("Learning rate (warmup + decay)"); axes[1, 1].set_xlabel("epoch")
    for ax in axes.ravel():
        ax.legend(fontsize=8)
    fig.suptitle("Giám sát quá trình huấn luyện", fontweight="bold")
    return _save(fig, path)


def plot_overfit_gap(histories, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in histories.items():
        gap = np.array(h["train_acc"]) - np.array(h["dev_acc"])
        ax.plot(h["epoch"], gap, marker="o", ms=3, label=name, c=PALETTE.get(name))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Khoảng cách train−dev accuracy (đo OVERFIT)")
    ax.set_xlabel("epoch"); ax.set_ylabel("train_acc − dev_acc"); ax.legend()
    return _save(fig, path)


def plot_grad_norm(step_logs, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, s in step_logs.items():
        ax.plot(s["step"], s["grad_norm"], alpha=0.8, label=name, c=PALETTE.get(name))
    ax.set_title("Chuẩn gradient theo bước (ổn định nhờ gradient clipping)")
    ax.set_xlabel("training step"); ax.set_ylabel("||g||₂ (trước khi clip)"); ax.legend()
    return _save(fig, path)


def plot_confusion(results, path, normalize=True):
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = np.array(results[name]["confusion"], dtype=float)
        if normalize:
            cm = cm / cm.sum(1, keepdims=True).clip(min=1)
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else ".0f", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax, cbar=False,
                    vmin=0, vmax=1 if normalize else None)
        ax.set_title(f"{name}  (acc={results[name]['acc']:.3f}, "
                     f"macroF1={results[name]['macro_f1']:.3f})")
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    fig.suptitle("Ma trận nhầm lẫn (chuẩn hoá theo hàng = recall mỗi lớp)",
                 fontweight="bold")
    return _save(fig, path)


def plot_per_class_f1(results, path):
    names = list(results.keys())
    x = np.arange(3); w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, name in enumerate(names):
        f1 = results[name]["per_class_f1"]
        bars = ax.bar(x + i * w - 0.4 + w/2, f1, w, label=name, color=PALETTE.get(name))
        for b, v in zip(bars, f1):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1"); ax.set_title("F1 theo từng lớp"); ax.legend()
    return _save(fig, path)


def plot_roc_pr(results, path):
    names = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in names:
        r = results[name]
        y = label_binarize(r["y_true"], classes=[0, 1, 2])
        prob = r["prob"]
        for k, lab in enumerate(LABELS):
            fpr, tpr, _ = roc_curve(y[:, k], prob[:, k])
            axes[0].plot(fpr, tpr, label=f"{name}-{lab} (AUC={auc(fpr,tpr):.2f})",
                         lw=1.4)
            pr, rc, _ = precision_recall_curve(y[:, k], prob[:, k])
            axes[1].plot(rc, pr, label=f"{name}-{lab}", lw=1.4)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_title("ROC (one-vs-rest)"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[1].set_title("Precision–Recall"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    for ax in axes:
        ax.legend(fontsize=7)
    return _save(fig, path)


def plot_minority_recall_curve(histories, class_idx, path):
    """Recall của lớp thiểu số theo epoch — minh hoạ class-weight giúp gì."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in histories.items():
        rec = [r[class_idx] for r in h["dev_per_class_recall"]]
        ax.plot(h["epoch"], rec, marker="o", ms=3, label=name)
    ax.set_title(f"Dev recall lớp '{LABELS[class_idx]}' theo epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("recall"); ax.legend()
    return _save(fig, path)


def plot_ablation_bars(ablation, metric, title, path, ylabel="macro-F1 (test)"):
    """ablation: list of (label, value). Vẽ cột so sánh CÓ/KHÔNG dùng phương pháp."""
    labels = [a[0] for a in ablation]; vals = [a[1] for a in ablation]
    colors = ["#d62728" if "KHÔNG" in l or "no-" in l.lower() else "#1f77b4" for l in labels]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(labels)), 4.2))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.18)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, path)


def plot_model_cost(stats, path):
    """stats: {name: {'params':..,'time_per_epoch':..,'best_f1':..}}"""
    names = list(stats.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(names, [stats[n]["params"]/1e6 for n in names],
                color=[PALETTE.get(n) for n in names])
    axes[0].set_title("Số tham số (triệu)"); axes[0].set_ylabel("M params")
    axes[1].bar(names, [stats[n]["time_per_epoch"] for n in names],
                color=[PALETTE.get(n) for n in names])
    axes[1].set_title("Thời gian / epoch (giây)")
    axes[2].bar(names, [stats[n]["best_f1"] for n in names],
                color=[PALETTE.get(n) for n in names])
    axes[2].set_title("Test macro-F1"); axes[2].set_ylim(0, 1)
    for ax in axes:
        for p in ax.patches:
            ax.text(p.get_x()+p.get_width()/2, p.get_height(), f"{p.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Đánh đổi: dung lượng / tốc độ / chất lượng", fontweight="bold")
    return _save(fig, path)


def plot_attention(tokens, attn_matrix, path, title="Self-attention (lớp cuối, trung bình các đầu)"):
    """attn_matrix: (L,L) trung bình theo head. tokens: list[str] độ dài L."""
    L = len(tokens)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * L), max(5, 0.5 * L)))
    sns.heatmap(attn_matrix[:L, :L], xticklabels=tokens, yticklabels=tokens,
                cmap="viridis", ax=ax, cbar=True)
    ax.set_title(title); ax.set_xlabel("Token được chú ý (key)")
    ax.set_ylabel("Token truy vấn (query)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    return _save(fig, path)
