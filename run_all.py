"""Driver: chạy TOÀN BỘ thí nghiệm và lưu metrics + biểu đồ.

  python run_all.py            # chạy đầy đủ (EDA + 2 mô hình chính + ablation)
  python run_all.py --quick    # rút gọn số epoch (kiểm tra nhanh pipeline)

Sinh ra:
  results/metrics/*.json , results/metrics/test_arrays.npz
  results/figures/*.png
"""
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch

from src.config import (Config, set_seed, get_device, FIG_DIR, MET_DIR, CKPT_DIR,
                        LABEL_NAMES)
from src.dataset import build_dataloaders, class_weights_from_counts
from src.models import build_model, count_params
from src.engine import train_model, evaluate
from src import plots, eda


def json_safe(o):
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items() if k not in ("y_true", "y_pred", "prob")}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def dump(obj, path):
    Path(path).write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2),
                          encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device} | torch {torch.__version__}")
    cfg = Config()
    if args.quick:
        cfg.train.epochs = 3
    set_seed(cfg.seed)
    cfg.save(MET_DIR / "config.json")

    # ----------------------------------------------------------------- EDA
    print("\n===== EDA =====")
    eda_stats, _ = eda.run_eda(cfg, FIG_DIR)
    dump(eda_stats, MET_DIR / "eda.json")
    print("vocab(clean)=%d vocab(raw)=%d  OOV clean=%s raw=%s" % (
        eda_stats["vocab_clean"], eda_stats["vocab_raw"],
        eda_stats["oov_clean"], eda_stats["oov_raw"]))

    # -------------------------------------------------- dữ liệu (cấu hình chính)
    print("\n===== Build dataloaders (clean=True) =====")
    loaders, vocab, meta = build_dataloaders(cfg)
    loaders["vocab"] = vocab
    cw = class_weights_from_counts(meta["class_counts"])
    print("class_counts=%s  class_weights=%s" % (meta["class_counts"],
                                                  [round(x, 3) for x in cw.tolist()]))
    dump({"oov": meta["oov"], "class_counts": meta["class_counts"],
          "vocab_size": meta["vocab_size"], "class_weights": cw.tolist()},
         MET_DIR / "data_meta.json")

    histories, step_logs, results, cost, arrays = {}, {}, {}, {}, {}

    # ----------------------------------------------- huấn luyện 2 mô hình chính
    for name in ("lstm", "transformer"):
        print(f"\n===== TRAIN {name.upper()} (cấu hình đầy đủ) =====")
        set_seed(cfg.seed)
        model = build_model(name, len(vocab), cfg)
        n_params = count_params(model)
        t0 = time.time()
        model, log = train_model(model, loaders, cfg, device, class_weight=cw, tag=name)
        train_time = time.time() - t0
        test = evaluate(model, loaders["test"], device)
        torch.save(model.state_dict(), CKPT_DIR / f"{name}.pt")
        histories[name] = log["history"]; step_logs[name] = log["step_log"]
        results[name] = test
        arrays[f"{name}_y_true"] = test["y_true"]; arrays[f"{name}_y_pred"] = test["y_pred"]
        arrays[f"{name}_prob"] = test["prob"]
        cost[name] = {"params": n_params,
                      "time_per_epoch": float(np.mean(log["history"]["epoch_time"])),
                      "total_train_time": train_time,
                      "best_epoch": log["best_epoch"],
                      "best_dev_macro_f1": log["best_dev_macro_f1"],
                      "best_f1": test["macro_f1"], "test_acc": test["acc"]}
        print(f"  -> TEST acc={test['acc']:.4f} macroF1={test['macro_f1']:.4f} "
              f"perclassF1={[round(x,3) for x in test['per_class_f1']]} "
              f"params={n_params/1e6:.2f}M time/ep={cost[name]['time_per_epoch']:.1f}s")

    # ------------------------------------------------------- biểu đồ chính
    plots.plot_training_curves(histories, FIG_DIR / "05_training_curves.png")
    plots.plot_overfit_gap(histories, FIG_DIR / "06_overfit_gap.png")
    plots.plot_grad_norm(step_logs, FIG_DIR / "07_grad_norm.png")
    plots.plot_confusion(results, FIG_DIR / "08_confusion.png", normalize=True)
    plots.plot_per_class_f1(results, FIG_DIR / "09_per_class_f1.png")
    plots.plot_roc_pr(results, FIG_DIR / "10_roc_pr.png")
    plots.plot_minority_recall_curve(histories, 2, FIG_DIR / "11_hate_recall.png")
    plots.plot_model_cost(cost, FIG_DIR / "12_model_cost.png")

    dump({"histories": histories, "step_logs": step_logs}, MET_DIR / "training_logs.json")
    dump(results, MET_DIR / "test_results.json")
    dump(cost, MET_DIR / "cost.json")
    np.savez_compressed(MET_DIR / "test_arrays.npz", **arrays)

    # ----------------------------------------- attention map (Transformer)
    print("\n===== Attention example =====")
    save_attention_example(loaders, vocab, cfg, device)

    # --------------------------------------------------------- ABLATION
    print("\n===== ABLATIONS =====")
    ablations = run_ablations(loaders, vocab, meta, cw, cfg, device, quick=args.quick)
    dump(ablations, MET_DIR / "ablations.json")
    build_ablation_figures(ablations, results)

    print("\nDONE. metrics ->", MET_DIR, " figures ->", FIG_DIR)


def save_attention_example(loaders, vocab, cfg, device):
    from src.models import TransformerClassifier
    model = TransformerClassifier(len(vocab), cfg).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "transformer.pt", map_location=device))
    model.eval()
    # tìm một câu HATE đủ ngắn để minh hoạ
    examples = []
    for x, mask, lengths, y in loaders["test"]:
        for i in range(len(y)):
            L = int(mask[i].sum())
            if y[i].item() == 2 and 4 <= L <= 14:
                examples.append((x[i:i+1], mask[i:i+1], lengths[i:i+1], L)); break
        if examples:
            break
    if not examples:
        for x, mask, lengths, y in loaders["test"]:
            L = int(mask[0].sum())
            examples.append((x[0:1], mask[0:1], lengths[0:1], L)); break
    x, mask, lengths, L = examples[0]
    x, mask, lengths = x.to(device), mask.to(device), lengths.to(device)
    with torch.no_grad():
        logits, attn = model(x, mask, lengths, return_attn=True)
    toks = vocab.decode(x[0, :L].cpu().tolist())
    A = attn[0].mean(0).cpu().numpy()[:L, :L]  # trung bình theo head
    pred = int(logits.argmax(-1).item())
    plots.plot_attention(toks, A, FIG_DIR / "13_attention.png",
                         title=f"Self-attention (dự đoán={LABEL_NAMES[pred]})")
    dump({"tokens": toks, "attn": A.tolist(), "pred": pred}, MET_DIR / "attention_example.json")
    print("  saved attention for tokens:", toks)


def run_ablations(loaders, vocab, meta, cw, cfg, device, quick=False):
    """Mỗi ablation: train lại với 1 thay đổi, đo macro-F1 + recall lớp HATE trên test."""
    res = {}
    abl_cfg = copy.deepcopy(cfg)
    abl_cfg.train.epochs = 3 if quick else 15  # ablation nhẹ hơn cho nhanh

    def run(name, model_name, cfg_mod, class_weight, loaders_use=None):
        set_seed(cfg_mod.seed)
        ld = loaders_use or loaders
        model = build_model(model_name, len(vocab if loaders_use is None else ld["vocab"]), cfg_mod)
        model, log = train_model(model, ld, cfg_mod, device, class_weight=class_weight,
                                 tag=name, verbose=False)
        test = evaluate(model, ld["test"], device)
        print(f"  [{name}] test macroF1={test['macro_f1']:.4f} "
              f"HATE_recall={test['per_class_recall'][2]:.3f} "
              f"OFF_recall={test['per_class_recall'][1]:.3f}")
        return {"macro_f1": test["macro_f1"], "acc": test["acc"],
                "per_class_recall": test["per_class_recall"],
                "per_class_f1": test["per_class_f1"],
                "grad_norm_steps": log["step_log"]["grad_norm"],
                "steps": log["step_log"]["step"]}

    # (A) class weight ON/OFF cho cả hai mô hình
    c = copy.deepcopy(abl_cfg)
    res["lstm_with_cw"] = run("lstm +classweight", "lstm", c, cw)
    res["lstm_no_cw"] = run("lstm -classweight", "lstm", c, None)
    res["tf_with_cw"] = run("transformer +classweight", "transformer", c, cw)
    res["tf_no_cw"] = run("transformer -classweight", "transformer", c, None)

    # (B) positional encoding ON/OFF (Transformer)
    c_pos = copy.deepcopy(abl_cfg); c_pos.model.use_pos_enc = False
    res["tf_no_posenc"] = run("transformer -posenc", "transformer", c_pos, cw)

    # (C) warmup/scheduler ON/OFF (Transformer)
    c_nw = copy.deepcopy(abl_cfg); c_nw.train.scheduler = "none"
    res["tf_no_warmup"] = run("transformer -warmup", "transformer", c_nw, cw)

    # (D) gradient clipping ON/OFF (Transformer) — quan sát chuẩn gradient
    c_ng = copy.deepcopy(abl_cfg); c_ng.train.grad_clip = 1e9
    res["tf_no_clip"] = run("transformer -gradclip", "transformer", c_ng, cw)

    # (E) text cleaning ON/OFF — cần dataloaders riêng (clean=False)
    c_raw = copy.deepcopy(abl_cfg); c_raw.data.clean = False
    ld_raw, vocab_raw, meta_raw = build_dataloaders(c_raw)
    ld_raw["vocab"] = vocab_raw
    cw_raw = class_weights_from_counts(meta_raw["class_counts"])
    res["tf_no_clean"] = run("transformer -clean", "transformer", c_raw, cw_raw, ld_raw)
    res["_meta"] = {"oov_clean": meta["oov"], "oov_raw": meta_raw["oov"],
                    "vocab_clean": meta["vocab_size"], "vocab_raw": meta_raw["vocab_size"]}
    return res


def build_ablation_figures(abl, main_results):
    # class weight
    plots.plot_ablation_bars(
        [("LSTM\nCÓ class-weight", abl["lstm_with_cw"]["macro_f1"]),
         ("LSTM\nKHÔNG class-weight", abl["lstm_no_cw"]["macro_f1"]),
         ("TF\nCÓ class-weight", abl["tf_with_cw"]["macro_f1"]),
         ("TF\nKHÔNG class-weight", abl["tf_no_cw"]["macro_f1"])],
        "macro_f1", "Ảnh hưởng của CLASS WEIGHTING tới macro-F1",
        FIG_DIR / "14_ablation_classweight_f1.png")
    # HATE recall under class weight
    plots.plot_ablation_bars(
        [("LSTM\nCÓ", abl["lstm_with_cw"]["per_class_recall"][2]),
         ("LSTM\nKHÔNG", abl["lstm_no_cw"]["per_class_recall"][2]),
         ("TF\nCÓ", abl["tf_with_cw"]["per_class_recall"][2]),
         ("TF\nKHÔNG", abl["tf_no_cw"]["per_class_recall"][2])],
        "recall", "Class-weight cứu RECALL lớp thiểu số HATE",
        FIG_DIR / "15_ablation_classweight_recall.png", ylabel="recall lớp HATE (test)")
    # positional encoding
    plots.plot_ablation_bars(
        [("Transformer\nCÓ pos-enc", abl["tf_with_cw"]["macro_f1"]),
         ("Transformer\nKHÔNG pos-enc", abl["tf_no_posenc"]["macro_f1"])],
        "macro_f1", "Positional Encoding với Transformer",
        FIG_DIR / "16_ablation_posenc.png")
    # cleaning
    plots.plot_ablation_bars(
        [("Transformer\nCÓ làm sạch", abl["tf_with_cw"]["macro_f1"]),
         ("Transformer\nKHÔNG làm sạch", abl["tf_no_clean"]["macro_f1"])],
        "macro_f1", "Tiền xử lý tiếng Việt với chất lượng",
        FIG_DIR / "17_ablation_clean.png")
    # warmup
    plots.plot_ablation_bars(
        [("Transformer\nCÓ warmup", abl["tf_with_cw"]["macro_f1"]),
         ("Transformer\nKHÔNG warmup", abl["tf_no_warmup"]["macro_f1"])],
        "macro_f1", "LR Warmup với Transformer",
        FIG_DIR / "18_ablation_warmup.png")
    # grad clip — vẽ chuẩn gradient theo bước
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(abl["tf_with_cw"]["steps"], abl["tf_with_cw"]["grad_norm_steps"],
            label="CÓ grad-clip (=1.0)", c="#1f77b4")
    ax.plot(abl["tf_no_clip"]["steps"], abl["tf_no_clip"]["grad_norm_steps"],
            label="KHÔNG grad-clip", c="#d62728", alpha=0.8)
    ax.set_title("Gradient clipping giữ chuẩn gradient bị chặn (ổn định)")
    ax.set_xlabel("step"); ax.set_ylabel("||g||₂"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "19_ablation_gradclip.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
