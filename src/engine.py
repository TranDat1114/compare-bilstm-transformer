"""Bộ máy huấn luyện: hàm mất mát, lịch học (LR schedule), early stopping,
ghi nhật ký (loss, accuracy, macro-F1, gradient-norm, learning-rate) theo từng
epoch/step để vẽ biểu đồ GIÁM SÁT quá trình train.
"""
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)


# ------------------------------------------------------------------- hàm mất mát
class FocalLoss(nn.Module):
    r"""Focal Loss (Lin et al., 2017):  FL(p_t) = -α_t (1-p_t)^γ log(p_t).
    Hệ số (1-p_t)^γ giảm trọng số mẫu DỄ (p_t lớn) -> mô hình tập trung mẫu khó
    và lớp thiểu số. γ=0 trở về cross-entropy có trọng số."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(cfg, class_weight, device):
    w = class_weight.to(device) if (cfg.train.use_class_weight and class_weight is not None) else None
    if cfg.train.loss == "focal":
        return FocalLoss(weight=w, gamma=cfg.train.focal_gamma)
    if cfg.train.loss in ("ce", "weighted_ce"):
        return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.train.label_smoothing)
    raise ValueError(cfg.train.loss)


# --------------------------------------------------------------- lịch học (warmup)
def make_scheduler(optimizer, cfg, steps_per_epoch):
    total = steps_per_epoch * cfg.train.epochs
    warmup = int(total * cfg.train.warmup_ratio)
    if cfg.train.scheduler == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

    def lr_lambda(step):  # tăng tuyến tính tới 1, rồi giảm tuyến tính về 0
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------- đánh giá
@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    ys, ps, probs, losses = [], [], [], []
    for x, mask, lengths, y in loader:
        x, mask, lengths, y = x.to(device), mask.to(device), lengths.to(device), y.to(device)
        logits = model(x, mask, lengths)
        if loss_fn is not None:
            losses.append(loss_fn(logits, y).item())
        p = logits.argmax(-1)
        probs.append(F.softmax(logits, -1).cpu().numpy())
        ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    prob = np.concatenate(probs)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": f1.tolist(), "per_class_precision": pr.tolist(),
        "per_class_recall": rc.tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "y_true": y_true, "y_pred": y_pred, "prob": prob,
    }


# --------------------------------------------------------------------- train
def train_model(model, loaders, cfg, device, class_weight=None, tag="model",
                verbose=True):
    model.to(device)
    loss_fn = build_loss(cfg, class_weight, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    steps_per_epoch = len(loaders["train"])
    sched = make_scheduler(opt, cfg, steps_per_epoch)

    hist = {k: [] for k in ["epoch", "train_loss", "dev_loss", "train_acc", "dev_acc",
                            "dev_macro_f1", "dev_weighted_f1", "lr", "grad_norm",
                            "epoch_time", "dev_per_class_recall"]}
    step_log = {"step": [], "loss": [], "lr": [], "grad_norm": []}
    best_f1, best_state, bad, best_epoch = -1.0, None, 0, -1
    gstep = 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train(); t0 = time.time()
        run_loss = run_correct = run_n = 0
        gnorm_epoch = []
        for x, mask, lengths, y in loaders["train"]:
            x, mask, lengths, y = x.to(device), mask.to(device), lengths.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x, mask, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step(); sched.step(); gstep += 1
            run_loss += loss.item() * y.size(0)
            run_correct += (logits.argmax(-1) == y).sum().item(); run_n += y.size(0)
            gnorm_epoch.append(float(gnorm))
            if gstep % 20 == 0:
                step_log["step"].append(gstep); step_log["loss"].append(loss.item())
                step_log["lr"].append(opt.param_groups[0]["lr"])
                step_log["grad_norm"].append(float(gnorm))
        dev = evaluate(model, loaders["dev"], device, loss_fn)
        hist["epoch"].append(epoch)
        hist["train_loss"].append(run_loss / run_n)
        hist["train_acc"].append(run_correct / run_n)
        hist["dev_loss"].append(dev["loss"]); hist["dev_acc"].append(dev["acc"])
        hist["dev_macro_f1"].append(dev["macro_f1"])
        hist["dev_weighted_f1"].append(dev["weighted_f1"])
        hist["dev_per_class_recall"].append(dev["per_class_recall"])
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["grad_norm"].append(float(np.mean(gnorm_epoch)))
        hist["epoch_time"].append(time.time() - t0)
        if verbose:
            print(f"[{tag}] ep{epoch:02d} train_loss={hist['train_loss'][-1]:.4f} "
                  f"dev_loss={dev['loss']:.4f} dev_acc={dev['acc']:.4f} "
                  f"dev_macroF1={dev['macro_f1']:.4f} lr={hist['lr'][-1]:.2e} "
                  f"({hist['epoch_time'][-1]:.1f}s)")
        # early stopping theo macro-F1 (phù hợp dữ liệu mất cân bằng)
        if dev["macro_f1"] > best_f1 + 1e-4:
            best_f1, best_epoch = dev["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.train.patience:
                if verbose:
                    print(f"[{tag}] early stop @ep{epoch} (best dev macroF1={best_f1:.4f} "
                          f"@ep{best_epoch})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": hist, "step_log": step_log,
                   "best_dev_macro_f1": best_f1, "best_epoch": best_epoch}
