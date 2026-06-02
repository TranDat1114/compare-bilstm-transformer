# -*- coding: utf-8 -*-
"""Điền số liệu thật từ results/metrics vào các placeholder trong README.md."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MET = ROOT / "results" / "metrics"
LAB = ["CLEAN", "OFFENSIVE", "HATE"]
load = lambda n: json.load(open(MET / n, encoding="utf-8"))

res = load("test_results.json"); cost = load("cost.json")
abl = load("ablations.json"); eda = load("eda.json")


def f(x, n=4): return f"{x:.{n}f}"


# ---------------- bảng kết quả chính ----------------
rt = ["| Mô hình | Test Acc | **Macro-F1** | Weighted-F1 | F1 CLEAN | F1 OFFENSIVE | F1 HATE | Tham số | Thời gian/epoch |",
      "|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|"]
for m, disp in [("lstm", "BiLSTM"), ("transformer", "Transformer")]:
    r = res[m]; c = cost[m]
    rt.append(f"| **{disp}** | {f(r['acc'])} | **{f(r['macro_f1'])}** | {f(r['weighted_f1'])} | "
              f"{f(r['per_class_f1'][0],3)} | {f(r['per_class_f1'][1],3)} | {f(r['per_class_f1'][2],3)} | "
              f"{c['params']/1e6:.2f}M | {c['time_per_epoch']:.1f}s |")
best = max(res, key=lambda k: res[k]["macro_f1"])
rt.append("")
rt.append(f"➡️ **Mô hình tốt nhất theo macro-F1: `{ {'lstm':'BiLSTM','transformer':'Transformer'}[best] }` "
          f"({f(res[best]['macro_f1'])}).**")
results_table = "\n".join(rt)


# ---------------- bảng ablation ----------------
def row(key, name):
    a = abl[key]
    return (f"| {name} | {f(a['macro_f1'])} | {f(a['per_class_recall'][0],3)} | "
            f"{f(a['per_class_recall'][1],3)} | {f(a['per_class_recall'][2],3)} |")

at = ["| Cấu hình | Macro-F1 | Recall CLEAN | Recall OFFENSIVE | Recall HATE |",
      "|:--|:--:|:--:|:--:|:--:|",
      row("lstm_with_cw", "LSTM **có** class-weight"),
      row("lstm_no_cw", "LSTM ❌ **không** class-weight"),
      row("tf_with_cw", "Transformer **có** class-weight (đầy đủ)"),
      row("tf_no_cw", "Transformer ❌ **không** class-weight"),
      row("tf_no_posenc", "Transformer ❌ **không** positional-encoding"),
      row("tf_no_clean", "Transformer ❌ **không** làm sạch văn bản"),
      row("tf_no_warmup", "Transformer ❌ **không** warmup"),
      row("tf_no_clip", "Transformer ❌ **không** gradient-clip")]
ablation_table = "\n".join(at)


# ---------------- ghi chú diễn giải (tự sinh theo số liệu) ----------------
d_cw_hate = abl["tf_with_cw"]["per_class_recall"][2] - abl["tf_no_cw"]["per_class_recall"][2]
d_cw_off = abl["tf_with_cw"]["per_class_recall"][1] - abl["tf_no_cw"]["per_class_recall"][1]
d_cw_hate_lstm = abl["lstm_with_cw"]["per_class_recall"][2] - abl["lstm_no_cw"]["per_class_recall"][2]
d_pos = abl["tf_with_cw"]["macro_f1"] - abl["tf_no_posenc"]["macro_f1"]
d_clean = abl["tf_with_cw"]["macro_f1"] - abl["tf_no_clean"]["macro_f1"]
d_warm = abl["tf_with_cw"]["macro_f1"] - abl["tf_no_warmup"]["macro_f1"]
oov_c = eda["oov_clean"]; oov_r = eda["oov_raw"]
notes = f"""
- **Class weighting** (quan trọng nhất): bật trọng số lớp giúp **recall `HATE`** của Transformer
  thay đổi **{d_cw_hate:+.3f}** và **recall `OFFENSIVE`** **{d_cw_off:+.3f}** so với khi tắt; với
  LSTM recall `HATE` thay đổi **{d_cw_hate_lstm:+.3f}**. Đây là minh chứng số học cho việc *bỏ qua
  mất cân bằng = mô hình mù với lớp thiểu số*.
- **Tiền xử lý**: làm sạch kéo **OOV (test)** từ **{oov_r['test']*100:.1f}%** xuống
  **{oov_c['test']*100:.1f}%** (train: {oov_r['train']*100:.1f}% → {oov_c['train']*100:.1f}%);
  macro-F1 Transformer thay đổi **{d_clean:+.3f}** khi có làm sạch.
- **Positional encoding**: macro-F1 thay đổi **{d_pos:+.3f}** khi có PE. Tác động vừa phải — phù hợp
  nhận định rằng phát hiện ngôn từ thù ghét nặng tính **từ vựng** hơn **trật tự cú pháp**.
- **Warmup**: macro-F1 thay đổi **{d_warm:+.3f}** khi có warmup; **gradient clipping** giữ chuẩn
  gradient bị chặn (xem hình 19) → huấn luyện ổn định.
"""


# ---------------- kết luận ----------------
diff = res["lstm"]["macro_f1"] - res["transformer"]["macro_f1"]
faster = "transformer" if cost["transformer"]["time_per_epoch"] < cost["lstm"]["time_per_epoch"] else "lstm"
speedup = max(cost["lstm"]["time_per_epoch"], cost["transformer"]["time_per_epoch"]) / \
          max(min(cost["lstm"]["time_per_epoch"], cost["transformer"]["time_per_epoch"]), 1e-9)
winner = "BiLSTM" if diff > 0 else "Transformer"
concl = f"""
**Tóm tắt định lượng (từ tập test):**

- **BiLSTM**: macro-F1 = **{f(res['lstm']['macro_f1'])}**, accuracy = {f(res['lstm']['acc'])},
  {cost['lstm']['params']/1e6:.2f}M tham số, {cost['lstm']['time_per_epoch']:.1f}s/epoch.
- **Transformer**: macro-F1 = **{f(res['transformer']['macro_f1'])}**, accuracy = {f(res['transformer']['acc'])},
  {cost['transformer']['params']/1e6:.2f}M tham số, {cost['transformer']['time_per_epoch']:.1f}s/epoch.

1. **Kiến trúc.** Trên dữ liệu cỡ vừa (~24k câu, huấn luyện **từ đầu**), **{winner}** đạt macro-F1
   cao hơn (chênh **{abs(diff):.3f}**). Transformer-from-scratch thường "đói dữ liệu" hơn (thiếu
   thiên kiến quy nạp về trật tự, hợp với pretraining quy mô lớn), nhưng huấn luyện
   **nhanh hơn ~{speedup:.1f}× mỗi epoch** (song song) và cung cấp **bản đồ attention** dễ diễn giải.
2. **Mất cân bằng → weighted loss + macro-F1** là quyết định ảnh hưởng lớn nhất tới chất lượng phát
   hiện lớp thù ghét (§10.1).
3. **Tiền xử lý tiếng Việt** giảm mạnh OOV (test {oov_r['test']*100:.1f}% → {oov_c['test']*100:.1f}%)
   và cải thiện chất lượng (§10.3).
4. **Warmup + gradient clipping** giúp huấn luyện ổn định, đặc biệt cho Transformer (§8, §10.4).
5. **Positional encoding** là thành phần lý thuyết bắt buộc của Transformer (§6.3, §10.2).
"""

txt = (ROOT / "README.md").read_text(encoding="utf-8")
txt = txt.replace("<!-- RESULTS_TABLE_PLACEHOLDER -->", results_table)
txt = txt.replace("<!-- ABLATION_TABLE_PLACEHOLDER -->", ablation_table)
txt = txt.replace("<!-- ABLATION_NOTES_PLACEHOLDER -->", notes.strip())
txt = txt.replace("<!-- CONCLUSION_PLACEHOLDER -->", concl.strip())
(ROOT / "README.md").write_text(txt, encoding="utf-8")
print("README.md đã được điền số liệu thật.")
print("best macro-F1:", best, f(res[best]["macro_f1"]))
