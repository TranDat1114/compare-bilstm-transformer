# -*- coding: utf-8 -*-
"""Sinh notebook trình bày: nạp metrics đã lưu (results/metrics) và VẼ LẠI mọi biểu
đồ ngay trong cell (live), kèm lý thuyết toán học. Không huấn luyện lại khi chạy
nbconvert (RUN_TRAINING=False) -> nhanh & ổn định."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(t): cells.append(new_markdown_cell(t.strip("\n")))
def code(t): cells.append(new_code_cell(t.strip("\n")))

# ============================================================== 0. TIÊU ĐỀ
md(r"""
# So sánh **BiLSTM** và **Transformer** cho phát hiện ngôn từ thù ghét tiếng Việt (ViHSD)

> Đồ án môn *Toán cho AI* — xây dựng, huấn luyện và **so sánh** hai kiến trúc học sâu trên cùng
> một bộ dữ liệu, đồng thời phân tích **các phương pháp xử lý dữ liệu / huấn luyện** bằng thí
> nghiệm cắt bỏ (*ablation*) và biểu đồ minh hoạ.

**Bài toán.** Phân loại 3 lớp một bình luận mạng xã hội:
`0 = CLEAN` (bình thường), `1 = OFFENSIVE` (công kích), `2 = HATE` (thù ghét).

**Notebook này làm gì?** Nạp lại toàn bộ kết quả đã huấn luyện (trong `results/metrics/`)
và **vẽ lại tại chỗ** các biểu đồ giám sát & so sánh. Toàn bộ mã huấn luyện nằm trong gói
`src/` và `run_all.py`. Đặt `RUN_TRAINING=True` ở cell dưới để huấn luyện lại từ đầu trên GPU.

Mục lục:
1. Cấu hình & nạp dữ liệu kết quả
2. Khám phá dữ liệu (EDA) — vì sao mất cân bằng & nhiễu là vấn đề
3. Tiền xử lý tiếng Việt (demo trực tiếp) + lý thuyết
4. Hai kiến trúc & nền tảng toán học (LSTM, Self-Attention, Positional Encoding)
5. Phương pháp huấn luyện (loss có trọng số/focal, warmup, gradient clipping, early stopping)
6. Giám sát huấn luyện (loss/acc/F1, gradient-norm, learning-rate)
7. Đánh giá trên test (confusion, F1 theo lớp, ROC/PR) + bản đồ attention
8. **Ablation** — có/không dùng từng phương pháp thì sao?
9. Kết luận
""")

# ============================================================== 1. SETUP
md(r"""
## 1. Cấu hình & nạp kết quả

Mọi biểu đồ bên dưới được **sinh lại trực tiếp** từ dữ liệu số đã lưu (`.json`, `.npz`) bằng
chính các hàm trong `src/plots.py` — đảm bảo biểu đồ trong notebook trùng khớp báo cáo.
""")
code(r"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))          # để import gói src
import numpy as np, pandas as pd
from pathlib import Path
from IPython.display import Image, display, Markdown
from src import plots
from src.config import FIG_DIR, MET_DIR, LABEL_NAMES

RUN_TRAINING = False   # True -> chạy lại run_all.py (cần GPU, ~25-30 phút)
if RUN_TRAINING:
    import subprocess; subprocess.run([sys.executable, "run_all.py"], check=True)

def load(name): return json.load(open(MET_DIR / name, encoding="utf-8"))
def show(path): display(Image(filename=str(path)))

eda      = load("eda.json")
meta     = load("data_meta.json")
logs     = load("training_logs.json")
results  = load("test_results.json")
cost     = load("cost.json")
abl      = load("ablations.json")
attn_ex  = load("attention_example.json")
arrays   = np.load(MET_DIR / "test_arrays.npz")
# gắn lại mảng xác suất vào results để vẽ ROC/PR (json đã lược bỏ mảng lớn)
for m in ("lstm", "transformer"):
    results[m]["y_true"] = arrays[f"{m}_y_true"]
    results[m]["y_pred"] = arrays[f"{m}_y_pred"]
    results[m]["prob"]   = arrays[f"{m}_prob"]
print("Đã nạp kết quả cho:", list(results.keys()))
print("Kích thước từ điển:", meta["vocab_size"], "| trọng số lớp:",
      [round(w,3) for w in meta["class_weights"]])
""")

# ============================================================== 2. EDA
md(r"""
## 2. Khám phá dữ liệu (EDA)

Ba tập `train / dev / test` với cột `free_text` (bình luận) và `label_id` (nhãn 0/1/2).
Hai đặc điểm chi phối mọi quyết định thiết kế:

### 2.1 Mất cân bằng lớp nghiêm trọng
Lớp `CLEAN` chiếm ~83%, hai lớp cần phát hiện (`OFFENSIVE`, `HATE`) chỉ ~7% và ~11%.

> **Hệ quả toán học.** Một bộ phân loại "luôn đoán CLEAN" đạt **accuracy ≈ 83%** nhưng **vô
> dụng** (không bắt được câu thù ghét nào). Vì vậy ta KHÔNG dùng accuracy làm thước đo chính,
> mà dùng **macro-F1** — trung bình **không trọng số** của F1 ba lớp, nên lớp thiểu số có
> tiếng nói ngang lớp đa số:
> $$\text{macro-F1}=\frac{1}{K}\sum_{c=1}^{K} F1_c,\qquad
>   F1_c=\frac{2\,P_c R_c}{P_c+R_c},\quad P_c=\frac{TP_c}{TP_c+FP_c},\ R_c=\frac{TP_c}{TP_c+FN_c}.$$
""")
code(r"""
df_dist = pd.DataFrame(eda["label_counts"], index=LABEL_NAMES)
df_pct  = pd.DataFrame(eda["label_pct"],   index=LABEL_NAMES).round(2)
print("Số mẫu mỗi lớp:\n", df_dist, "\n\nTỉ lệ % mỗi lớp:\n", df_pct)
show(FIG_DIR / "01_label_distribution.png")
""")
md(r"""
### 2.2 Độ dài câu — chọn `max_len`
Phần lớn câu rất ngắn (trung vị ~8 token) nhưng đuôi phải dài. Ta cắt/đệm về
`max_len = 64` (~phân vị 99) để cân bằng giữa **giữ thông tin** và **chi phí tính toán**
(self-attention tốn $O(L^2)$ theo độ dài $L$).
""")
code(r"""
print(pd.DataFrame(eda["length"]).T)
show(FIG_DIR / "02_length_hist.png")
""")
md(r"""
### 2.3 Nhiễu đặc thù tiếng Việt → động lực cho tiền xử lý
Dữ liệu là bình luận thật: teencode (`ko`→`không`), emoji/emoticon (`:))`, `❤️`), kéo dài ký
tự (`quááá`). Token `:))` và `ko` nằm trong nhóm xuất hiện nhiều nhất. Biểu đồ dưới so sánh
**tỉ lệ OOV** (token rơi vào `<unk>`) khi **có** và **không** tiền xử lý — tiền xử lý kéo OOV
giảm rõ rệt (xem mục 3 & 8).
""")
code(r"""
print("OOV (có làm sạch):", eda["oov_clean"])
print("OOV (không làm sạch):", eda["oov_raw"])
print("Vocab: clean =", eda["vocab_clean"], "| raw =", eda["vocab_raw"])
show(FIG_DIR / "03_oov_clean_vs_raw.png")
show(FIG_DIR / "04_top_tokens.png")
""")

# ============================================================== 3. PREPROCESS
md(r"""
## 3. Tiền xử lý văn bản tiếng Việt — lý thuyết & demo

Pipeline (`src/preprocessing.py`) gồm các bước, mỗi bước có **lý do** và **rủi ro nếu bỏ**:

| Bước | Việc làm | Vì sao | Nếu BỎ |
|---|---|---|---|
| Chuẩn hoá Unicode **NFC** | gộp ký tự + dấu | tiếng Việt có dạng tổ hợp/dựng sẵn khác nhau cho cùng chữ | "à" (2 dạng) thành 2 token khác nhau |
| Lowercase | hạ chữ thường | giảm phân mảnh vốn từ | "HATE" ≠ "hate" |
| URL/@mention/sđt → token | thay bằng `<url> <user> <phone>` | các chuỗi này là nhiễu, vô hạn dạng | mỗi link là 1 token lạ → OOV |
| Emoticon/emoji → token cảm xúc | `:))`→`<troll_emo>`, `❤️`→`<emoji>` | giữ **sắc thái** (mỉa mai!) | mất tín hiệu cảm xúc / OOV |
| Rút gọn kéo dài | `quááá`→`quá` | gộp biến thể | `quá, quáá, quááá` là 3 token |
| Chuẩn hoá teencode | `ko`→`không`, `dc`→`được` | gộp về dạng chuẩn | vốn từ phân mảnh, OOV cao |

> **Vì sao OOV quan trọng (toán học).** Mỗi token tra một vector nhúng $E_i\in\mathbb R^d$. Token
> ngoài từ điển bị thay bằng **một** vector `<unk>` dùng chung → mất khả năng phân biệt. OOV cao
> ⇒ nhiều câu bị "xoá nhoà" thành chuỗi `<unk>` ⇒ mô hình mất thông tin đầu vào.
""")
code(r"""
from src.preprocessing import clean_text, simple_tokenize
demo = [
 "Em được làm fan cứng luôn rồi nè ❤️ reaction quá hay quá cute coi mấy giờ này =]]]",
 "ko hiểu sao bọn nó cứ chửi nhau =))))) vcl thật sự",
 "Đm thằng này ngu vl :)) đọc link https://abc.xyz đi @user gọi 0901234567",
]
for s in demo:
    print("RAW  :", s)
    print("RAW-tok (no-clean):", simple_tokenize(s))
    print("CLEAN:", clean_text(s)); print("-"*100)
""")

# ============================================================== 4. MODELS THEORY
md(r"""
## 4. Hai kiến trúc & nền tảng toán học

Cả hai mô hình **dùng chung**: lớp nhúng $E\in\mathbb R^{V\times d}$ ($d=200$), cùng cách gộp
chuỗi (**masked mean+max pooling**) và cùng đầu phân loại MLP. **Khác biệt duy nhất** là cách
trộn thông tin giữa các token.

### 4.1 BiLSTM (hồi quy)
LSTM xử lý **tuần tự**, tại mỗi bước $t$ cập nhật trạng thái nhớ $c_t$ và ẩn $h_t$ qua các *cổng*:
$$
\begin{aligned}
f_t&=\sigma(W_f[h_{t-1},x_t]+b_f) &\text{(quên)}\\
i_t&=\sigma(W_i[h_{t-1},x_t]+b_i) &\text{(nạp)}\\
\tilde c_t&=\tanh(W_c[h_{t-1},x_t]+b_c)\\
c_t&=f_t\odot c_{t-1}+i_t\odot \tilde c_t &\text{(ô nhớ)}\\
o_t&=\sigma(W_o[h_{t-1},x_t]+b_o),\quad h_t=o_t\odot\tanh(c_t)
\end{aligned}
$$
Cổng quên $f_t$ tạo "đường cao tốc" cho gradient ($\partial c_t/\partial c_{t-1}=f_t$) ⇒ giảm
**triệt tiêu gradient** so với RNN thường. **Bi**directional: chạy 2 chiều rồi nối
$h_t=[\overrightarrow{h_t};\overleftarrow{h_t}]$ để mỗi vị trí thấy cả ngữ cảnh trái–phải.
Độ phức tạp $O(L)$ nhưng **tuần tự** (khó song song theo thời gian).

### 4.2 Transformer Encoder (self-attention)
Mọi token "nhìn" mọi token **song song**. Lõi là *Scaled Dot-Product Attention*:
$$\text{Attention}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V,$$
với $Q=XW_Q,\,K=XW_K,\,V=XW_V$. Tích $QK^\top$ đo độ tương hợp mọi cặp token; chia $\sqrt{d_k}$
giữ phương sai ổn định để softmax không bão hoà. **Multi-head**: chạy $h$ "đầu" song song trên
các không gian con $d_k=d/h$ rồi nối lại — học nhiều kiểu quan hệ. Mỗi lớp encoder (Pre-LN):
$$x\leftarrow x+\text{MHA}(\text{LN}(x)),\qquad x\leftarrow x+\text{FFN}(\text{LN}(x)).$$
Độ phức tạp $O(L^2 d)$ (đắt hơn theo độ dài) nhưng **song song hoàn toàn** và đường đi thông tin
giữa 2 token bất kỳ chỉ dài $O(1)$ (LSTM là $O(L)$) ⇒ bắt phụ thuộc xa tốt hơn về lý thuyết.

### 4.3 Positional Encoding — vì sao **bắt buộc** với Transformer
Self-attention **bất biến hoán vị**: đổi thứ tự token, tập đầu ra chỉ bị hoán vị tương ứng ⇒
*"tôi ghét bạn"* và *"bạn ghét tôi"* có cùng biểu diễn nếu không thêm vị trí. Ta cộng mã vị trí
sin/cos:
$$PE_{(pos,2i)}=\sin\!\big(pos/10000^{2i/d}\big),\quad PE_{(pos,2i+1)}=\cos\!\big(pos/10000^{2i/d}\big).$$
(LSTM **không cần** vì thứ tự đã ngầm trong tính tuần tự.) Ablation ở mục 8 đo tác động thực tế.
""")
code(r"""
# Dựng mô hình (không huấn luyện) để in kiến trúc & đếm tham số
import torch
from src.config import Config
from src.models import build_model, count_params
cfg = Config()
for name in ("lstm", "transformer"):
    m = build_model(name, meta["vocab_size"], cfg)
    print(f"\n===== {name.upper()} =====  tham số: {count_params(m)/1e6:.2f}M")
    print(m)
""")

# ============================================================== 5. TRAINING METHODS
md(r"""
## 5. Phương pháp huấn luyện

### 5.1 Hàm mất mát cho dữ liệu mất cân bằng
Cross-entropy chuẩn coi mọi mẫu như nhau ⇒ bị lớp đa số lấn át. Ta dùng **weighted CE** với
trọng số nghịch tần suất (cân bằng):
$$w_c=\frac{N}{K\,n_c},\qquad \mathcal L=-\frac{1}{B}\sum_{j} w_{y_j}\log p_{y_j}(x_j).$$
(So sánh thêm **Focal Loss** $-(1-p_t)^\gamma\log p_t$ hạ trọng số mẫu dễ.) Mục 8 cho thấy
bỏ trọng số ⇒ recall lớp `HATE/OFFENSIVE` **sụp đổ**.

### 5.2 Lịch học: **warmup + decay tuyến tính**
LR tăng tuyến tính trong 10% bước đầu rồi giảm dần. Warmup tránh các bước đầu (gradient lớn,
thống kê Adam chưa ổn định) làm hỏng tham số — đặc biệt quan trọng với Transformer (LayerNorm +
residual nhạy với cú sốc ban đầu).

### 5.3 **Gradient clipping**
Chặn chuẩn gradient: nếu $\lVert g\rVert_2>\tau$ thì $g\leftarrow \tau\,g/\lVert g\rVert_2$.
Ngăn "vách gradient" gây bước nhảy huỷ hoại (bùng nổ gradient). Mục 6/8 vẽ chuẩn gradient để thấy.

### 5.4 **Early stopping** theo dev macro-F1
Dừng khi macro-F1 trên `dev` không cải thiện sau `patience` epoch, giữ lại checkpoint tốt nhất ⇒
chống overfit, tiết kiệm tính toán. Ta theo dõi macro-F1 (không phải loss) vì đó là mục tiêu thật.
""")

# ============================================================== 6. MONITOR
md(r"""
## 6. Giám sát quá trình huấn luyện

Bốn bảng theo dõi mỗi epoch: **loss** (train liền nét, dev nét đứt — chênh lệch lớn = overfit),
**accuracy**, **dev macro-F1** (chỉ số chọn mô hình), và **learning-rate** (thấy rõ warmup+decay).
""")
code(r"""
hist = logs["histories"]; steps = logs["step_logs"]
show(plots.plot_training_curves(hist, FIG_DIR / "05_training_curves.png"))
""")
md(r"""
**Khoảng cách train−dev** (đo overfit) và **chuẩn gradient theo bước** (đo ổn định — nhờ
gradient clipping, $\lVert g\rVert$ bị chặn, không bùng nổ):
""")
code(r"""
show(plots.plot_overfit_gap(hist, FIG_DIR / "06_overfit_gap.png"))
show(plots.plot_grad_norm(steps, FIG_DIR / "07_grad_norm.png"))
# bảng tóm tắt hội tụ
summ = pd.DataFrame({m: {
    "epoch tốt nhất": cost[m]["best_epoch"],
    "dev macroF1 (best)": round(cost[m]["best_dev_macro_f1"],4),
    "thời gian/epoch (s)": round(cost[m]["time_per_epoch"],1),
    "tham số (M)": round(cost[m]["params"]/1e6,2),
} for m in ("lstm","transformer")}).T
print(summ)
""")

# ============================================================== 7. EVAL
md(r"""
## 7. Đánh giá trên tập test

### 7.1 Ma trận nhầm lẫn (chuẩn hoá theo hàng = **recall** mỗi lớp)
Đường chéo càng đậm càng tốt. Chú ý lớp `OFFENSIVE`/`HATE` — nơi khó nhất.
""")
code(r"""
show(plots.plot_confusion(results, FIG_DIR / "08_confusion.png", normalize=True))
# báo cáo chi tiết
rows = []
for m in ("lstm","transformer"):
    r = results[m]
    rows.append({"model": m, "accuracy": round(r["acc"],4),
                 "macro-F1": round(r["macro_f1"],4), "weighted-F1": round(r["weighted_f1"],4),
                 **{f"F1_{LABEL_NAMES[i]}": round(r["per_class_f1"][i],3) for i in range(3)}})
display(pd.DataFrame(rows).set_index("model"))
""")
md(r"""### 7.2 F1 theo từng lớp & đường cong ROC / Precision–Recall (one-vs-rest)""")
code(r"""
show(plots.plot_per_class_f1(results, FIG_DIR / "09_per_class_f1.png"))
show(plots.plot_roc_pr(results, FIG_DIR / "10_roc_pr.png"))
""")
md(r"""
### 7.3 Bản đồ Self-Attention (khả diễn giải của Transformer)
Lấy một câu `HATE` thật, vẽ ma trận attention lớp cuối (trung bình các đầu). Hàng = token *truy
vấn*, cột = token được *chú ý*. Đây là điều LSTM **không** cung cấp trực tiếp.
""")
code(r"""
toks = attn_ex["tokens"]; A = np.array(attn_ex["attn"])
print("Câu (token):", " ".join(toks), "| dự đoán:", LABEL_NAMES[attn_ex["pred"]])
show(plots.plot_attention(toks, A, FIG_DIR / "13_attention.png",
     title=f"Self-attention (dự đoán = {LABEL_NAMES[attn_ex['pred']]})"))
""")

# ============================================================== 8. ABLATION
md(r"""
## 8. Ablation — "có" vs "không" dùng từng phương pháp

Mỗi thí nghiệm **chỉ đổi một yếu tố**, huấn luyện lại và đo trên test. Cột **đỏ** = bỏ phương pháp.

### 8.1 Trọng số lớp (class weighting) — quan trọng nhất với dữ liệu mất cân bằng
Bỏ trọng số: macro-F1 có thể *trông* không tệ (nhờ lớp đa số) nhưng **recall lớp HATE/OFFENSIVE
sụp đổ** — mô hình "phớt lờ" lớp thiểu số. Đây chính là lý do dùng weighted loss.
""")
code(r"""
def g(k, field):
    v = abl[k]; return v[field] if not isinstance(v.get(field), list) else v[field]
tab = []
for k,lab in [("lstm_with_cw","LSTM +CW"),("lstm_no_cw","LSTM −CW"),
              ("tf_with_cw","TF +CW"),("tf_no_cw","TF −CW")]:
    tab.append({"cấu hình":lab,"macro-F1":round(abl[k]["macro_f1"],4),
                "recall CLEAN":round(abl[k]["per_class_recall"][0],3),
                "recall OFFENSIVE":round(abl[k]["per_class_recall"][1],3),
                "recall HATE":round(abl[k]["per_class_recall"][2],3)})
display(pd.DataFrame(tab))
show(FIG_DIR / "14_ablation_classweight_f1.png")
show(FIG_DIR / "15_ablation_classweight_recall.png")
""")
md(r"""
### 8.2 Positional Encoding (Transformer)
Bỏ mã vị trí: self-attention trở nên **bất biến thứ tự** ⇒ mất thông tin trật tự từ.
""")
code(r'show(FIG_DIR / "16_ablation_posenc.png")')
md(r"""### 8.3 Tiền xử lý tiếng Việt (làm sạch)""")
code(r"""
print("OOV & vocab:", {k:abl["_meta"][k] for k in abl["_meta"]})
show(FIG_DIR / "17_ablation_clean.png")
""")
md(r"""### 8.4 LR Warmup & Gradient Clipping (ổn định huấn luyện)""")
code(r"""
show(FIG_DIR / "18_ablation_warmup.png")
show(FIG_DIR / "19_ablation_gradclip.png")
""")

# ============================================================== 9. CONCLUSION
md(r"""
## 9. So sánh tổng thể & kết luận
""")
code(r"""
show(plots.plot_model_cost(cost, FIG_DIR / "12_model_cost.png"))
final = pd.DataFrame({m:{"test accuracy":round(results[m]["acc"],4),
                         "test macro-F1":round(results[m]["macro_f1"],4),
                         "tham số (M)":round(cost[m]["params"]/1e6,2),
                         "thời gian/epoch (s)":round(cost[m]["time_per_epoch"],1)}
                      for m in ("lstm","transformer")}).T
display(final)
""")
md(r"""
**Kết luận chính** (số liệu cụ thể ở README & bảng trên):

1. **Kiến trúc.** Trên dữ liệu cỡ vừa (~24k câu, huấn luyện *từ đầu*), **BiLSTM** thường nhỉnh hơn
   Transformer-from-scratch về macro-F1 vì Transformer "đói dữ liệu" (thiếu thiên kiến quy nạp về
   trật tự, cần nhiều dữ liệu/pretraining hơn). Đổi lại Transformer **huấn luyện nhanh hơn/epoch**
   (song song) và cho **bản đồ attention** dễ diễn giải.
2. **Mất cân bằng → weighted loss + macro-F1** là quyết định quan trọng nhất: bỏ đi thì recall lớp
   thù ghét sụp đổ (mục 8.1).
3. **Tiền xử lý tiếng Việt** giảm mạnh OOV và cải thiện chất lượng (mục 2.3, 8.3).
4. **Warmup + gradient clipping** giúp huấn luyện ổn định, đặc biệt cho Transformer (mục 6, 8.4).
5. **Positional encoding** là thành phần lý thuyết bắt buộc của Transformer (mục 4.3, 8.2).

> Toàn bộ phương pháp, lý thuyết toán học và phân tích chi tiết xem trong `README.md`.
""")

nb = new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                              "name": "python3"},
               "language_info": {"name": "python"}}
out = "LSTM_vs_Transformer.ipynb"
nbf.write(nb, out)
print("WROTE", out, "with", len(cells), "cells")
