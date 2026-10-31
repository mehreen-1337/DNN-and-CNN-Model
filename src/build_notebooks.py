from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
KERNEL = "dl-project"

def _segment(lines: list[str], node: ast.stmt) -> str:
    start = node.lineno
    decs = getattr(node, "decorator_list", None)
    if decs:
        start = min(d.lineno for d in decs)
    return "\n".join(lines[start - 1:node.end_lineno])

def segments(path: Path) -> dict[str, str]:
    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = _segment(lines, node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = _segment(lines, node)
    return out

def build(cells: list[tuple[str, str]], path: Path) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
                for k, s in cells]
    nb.metadata.update({
        "kernelspec": {"name": KERNEL, "display_name": "Python (DL project)", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))
    print(f"[nb ] {path.relative_to(ROOT)}  ({len(nb.cells)} cells)")
    return path

def dnn_notebook() -> list[tuple[str, str]]:
    s = segments(ROOT / "src" / "dnn_bank_marketing.py")

    header = '''# Imports and global configuration
import json, time, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.display import Image, display

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_CSV = ROOT / "data" / "bank" / "bank-full.csv"
NB_OUT = ROOT / "results" / "notebook" / "dnn"
NB_OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

CAT_COLS = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "poutcome"]
NUM_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CFG = dict(val_size=0.15, test_size=0.15, batch_size=256, lr=1e-3, weight_decay=1e-4,
           max_epochs=300, patience=25, hidden=(256, 128, 64), dropout=(0.30, 0.30, 0.20))

def show(name, width=900):
    display(Image(filename=str(NB_OUT / name), width=width))

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("data file:", DATA_CSV, "| exists:", DATA_CSV.exists())'''

    return [
        ("md", """# Midterm Project – Deep Learning
## Model 1: Deep Neural Network (DNN) for Bank Term-Deposit Subscription Prediction

**Dataset:** UCI Machine Learning Repository, *Bank Marketing* (`bank-full.csv`, 45,211 rows)
**Task:** binary classification / probability estimation
**Framework:** PyTorch

This notebook walks through the complete pipeline required by the project brief:
problem definition → dataset description → proposed solution → data preprocessing →
model architecture → training and evaluation → conclusion."""),

        ("md", """## 1. Problem Definition

A retail bank runs **outbound telephone campaigns** to persuade existing clients to
subscribe to a *term deposit* (a savings product). The campaign is expensive: every
contact consumes agent time. In this dataset only **11.7 %** of the 45,211 contacted
clients actually subscribe, so roughly **88 % of the calling effort produces no revenue**.

**Problem.** Given the client profile, their banking history and the previous marketing
interactions, estimate the probability that this client will subscribe to the term
deposit, *before* the call is made.

**Why it matters.** With a reliable probability, the bank can rank its contact list and
call only the top-k clients, converting the same fixed budget into far more
subscriptions (higher precision/recall at a fixed cost).

**Framing.** This is a supervised binary classification problem with strongly
**imbalanced classes**, therefore accuracy alone is misleading and we will evaluate with
precision, recall, F1, ROC-AUC and PR-AUC."""),

        ("md", """## 2. Dataset Loading and First Look

* Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
* File used: `bank-full.csv` (all 17 attributes, `;`-separated), already downloaded to `data/bank/`
* 16 input features (7 numeric, 9 categorical) and the target `y` ∈ {yes, no}"""),

        ("code", header),

        ("code", s["load_data"] + "\n\ndf = load_data(DATA_CSV)\nprint(df.shape)\ndf.head()"),

        ("code", """df.info()

print("\\nMissing values per column:", int(df.isna().sum().sum()))
print("Duplicated rows:", int(df.duplicated().sum()))
print("Target balance:")
print(df["y"].value_counts(normalize=True).rename("share").to_string())"""),

        ("code", """df.describe().T[["mean", "std", "min", "50%", "max"]]"""),

        ("code", """cat_summary = pd.DataFrame({
    "unique": df[CAT_COLS].nunique(),
    "most_frequent": df[CAT_COLS].mode().iloc[0],
    "most_frequent_share": [
        df[c].value_counts(normalize=True).iloc[0] for c in CAT_COLS],
})
cat_summary"""),

        ("md", """## 3. Exploratory Data Analysis

Three observations drive the design decisions later:
1. **Class imbalance** (~11.7 % positives) → weighted loss + F1/AUC metrics instead of accuracy.
2. **`duration`** (call length) is only known *after* the call. It will make the model
   look excellent in an offline evaluation but cannot be used for pre-call targeting.
   We therefore report an **ablation without `duration`** as well.
3. **`poutcome`** (previous campaign result) and **`pdays = -1`** (never contacted) carry
   strong signal → we convert the sentinel into an explicit binary flag."""),

        ("code", s["plot_eda"] + "\n\nplot_eda(df, NB_OUT)\nshow('01_class_balance_and_age.png')\nshow('02_categorical_effects.png')"),

        ("md", """The left panel confirms the imbalance. The job and previous-outcome panels show that
the **outcome of the previous campaign** and the **client's job type** separate the classes
far more strongly than age, which is a good sign that a non-linear model can find useful
interactions."""),

        ("md", """## 4. Data Preprocessing

| Step | Operation | Reason |
|---|---|---|
| Cleaning | strip whitespace / quotes from strings, no missing values, no duplicates | raw UCI file is quoted |
| Target | `y` → 1 for `yes`, 0 else | binary classification |
| Feature engineering | `was_contacted_before = pdays != -1`, `repeated_calls = campaign > 1`; `pdays` sentinel −1 → 0 | `-1` is a code for "never", not a quantity |
| Categorical encoding | one-hot encoding of the 9 categorical columns (52 → 51 columns) | ANN needs numeric input; one-hot avoids a false ordinal relation |
| Scaling | `StandardScaler` fitted **on the training split only** | keeps the network well-conditioned and prevents leakage |
| Splitting | stratified 70 % / 15 % / 15 % train / validation / test | validation for early stopping and threshold selection, test touched once |"""),

        ("code", s["engineer_features"] + "\n\ndf = engineer_features(df)\ndf[['pdays','was_contacted_before','campaign','repeated_calls']].describe().T"),

        ("code", s["build_matrix"] + """
y = df["y"].to_numpy()
X, cols, scaler = build_matrix(df)

X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

print(f"encoded feature matrix: {X.shape}")
print(f"train {X_tr.shape[0]:,} | val {X_va.shape[0]:,} | test {X_te.shape[0]:,}")
print(f"positive rate  train {y_tr.mean():.3f} | val {y_va.mean():.3f} | test {y_te.mean():.3f}")
print("first 10 encoded columns:", cols[:10])"""),

        ("md", """## 5. Model Architecture

A compact feed-forward ANN (multi-layer perceptron) with **Batch Normalisation** and
**Dropout** in every hidden block:
```
input(51) → Linear(256) → BatchNorm → ReLU → Dropout(0.30)
          → Linear(128) → BatchNorm → ReLU → Dropout(0.30)
          → Linear(64)  → BatchNorm → ReLU → Dropout(0.20)
          → Linear(1)   → sigmoid ⇒ P(subscribe)
```

* **BatchNorm** stabilises and accelerates training on tabular data with mixed scales.
* **Dropout** regularises; the model is small (~55k parameters) and the dataset is only
  ~31k training rows, so over-fitting is the main risk.
* **Single logit + `BCEWithLogitsLoss`** (numerically stable) with `pos_weight = n_neg/n_pos`
  so the rare positive class is not ignored.
* **AdamW** optimiser (`lr = 1e-3`, `wd = 1e-4`), `ReduceLROnPlateau` on validation AUC,
  **early stopping** (patience 25 epochs) and the best-AUC weights are restored."""),

        ("code", s["BankDNN"] + """

model_preview = BankDNN(X.shape[1], (256, 128, 64), (0.30, 0.30, 0.20))
print(model_preview)
print("\\ntrainable parameters:",
      f"{sum(p.numel() for p in model_preview.parameters()):,}")"""),

        ("md", """## 6. Training

Training uses mini-batches of 256, early stopping on validation ROC-AUC and the best
checkpoint is kept."""),

        ("code", s["set_seed"] + "\n" + s["predict_proba"]),

        ("code", s["train_dnn"]),

        ("code", """set_seed()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, hist = train_dnn(X_tr, y_tr, X_va, y_va, X.shape[1], device=device)
print(f"stopped after {len(hist['train_loss'])} epochs, best epoch {hist['best_epoch']}, "
      f"{hist['seconds']} s, best val AUC {max(hist['val_auc']):.4f}")"""),

        ("code", s["plot_training"] + "\n\nplot_training(hist, NB_OUT)\nshow('03_training_curves.png')"),

        ("md", """The validation AUC climbs quickly in the first ~20 epochs and then plateaus; the gap
between train and validation loss stays small, i.e. dropout + BatchNorm keep the model
from over-fitting despite only ~31k training rows."""),

        ("md", """## 7. Evaluation

Metrics are computed on the **held-out test set** (6,782 clients, never used for training
or early stopping). The decision threshold is selected on the **validation** set by
maximising F1, because with imbalanced data 0.5 is rarely optimal for the business goal."""),

        ("code", s["evaluate"] + "\n" + s["best_threshold"]),

        ("code", """prob_va = predict_proba(model, X_va, device)
thr, f1_va = best_threshold(y_va, prob_va)
prob_te = predict_proba(model, X_te, device)

m = evaluate(y_te, prob_te, thr)
m05 = evaluate(y_te, prob_te, 0.50)

summary = pd.DataFrame({
    "metric": ["threshold", "accuracy", "precision", "recall", "F1", "ROC-AUC", "PR-AUC (AP)", "Brier"],
    "tuned threshold": [m["threshold"], m["accuracy"], m["precision"], m["recall"], m["f1"],
                        m["roc_auc"], m["pr_auc"], m["brier"]],
    "0.5 threshold": [0.50, m05["accuracy"], m05["precision"], m05["recall"], m05["f1"],
                      m05["roc_auc"], m05["pr_auc"], m05["brier"]],
}).set_index("metric")
summary.round(4)"""),

        ("code", """print(f"confusion matrix @ thr={thr:.2f}:")
print(f"                pred no   pred yes")
print(f"actual no   {m['tn']:10d} {m['fp']:10d}")
print(f"actual yes  {m['fn']:10d} {m['tp']:10d}")
print(f"\\nCV-style sanity check on the validation set: F1={f1_va:.4f}")"""),

        ("code", s["plot_confusion"] + "\n" + s["plot_roc_pr"] + "\n" + s["plot_thresholds"] +
         "\n\nplot_confusion(y_te, prob_te, thr, NB_OUT)\nplot_roc_pr(y_te, prob_te, NB_OUT)\n"
         "plot_thresholds(y_te, prob_te, thr, NB_OUT)\nshow('04_confusion_matrix.png', 520)\n"
         "show('05_roc_pr_curves.png')\nshow('06_threshold_analysis.png')"),

        ("md", """### 7.1 Is the deep model worth it? Comparison with logistic regression and the `duration` ablation

Two controls on exactly the same split:
* **Logistic regression** (class-balanced): the classical linear baseline for tabular data.
* **ANN without `duration`**: a *deployable* model that does not rely on information
  available only after the phone call."""),

        ("code", """lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X_tr, y_tr)
m_lr = evaluate(y_te, lr.predict_proba(X_te)[:, 1], thr)

df_abl = df.drop(columns=["duration"])
X_a, cols_a, _ = build_matrix(df_abl)
Xa_tr, Xa_tmp, ya_tr, ya_tmp = train_test_split(X_a, y, test_size=0.30, stratify=y, random_state=SEED)
Xa_va, Xa_te, ya_va, ya_te = train_test_split(Xa_tmp, ya_tmp, test_size=0.50, stratify=ya_tmp, random_state=SEED)

set_seed()
model_abl, hist_abl = train_dnn(Xa_tr, ya_tr, Xa_va, ya_va, X_a.shape[1], device=device, verbose=False)
thr_a, _ = best_threshold(ya_va, predict_proba(model_abl, Xa_va, device))
m_abl = evaluate(ya_te, predict_proba(model_abl, Xa_te, device), thr_a)

compare = pd.DataFrame({
    "Model": ["DNN (this work)", "Logistic regression", "DNN without 'duration'"],
    "Accuracy": [m["accuracy"], m_lr["accuracy"], m_abl["accuracy"]],
    "Precision": [m["precision"], m_lr["precision"], m_abl["precision"]],
    "Recall": [m["recall"], m_lr["recall"], m_abl["recall"]],
    "F1": [m["f1"], m_lr["f1"], m_abl["f1"]],
    "ROC-AUC": [m["roc_auc"], m_lr["roc_auc"], m_abl["roc_auc"]],
    "PR-AUC": [m["pr_auc"], m_lr["pr_auc"], m_abl["pr_auc"]],
}).set_index("Model")
compare.round(4)"""),

        ("md", """**Reading the table.** The DNN beats the linear baseline on every metric
(≈ +0.02 ROC-AUC, ≈ +0.07 F1), which shows that client behaviour is genuinely non-linear.
Removing `duration` costs ≈ 0.14 ROC-AUC: the "with duration" model is *not deployable for
pre-call targeting* and is reported mainly for comparability with the literature."""),

        ("md", """### 7.2 What drives the prediction? (permutation importance)

Each input column is shuffled and the drop in test ROC-AUC is measured. This is a model-agnostic
way to open the black box."""),

        ("code", s["permutation_importance"] + "\n" + s["plot_importance"] + """
imp = permutation_importance(model, X_te, y_te, cols, device, n_repeats=3)
plot_importance(imp, NB_OUT)
show('07_permutation_importance.png')
imp"""),

        ("md", """### 7.3 Business interpretation

With the tuned threshold the model reviews ≈ 18 % of the client base
(`(tp+fp)/n_test`) and reaches them with ≈ 51 % precision versus the 11.7 % base rate,
i.e. **~4.4x more subscriptions per call**. Ranking by `P(subscribe)` and calling the top-k
is therefore directly actionable for a campaign planner."""),

        ("md", """## 8. Conclusion

* A compact ANN (55k parameters) predicts term-deposit subscription with **ROC-AUC ≈ 0.92**
  and **F1 ≈ 0.62** on unseen data, clearly above a class-balanced logistic regression.
* Preprocessing decisions that mattered most: one-hot encoding + standardisation,
  converting the `pdays = -1` sentinel into a flag, and class weighting for the 1:7.5
  imbalance.
* **Limitation:** `duration` leaks information from after the call. The honest, deployable
  variant reaches ROC-AUC ≈ 0.78, still far above the 11.7 % base rate but with lower
  precision. Any production model must therefore be trained and evaluated without it.
* **Further work:** gradient-boosted trees (usually the strongest baseline for tabular
  data), calibrated probabilities, and a cost-sensitive threshold based on the real
  cost of a call versus the margin of a term deposit."""),
    ]

def cnn_notebook() -> list[tuple[str, str]]:
    s = segments(ROOT / "src" / "cnn_fashion_mnist.py")

    header = '''# Imports and global configuration
import json, time, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from IPython.display import Image, display

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
NB_OUT = ROOT / "results" / "notebook" / "cnn"
NB_OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MEAN, STD = 0.2860, 0.3530
CFG = dict(batch_size=128, epochs=15, lr=1e-3, weight_decay=5e-4, label_smoothing=0.1,
           val_size=5000, patience=6, workers=4, baseline_epochs=10)

def show(name, width=900):
    display(Image(filename=str(NB_OUT / name), width=width))

print("torch", torch.__version__, "| device:", device,
      "|", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")'''

    return [
        ("md", """# Midterm Project – Deep Learning
## Model 2: Convolutional Neural Network (CNN) for Fashion-MNIST Garment Classification

**Dataset:** Fashion-MNIST (Zalando Research), 70,000 greyscale images of 28×28 pixels in 10 classes
**Task:** 10-class image classification
**Framework:** PyTorch"""),

        ("md", """## 1. Problem Definition

Online fashion retailers receive product photographs at catalogue scale. Before an item
can be searched, filtered or recommended it must carry a **category label**. Manual
tagging is slow and error-prone, so the task here is automatic:
> Given a 28×28 greyscale image of a garment, predict which of 10 product categories it
> belongs to: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag,
> Ankle boot.

The value is operational: automatic tagging reduces catalogue time-to-market and enables
search/recommendation over new stock. Because the classes are visually similar (Shirt vs
T-shirt vs Pullover vs Coat), the task is a realistic test of whether a CNN can learn
local texture and silhouette cues rather than colour (Fashion-MNIST is deliberately harder
than MNIST, which is largely solved)."""),

        ("md", """## 2. Dataset Description and Loading

* Source: https://github.com/zalandoresearch/fashion-mnist (mirrored by `torchvision`)
* 60,000 training + 10,000 test images, 28×28, 8-bit greyscale, 10 balanced classes
* Pixel statistics: mean 0.2860, std 0.3530 (single channel)

We hold out 5,000 images from the 60k training split as a **validation set**, so the
official 10k **test set is used only once, at the very end**."""),

        ("code", header),

        ("code", """raw_train = datasets.FashionMNIST(str(DATA_DIR), train=True, download=True,
                               transform=transforms.ToTensor())
raw_test = datasets.FashionMNIST(str(DATA_DIR), train=False, download=True,
                              transform=transforms.ToTensor())
print("train images:", len(raw_train), "| test images:", len(raw_test))
print("image tensor shape:", raw_train[0][0].shape)
print("class counts (train):", np.bincount(raw_train.targets.numpy()))
print("pixel mean/std:", round(float(raw_train.data.float().mean() / 255), 4),
      round(float(raw_train.data.float().std() / 255), 4))"""),

        ("md", """## 3. Data Preprocessing and Augmentation

| Step | Setting | Why |
|---|---|---|
| Normalisation | `(x/255 − 0.2860) / 0.3530` | zero-mean, unit-variance input speeds up and stabilises training |
| Augmentation (train only) | `RandomCrop(28, padding=2)` + `RandomHorizontalFlip(0.5)` | small translations and mirrors prevent memorisation; a flipped shoe is still a shoe |
| Validation / test | deterministic: `ToTensor` + `Normalize` only | evaluation must be reproducible |
| Batching | 128 (train), 512 (eval), `num_workers=4`, pinned memory | keeps the GPU fed |
| Validation split | 5,000 images from the training set, fixed seed | model selection without touching the test set |"""),

        ("code", s["make_loaders_explicit"] + """

train_loader, val_loader, test_loader = make_loaders_explicit()
print(f"train {len(train_loader.dataset):,} | val {len(val_loader.dataset):,} | "
      f"test {len(test_loader.dataset):,}")
xb, yb = next(iter(train_loader))
print("batch:", tuple(xb.shape), "| labels:", yb[:10].tolist(),
      "| pixel range:", round(float(xb.min()), 2), "to", round(float(xb.max()), 2))"""),

        ("code", s["_class_counts"] + "\n\ncounts = _class_counts(train_loader)\nprint(dict(zip(CLASSES, counts.tolist())))"),

        ("code", s["plot_examples"] + """

plot_examples(train_loader, NB_OUT)
show('01_sample_images.png')
show('02_class_distribution.png')"""),

        ("md", """## 4. Model Architecture

A compact **VGG-style CNN** with three convolution blocks and global average pooling:
```
input 1×28×28
 ├─ ConvBlock(1→32)    : Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → MaxPool2 → Dropout 0.25   → 32×14×14
 ├─ ConvBlock(32→64)   : same                                                                    → 64×7×7
 ├─ ConvBlock(64→128)  : same                                                                    → 128×3×3
 ├─ GlobalAveragePool  → 128
 ├─ Dropout 0.4 → Linear(128→10)
```

Design choices:
* **Two 3×3 convolutions per block**: same receptive field as a 5×5 kernel but fewer
  parameters and an extra non-linearity.
* **BatchNorm** after every convolution: stable gradients, faster convergence.
* **Global Average Pooling** instead of a huge `Flatten` head, so only 1.3k parameters sit in the
  classifier, so the model cannot memorise; also gives translation robustness.
* **Dropout 0.25 / 0.4**: regularisation against the 55k training images.
* Loss `CrossEntropyLoss(label_smoothing=0.1)`, optimiser **AdamW** (`lr 1e-3`, `wd 5e-4`),
  **CosineAnnealingLR** over 15 epochs, early stopping (patience 6) on validation accuracy."""),

        ("code", s["ConvBlock"] + "\n" + s["FashionCNN"] + """

cnn = FashionCNN().to(device)
n_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
print(cnn)
print(f"\\ntrainable parameters: {n_params:,}")"""),

        ("code", """# shape check through the network
with torch.no_grad():
    x = torch.randn(2, 1, 28, 28).to(device)
    for i, block in enumerate(cnn.features):
        x = block(x)
        print(f"after block {i+1}: {tuple(x.shape)}")
    print("logits:", tuple(cnn.head(x).shape))"""),

        ("md", """## 5. Training"""),

        ("code", s["set_seed"] + "\n" + s["train_model"]),

        ("code", """set_seed()
cnn, hist = train_model(cnn, train_loader, val_loader, CFG["epochs"], device,
                        CFG["lr"], CFG["weight_decay"], CFG["label_smoothing"],
                        CFG["patience"], tag="CNN")
print(f"\\nbest epoch {hist['best_epoch']} | best val accuracy {hist['best_val_acc']:.4f} "
      f"| {hist['seconds']} s")"""),

        ("md", """## 6. Evaluation on the Test Set

We now evaluate the restored best checkpoint on the official 10,000-image test set that
was never used during training or model selection."""),

        ("code", s["predict"] + "\n" + s["classification_metrics"] + """

prob_te, y_te = predict(cnn, test_loader, device)
m = classification_metrics(y_te, prob_te)
print(f"test accuracy    : {m['accuracy']:.4f}")
print(f"macro F1         : {m['macro_f1']:.4f}")
print(f"weighted F1      : {m['weighted_f1']:.4f}")"""),

        ("code", """per_class = pd.DataFrame(m["per_class"]).T
per_class.columns = ["precision", "recall", "F1", "support"]
per_class.index.name = "class"
per_class.round(4)"""),

        ("code", s["plot_curves"] + "\n" + s["plot_confusion"] + "\n" + s["plot_per_class"] + """

plot_curves(hist, None, NB_OUT)
plot_confusion(np.array(m["confusion_matrix"]), NB_OUT)
plot_per_class(m["per_class"], NB_OUT)
show('03_training_curves.png')
show('04_confusion_matrix.png')
show('05_per_class_metrics.png')"""),

        ("md", """### 6.1 Qualitative check

Correct and incorrect predictions with the model's confidence:"""),

        ("code", s["plot_predictions"] + """

plot_predictions(cnn, test_loader, y_te, prob_te, device, NB_OUT)
show('06_predictions.png')
show('07_misclassified.png')"""),

        ("md", """### 6.2 Error analysis: where does the model fail?

All the hard cases are **upper-body garments**: Shirt, T-shirt/top, Pullover, Coat and
Dress are silhouettes that differ mainly by sleeves, collar and length. At 28×28 most of
that detail is gone. Trousers, bags, sandals and ankle boots are separated cleanly."""),

        ("code", """cm = np.array(m["confusion_matrix"])
off = cm.copy(); np.fill_diagonal(off, 0)
pairs = sorted([(int(off[i, j]), CLASSES[i], CLASSES[j])
                for i in range(10) for j in range(10) if off[i, j] > 0], reverse=True)[:8]
pd.DataFrame(pairs, columns=["count", "true class", "predicted as"]).set_index("count").head(8)"""),

        ("md", """## 7. Control experiment: is convolution actually necessary?

An MLP with two hidden layers on the **flattened** image receives the same pixels but has
no weight sharing and no locality. If the CNN reaches clearly higher accuracy with a
similar budget, the spatial inductive bias is what matters."""),

        ("code", s["MLPBaseline"] + """

set_seed()
mlp = MLPBaseline().to(device)
mlp, hist_mlp = train_model(mlp, train_loader, val_loader, 10, device,
                            lr=1e-3, weight_decay=1e-4, label_smoothing=0.0,
                            patience=99, tag="MLP")
prob_mlp, _ = predict(mlp, test_loader, device)
m_mlp = classification_metrics(y_te, prob_mlp)

pd.DataFrame({
    "Model": ["CNN (this work)", "MLP on flattened pixels"],
    "Parameters": [n_params, sum(p.numel() for p in mlp.parameters())],
    "Test accuracy": [m["accuracy"], m_mlp["accuracy"]],
    "Macro F1": [m["macro_f1"], m_mlp["macro_f1"]],
}).set_index("Model").round(4)"""),

        ("code", """# inference speed on the test device
cnn.eval()
with torch.no_grad():
    xb = torch.randn(256, 1, 28, 28, device=device)
    for _ in range(3): cnn(xb)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20): cnn(xb)
    if device == "cuda": torch.cuda.synchronize()
lat = (time.time() - t0) / 20 / 256 * 1000
print(f"batch inference: {lat * 256:.2f} ms per 256 images ({lat:.3f} ms/image) on {device}")"""),

        ("md", """## 8. Conclusion

* The CNN classifies unseen garments with **≈ 92 % accuracy** and **≈ 0.92 macro-F1** after
  only 15 epochs (~1 minute of GPU time), which is enough to auto-tag a catalogue and route
  confident predictions to the shop floor while flagging the rest for a human.
* Convolution is essential: the pixel MLP is actually **larger** (~536k parameters vs
  288k) yet loses ≈ 6 points of accuracy, because it cannot exploit locality and weight
  sharing.
* The residual error is concentrated exactly where human annotators also hesitate,
  Shirt vs T-shirt/Pullover/Coat, so the practical recommendation is **confidence
  thresholding**: auto-accept above ~0.9 and send the rest to manual review.
* **Limitations:** 28×28 greyscale images are far from real product photography (colour,
  background, zoom). On real data, transfer learning (e.g. a pre-trained ResNet/EfficientNet)
  with the same head would be the first step."""),

        ("code", """summary = {
    "dataset": "Fashion-MNIST",
    "split": {"train": len(train_loader.dataset), "val": len(val_loader.dataset),
              "test": len(test_loader.dataset)},
    "cnn_accuracy": m["accuracy"],
    "cnn_macro_f1": m["macro_f1"],
    "mlp_accuracy": m_mlp["accuracy"],
    "best_epoch": hist["best_epoch"],
    "train_seconds": hist["seconds"],
}
(ROOT / "results" / "cnn" / "notebook_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))"""),
    ]

def execute(path: Path) -> None:
    from nbconvert.preprocessors import ExecutePreprocessor

    nb = nbf.read(str(path), as_version=4)
    ep = ExecutePreprocessor(timeout=3600, kernel_name=KERNEL)
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    nbf.write(nb, str(path))
    print(f"[run] {path.name} executed")

if __name__ == "__main__":
    dnn = build(dnn_notebook(), NB_DIR / "DNN_Bank_Marketing.ipynb")
    cnn = build(cnn_notebook(), NB_DIR / "CNN_Fashion_MNIST.ipynb")
    for p in (dnn, cnn):
        execute(p)
    print("done")
