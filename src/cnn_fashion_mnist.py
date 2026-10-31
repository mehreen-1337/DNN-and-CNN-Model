from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 140

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = ROOT / "results" / "cnn"
SEED = 42
CFG = dict(
    batch_size=128,
    epochs=15,
    lr=1e-3,
    weight_decay=5e-4,
    label_smoothing=0.1,
    val_size=5000,
    patience=6,
    workers=4,
    baseline_epochs=10,
)
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MEAN, STD = 0.2860, 0.3530

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_loaders_explicit(cfg=CFG):
    train_tf = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])

    full_train_aug = datasets.FashionMNIST(str(DATA_DIR), train=True, download=True, transform=train_tf)
    full_train_raw = datasets.FashionMNIST(str(DATA_DIR), train=True, download=True, transform=eval_tf)
    test_set = datasets.FashionMNIST(str(DATA_DIR), train=False, download=True, transform=eval_tf)

    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(full_train_aug), generator=g)
    val_idx = idx[: cfg["val_size"]]
    tr_idx = idx[cfg["val_size"]:]

    train_set = torch.utils.data.Subset(full_train_aug, tr_idx.tolist())
    val_set = torch.utils.data.Subset(full_train_raw, val_idx.tolist())
    common = dict(num_workers=cfg["workers"], pin_memory=torch.cuda.is_available())
    return (DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, **common),
            DataLoader(val_set, batch_size=512, shuffle=False, **common),
            DataLoader(test_set, batch_size=512, shuffle=False, **common))

class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, p_drop: float = 0.25):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)), inplace=True)
        x = F.relu(self.b2(self.c2(x)), inplace=True)
        return self.drop(self.pool(x))

class FashionCNN(nn.Module):
    def __init__(self, n_classes: int = 10, widths=(32, 64, 128), p_drop=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, widths[0]),
            ConvBlock(widths[0], widths[1]),
            ConvBlock(widths[1], widths[2]),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(p_drop), nn.Linear(widths[2], n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

class MLPBaseline(nn.Module):
    def __init__(self, n_classes: int = 10, h=(512, 256), p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, h[0]), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h[0], h[1]), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h[1], n_classes),
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, val_loader, epochs, device, lr=1e-3,
                weight_decay=5e-4, label_smoothing=0.1, patience=6, tag="model"):
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_state, bad, t0 = -1.0, None, 0, time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tot, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(xb)
        tr_loss, tr_acc = tot / n, correct / n

        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += F.cross_entropy(out, yb).item() * len(xb)
                vc += (out.argmax(1) == yb).sum().item()
                vn += len(xb)
        val_loss, val_acc = vl / vn, vc / vn
        sched.step()

        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc),
                     ("val_loss", val_loss), ("val_acc", val_acc), ("lr", opt.param_groups[0]["lr"])]:
            hist[k].append(float(v))

        star = ""
        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = "  *"
        else:
            bad += 1
        print(f"[{tag}] epoch {ep:2d}/{epochs} | train {tr_loss:.4f}/{tr_acc:.4f} "
              f"| val {val_loss:.4f}/{val_acc:.4f}{star}")
        if bad >= patience:
            print(f"[{tag}] early stopping at epoch {ep} (best val acc {best_acc:.4f})")
            break

    model.load_state_dict(best_state)
    hist["best_val_acc"] = float(best_acc)
    hist["best_epoch"] = int(np.argmax(hist["val_acc"]) + 1)
    hist["seconds"] = round(time.time() - t0, 2)
    return model, hist

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels, imgs = [], [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        probs.append(F.softmax(out, dim=1).cpu())
        labels.append(yb)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()

def classification_metrics(y_true, prob) -> dict:
    pred = prob.argmax(1)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, pred):
        cm[t, p] += 1
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        per_class[name] = dict(precision=float(prec), recall=float(rec), f1=float(f1),
                               support=int(cm[i, :].sum()))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    support = sum(v["support"] for v in per_class.values())
    weighted_f1 = float(sum(v["f1"] * v["support"] for v in per_class.values()) / support)
    return dict(accuracy=float(np.trace(cm) / cm.sum()), macro_f1=macro_f1,
                weighted_f1=weighted_f1, per_class=per_class,
                confusion_matrix=cm.tolist())

def plot_examples(train_loader, out: Path) -> None:
    images, labels = next(iter(train_loader))
    fig, axes = plt.subplots(4, 8, figsize=(13, 7))
    for i, ax in enumerate(axes.ravel()):
        img = images[i] * STD + MEAN
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(CLASSES[labels[i]], fontsize=9)
        ax.axis("off")
    fig.suptitle("Fashion-MNIST samples (training split, augmented)")
    fig.tight_layout()
    fig.savefig(out / "01_sample_images.png", bbox_inches="tight")
    plt.close(fig)

    counts = _class_counts(train_loader)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(CLASSES, counts, color="#4C72B0")
    ax.bar_label(bars, fmt="%d", fontsize=8)
    ax.set_title("Class distribution (training split)")
    ax.set_ylabel("images"); plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out / "02_class_distribution.png", bbox_inches="tight")
    plt.close(fig)

def _class_counts(loader) -> np.ndarray:
    ds = loader.dataset
    base = ds.dataset if hasattr(ds, "dataset") else ds
    idx = ds.indices if hasattr(ds, "indices") else list(range(len(ds)))
    targets = np.array([base[i][1] for i in idx]) if hasattr(ds, "indices") else \
        np.array([t for _, t in ds])
    return np.bincount(targets, minlength=10)

def plot_curves(hist: dict, hist_mlp: dict | None, out: Path) -> None:
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(ep, hist["train_loss"], label="train")
    ax[0].plot(ep, hist["val_loss"], label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy loss"); ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="train")
    ax[1].plot(ep, hist["val_acc"], label="validation")
    if hist_mlp:
        e2 = np.arange(1, len(hist_mlp["val_acc"]) + 1)
        ax[1].plot(e2, hist_mlp["val_acc"], "--", color="grey", label="MLP baseline (val)")
    ax[1].axvline(hist["best_epoch"], ls=":", c="grey", label=f"best epoch ({hist['best_epoch']})")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "03_training_curves.png", bbox_inches="tight")
    plt.close(fig)

def plot_confusion(cm: np.ndarray, out: Path) -> None:
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(norm, annot=cm, fmt="d", cmap="Blues", cbar=True, ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={"size": 8})
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (test set) - cell text = count, colour = row-normalised")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(out / "04_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

def plot_per_class(per_class: dict, out: Path) -> None:
    names = list(per_class.keys())
    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for k, (metric, color) in enumerate(zip(["precision", "recall", "f1"],
                                            ["#4C72B0", "#DD8452", "#55A868"])):
        ax.bar(x + (k - 1) * w, [per_class[n][metric] for n in names], w, label=metric, color=color)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("score")
    ax.set_title("Per-class precision / recall / F1 on the test set")
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "05_per_class_metrics.png", bbox_inches="tight")
    plt.close(fig)

def plot_predictions(model, loader, y_true, prob, device, out: Path) -> None:
    model.eval()
    images, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            images.append(xb); labels.append(yb)
            if sum(len(i) for i in images) >= 512:
                break
    xb = torch.cat(images)[:32]
    yb = torch.cat(labels)[:32]
    with torch.no_grad():
        p = F.softmax(model(xb.to(device)), 1).cpu().numpy()
    fig, axes = plt.subplots(4, 8, figsize=(13, 7.2))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow((xb[i] * STD + MEAN).squeeze(), cmap="gray")
        pred = CLASSES[p[i].argmax()]
        ok = p[i].argmax() == yb[i].item()
        ax.set_title(f"p: {pred}\n({p[i].max():.2f})", fontsize=7,
                     color="green" if ok else "crimson")
        ax.axis("off")
    fig.suptitle("Model predictions on test images (green = correct, red = wrong)")
    fig.tight_layout()
    fig.savefig(out / "06_predictions.png", bbox_inches="tight")
    plt.close(fig)

    wrong = np.where(y_true != prob.argmax(1))[0][:16]
    ds = loader.dataset
    fig, axes = plt.subplots(4, 4, figsize=(9.5, 9.8))
    for ax, j in zip(axes.ravel(), wrong):
        img, lab = ds[int(j)]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"true: {CLASSES[lab]}\npred: {CLASSES[prob[j].argmax()]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Misclassified test images")
    fig.tight_layout()
    fig.savefig(out / "07_misclassified.png", bbox_inches="tight")
    plt.close(fig)

def run(cfg=CFG, out: Path = OUT, device: str | None = None) -> dict:
    set_seed()
    out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, val_loader, test_loader = make_loaders_explicit(cfg)
    n_tr = len(train_loader.dataset); n_va = len(val_loader.dataset); n_te = len(test_loader.dataset)
    print(f"split: train {n_tr:,} | val {n_va:,} | test {n_te:,}")
    plot_examples(train_loader, out)

    cnn = FashionCNN().to(device)
    n_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print(f"CNN parameters: {n_params:,}")
    cnn, hist = train_model(cnn, train_loader, val_loader, cfg["epochs"], device,
                            cfg["lr"], cfg["weight_decay"], cfg["label_smoothing"],
                            cfg["patience"], tag="CNN")

    prob_te, y_te = predict(cnn, test_loader, device)
    m_cnn = classification_metrics(y_te, prob_te)
    print(f"CNN test accuracy: {m_cnn['accuracy']:.4f} | macro-F1: {m_cnn['macro_f1']:.4f}")

    cnn.eval()
    xb = torch.randn(256, 1, 28, 28, device=device)
    with torch.no_grad():
        for _ in range(3):
            cnn(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            cnn(xb)
        if device == "cuda":
            torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / 20 / 256 * 1000

    print("Training MLP baseline ...")
    mlp = MLPBaseline().to(device)
    n_params_mlp = sum(p.numel() for p in mlp.parameters())
    mlp, hist_mlp = train_model(mlp, train_loader, val_loader, cfg["baseline_epochs"], device,
                                cfg["lr"], 1e-4, 0.0, 99, tag="MLP")
    prob_mlp, _ = predict(mlp, test_loader, device)
    m_mlp = classification_metrics(y_te, prob_mlp)
    print(f"MLP test accuracy: {m_mlp['accuracy']:.4f} | macro-F1: {m_mlp['macro_f1']:.4f}")

    plot_curves(hist, hist_mlp, out)
    cm = np.array(m_cnn["confusion_matrix"])
    plot_confusion(cm, out)
    plot_per_class(m_cnn["per_class"], out)
    plot_predictions(cnn, test_loader, y_te, prob_te, device, out)

    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = sorted([(int(off[i, j]), CLASSES[i], CLASSES[j])
                    for i in range(10) for j in range(10) if off[i, j] > 0], reverse=True)[:8]
    print("Top confusions:", pairs)

    results = dict(dataset="Fashion-MNIST (Zalando Research, 70k 28x28 grey images, 10 classes)",
                   split=dict(train=n_tr, val=n_va, test=n_te), device=str(device),
                   config=cfg, cnn_params=int(n_params), mlp_params=int(n_params_mlp),
                   cnn=m_cnn, mlp_baseline=m_mlp, top_confusions=pairs,
                   latency_ms_per_image=float(latency_ms))
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "history.json").write_text(json.dumps({"cnn": hist, "mlp": hist_mlp}, indent=2))
    torch.save({"state_dict": cnn.state_dict(), "classes": CLASSES}, out / "model_cnn.pt")
    print(f"Saved artifacts to {out}")
    return results

if __name__ == "__main__":
    res = run()
    print(json.dumps({"cnn_accuracy": res["cnn"]["accuracy"],
                      "cnn_macro_f1": res["cnn"]["macro_f1"],
                      "mlp_accuracy": res["mlp_baseline"]["accuracy"]}, indent=2))
