from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

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

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 140

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "bank" / "bank-full.csv"
OUT = ROOT / "results" / "dnn"
SEED = 42
CFG = dict(
    val_size=0.15,
    test_size=0.15,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=300,
    patience=25,
    hidden=(256, 128, 64),
    dropout=(0.30, 0.30, 0.20),
)

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CAT_COLS = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "poutcome"]
NUM_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

def load_data(path: Path = DATA_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip().strip('"') for c in df.columns]
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip().str.strip('"')
    df["y"] = (df["y"] == "yes").astype(int)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["was_contacted_before"] = (df["pdays"] != -1).astype(int)
    df["pdays"] = df["pdays"].where(df["pdays"] != -1, 0)
    df["repeated_calls"] = (df["campaign"] > 1).astype(int)
    return df

def build_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None,
                 scaler: StandardScaler | None = None):
    num = [c for c in NUM_COLS if c in df.columns]
    cat = [c for c in CAT_COLS if c in df.columns]
    dummies = pd.get_dummies(df[cat].astype(str), prefix=cat, dtype=np.float32)
    block = pd.concat([df[num].astype(np.float32), dummies], axis=1)

    if feature_cols is not None:
        block = block.reindex(columns=feature_cols, fill_value=0.0)

    X = block.to_numpy(dtype=np.float32)
    if scaler is None:
        scaler = StandardScaler().fit(X)
    X = scaler.transform(X).astype(np.float32)
    return X, list(block.columns), scaler

class BankDNN(nn.Module):
    def __init__(self, n_features: int, hidden=(256, 128, 64), dropout=(0.3, 0.3, 0.2)):
        super().__init__()
        layers, prev = [], n_features
        for h, p in zip(hidden, dropout):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), 4096):
        xb = torch.from_numpy(X[i:i + 4096]).to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)

def train_dnn(Xtr, ytr, Xva, yva, n_features, cfg=CFG, device="cpu", verbose=True):
    model = BankDNN(n_features, cfg["hidden"], cfg["dropout"]).to(device)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=5, min_lr=1e-5)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    n = len(Xtr_t)
    hist = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}
    best_auc, best_state, bad, t0 = -1.0, None, 0, time.time()

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot, cnt = 0.0, 0
        for i in range(0, n, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        train_loss = tot / cnt

        model.eval()
        with torch.no_grad():
            vl, vc = 0.0, 0
            for i in range(0, len(Xva), 4096):
                xb = torch.from_numpy(Xva[i:i + 4096]).to(device)
                yb = torch.from_numpy(yva[i:i + 4096].astype(np.float32)).to(device)
                vl += nn.functional.binary_cross_entropy_with_logits(model(xb), yb).item() * len(yb)
                vc += len(yb)
        val_loss = vl / vc
        val_auc = roc_auc_score(yva, predict_proba(model, Xva, device))
        sched.step(val_auc)

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_auc"].append(float(val_auc))
        hist["lr"].append(opt.param_groups[0]["lr"])

        if val_auc > best_auc + 1e-4:
            best_auc, bad = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (epoch == 1 or epoch % 10 == 0 or bad == 0):
            print(f"  epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
                  f"| val_AUC {val_auc:.4f} | lr {opt.param_groups[0]['lr']:.1e}")
        if bad >= cfg["patience"]:
            print(f"  early stopping at epoch {epoch} (best val AUC {best_auc:.4f})")
            break

    model.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmax(hist["val_auc"]) + 1)
    hist["seconds"] = round(time.time() - t0, 2)
    return model, hist

def evaluate(y_true, prob, threshold=0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, prob)),
        pr_auc=float(average_precision_score(y_true, prob)),
        brier=float(brier_score_loss(y_true, prob)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )

def best_threshold(y_true, prob) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), float(f1[i])

def plot_eda(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    counts = df["y"].value_counts().sort_index()
    bars = ax[0].bar(["No (0)", "Yes (1)"], counts.values, color=["#4C72B0", "#DD8452"])
    ax[0].bar_label(bars, fmt="%d")
    ax[0].set_title(f"Class balance  ({counts[1] / len(df):.1%} positive)")
    ax[0].set_ylabel("clients")

    sns.kdeplot(data=df, x="age", hue="y", common_norm=False, fill=True, ax=ax[1],
                palette={0: "#4C72B0", 1: "#DD8452"})
    ax[1].set_title("Age distribution by outcome")
    ax[1].set_xlim(17, 90)
    fig.tight_layout()
    fig.savefig(out / "01_class_balance_and_age.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    order = df.groupby("job")["y"].mean().sort_values().index
    sns.barplot(data=df, x="y", y="job", order=order, ax=ax[0], errorbar=None, color="#55A868")
    ax[0].set_title("Subscription rate by job")
    ax[0].set_xlabel("P(subscribe)")
    sns.barplot(data=df, x="y", y="poutcome", ax=ax[1], errorbar=None, color="#C44E52")
    ax[1].set_title("Subscription rate by previous campaign outcome")
    ax[1].set_xlabel("P(subscribe)")
    fig.tight_layout()
    fig.savefig(out / "02_categorical_effects.png", bbox_inches="tight")
    plt.close(fig)

def plot_training(hist: dict, out: Path) -> None:
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(ep, hist["train_loss"], label="train (weighted BCE)")
    ax[0].plot(ep, hist["val_loss"], label="validation (BCE)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss curves")
    ax[0].legend()
    ax[1].plot(ep, hist["val_auc"], color="#C44E52", label="validation ROC-AUC")
    ax[1].axvline(hist["best_epoch"], ls="--", c="grey", label=f"best epoch ({hist['best_epoch']})")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("ROC-AUC"); ax[1].set_title("Validation AUC")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "03_training_curves.png", bbox_inches="tight")
    plt.close(fig)

def plot_confusion(y_true, prob, thr, out: Path, tag="") -> None:
    cm = confusion_matrix(y_true, (prob >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (test, thr={thr:.2f})")
    fig.tight_layout()
    fig.savefig(out / f"04_confusion_matrix{tag}.png", bbox_inches="tight")
    plt.close(fig)

def plot_roc_pr(y_true, prob, out: Path, tag="", label="DNN") -> None:
    fpr, tpr, _ = roc_curve(y_true, prob)
    prec, rec, _ = precision_recall_curve(y_true, prob)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
    ax[0].plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc_score(y_true, prob):.4f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC curve (test set)"); ax[0].legend(loc="lower right")

    base = y_true.mean()
    ax[1].plot(rec, prec, lw=2, color="#C44E52",
               label=f"{label} (AP={average_precision_score(y_true, prob):.4f})")
    ax[1].axhline(base, ls="--", c="k", lw=1, label=f"random ({base:.3f})")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-Recall curve (test set)"); ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / f"05_roc_pr_curves{tag}.png", bbox_inches="tight")
    plt.close(fig)

def plot_thresholds(y_true, prob, thr_best, out: Path) -> None:
    thrs = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in thrs]
    ps = [precision_score(y_true, (prob >= t).astype(int), zero_division=0) for t in thrs]
    rs = [recall_score(y_true, (prob >= t).astype(int), zero_division=0) for t in thrs]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(thrs, ps, label="precision")
    ax.plot(thrs, rs, label="recall")
    ax.plot(thrs, f1s, label="F1", lw=2.5)
    ax.axvline(thr_best, ls="--", c="grey", label=f"chosen threshold = {thr_best:.2f}")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("score")
    ax.set_title("Precision / Recall / F1 versus threshold (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "06_threshold_analysis.png", bbox_inches="tight")
    plt.close(fig)

def permutation_importance(model, X, y, cols, device, n_repeats=5, top=15) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    drops = np.zeros(X.shape[1])
    base = roc_auc_score(y, predict_proba(model, X, device))
    for j in range(X.shape[1]):
        d = 0.0
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            d += base - roc_auc_score(y, predict_proba(model, Xp, device))
        drops[j] = d / n_repeats
    imp = pd.DataFrame({"feature": cols, "delta_auc": drops}).sort_values("delta_auc", ascending=False)
    return imp.head(top)

def plot_importance(imp: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.4))
    d = imp.iloc[::-1]
    ax.barh(d["feature"], d["delta_auc"], color="#4C72B0")
    ax.set_xlabel("mean drop in test ROC-AUC when shuffled")
    ax.set_title("Permutation importance (top %d features)" % len(imp))
    fig.tight_layout()
    fig.savefig(out / "07_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)

def run(df: pd.DataFrame | None = None, cfg: dict = CFG, out: Path = OUT,
        device: str | None = None, verbose=True) -> dict:
    set_seed()
    out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = engineer_features(load_data() if df is None else df)

    y = df["y"].to_numpy()
    X, cols, scaler = build_matrix(df)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=cfg["val_size"] + cfg["test_size"],
                                            stratify=y, random_state=SEED)
    Xva, Xte, yva, yte = train_test_split(
        Xtmp, ytmp, test_size=cfg["test_size"] / (cfg["val_size"] + cfg["test_size"]),
        stratify=ytmp, random_state=SEED)

    print(f"Rows: {len(df):,} | features after encoding: {len(cols)} | "
          f"split train/val/test = {len(Xtr):,}/{len(Xva):,}/{len(Xte):,} | device={device}")
    plot_eda(df, out)

    print("Training DNN ...")
    model, hist = train_dnn(Xtr, ytr, Xva, yva, X.shape[1], cfg, device, verbose)
    prob_va = predict_proba(model, Xva, device)
    thr, f1_va = best_threshold(yva, prob_va)
    prob_te = predict_proba(model, Xte, device)
    m_dnn = evaluate(yte, prob_te, thr)
    m_dnn["f1_at_0.5"] = evaluate(yte, prob_te, 0.5)["f1"]
    m_dnn["recall_at_0.5"] = evaluate(yte, prob_te, 0.5)["recall"]
    print(f"DNN  test: acc={m_dnn['accuracy']:.4f} prec={m_dnn['precision']:.4f} "
          f"rec={m_dnn['recall']:.4f} F1={m_dnn['f1']:.4f} AUC={m_dnn['roc_auc']:.4f} "
          f"AP={m_dnn['pr_auc']:.4f}")

    lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(Xtr, ytr)
    m_lr = evaluate(yte, lr.predict_proba(Xte)[:, 1], thr)
    print(f"LogReg test: acc={m_lr['accuracy']:.4f} F1={m_lr['f1']:.4f} AUC={m_lr['roc_auc']:.4f}")

    df_abl = df.drop(columns=["duration"])
    Xa, colsa, sca = build_matrix(df_abl)
    Atr, Atmp, atr, atmp = train_test_split(Xa, y, test_size=cfg["val_size"] + cfg["test_size"],
                                            stratify=y, random_state=SEED)
    Ava, Ate, ava, ate = train_test_split(
        Atmp, atmp, test_size=cfg["test_size"] / (cfg["val_size"] + cfg["test_size"]),
        stratify=atmp, random_state=SEED)
    print("Training ablation (no 'duration') ...")
    model_a, hist_a = train_dnn(Atr, atr, Ava, ava, Xa.shape[1], cfg, device, verbose)
    prob_a = predict_proba(model_a, Ate, device)
    thr_a, _ = best_threshold(ava, predict_proba(model_a, Ava, device))
    m_abl = evaluate(ate, prob_a, thr_a)
    print(f"Ablation test: AUC={m_abl['roc_auc']:.4f} F1={m_abl['f1']:.4f}")

    plot_training(hist, out)
    plot_confusion(yte, prob_te, thr, out)
    plot_roc_pr(yte, prob_te, out)
    plot_thresholds(yte, prob_te, thr, out)
    imp = permutation_importance(model, Xte, yte, cols, device)
    plot_importance(imp, out)
    imp.to_csv(out / "permutation_importance.csv", index=False)

    n_params = sum(p.numel() for p in model.parameters())
    results = dict(
        dataset="UCI Bank Marketing (ID 222), bank-full.csv",
        n_rows=int(len(df)), n_features=int(len(cols)), n_positive=int(y.sum()),
        positive_rate=float(y.mean()),
        params=int(n_params), device=str(device), config=cfg,
        split=dict(train=int(len(Xtr)), val=int(len(Xva)), test=int(len(Xte))),
        dnn=m_dnn, logistic_regression=m_lr, ablation_no_duration=m_abl,
        threshold_selected_on_val=float(thr),
        val_f1_at_selected_threshold=float(f1_va),
        top_features=imp.to_dict(orient="records"),
    )
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    hist["ablation_val_auc"] = hist_a["val_auc"]
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    torch.save({"state_dict": model.state_dict(), "n_features": X.shape[1],
                "hidden": cfg["hidden"], "dropout": cfg["dropout"]}, out / "model_dnn.pt")
    (out / "preprocessor.json").write_text(json.dumps({
        "feature_cols": cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "num_cols": NUM_COLS, "cat_cols": CAT_COLS,
        "engineered": ["was_contacted_before", "repeated_calls"],
    }, indent=2))
    print(f"Saved artifacts to {out}")
    return results

if __name__ == "__main__":
    res = run()
    print(json.dumps({k: res[k] for k in ("dnn", "logistic_regression", "ablation_no_duration")},
                     indent=2))
