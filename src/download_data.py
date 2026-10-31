from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] already present: {dest.relative_to(ROOT)}")
        return dest
    print(f"[get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    print(f"[ok  ] {dest.relative_to(ROOT)}  ({dest.stat().st_size/1e6:.2f} MB)")
    return dest

def get_bank_marketing() -> Path:
    out_dir = DATA / "bank"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "bank-full.csv"
    if target.exists():
        print(f"[skip] already present: {target.relative_to(ROOT)}")
        return target

    outer = _download(BANK_URL, out_dir / "bank+marketing.zip")
    inner = out_dir / "bank.zip"
    with zipfile.ZipFile(outer) as z:
        names = z.namelist()
        print(f"[zip ] outer members: {names}")
        if "bank.zip" in names:
            with z.open("bank.zip") as src, open(inner, "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            z.extractall(out_dir)

    if inner.exists():
        with zipfile.ZipFile(inner) as z2:
            print(f"[zip ] inner members: {z2.namelist()}")
            z2.extractall(out_dir)

    if not target.exists():
        cands = list(out_dir.rglob("bank-full.csv"))
        if not cands:
            sys.exit("bank-full.csv not found inside the downloaded archive")
        shutil.copy2(cands[0], target)

    _cleanup(out_dir, keep={"bank-full.csv"})
    print(f"[ok  ] {target.relative_to(ROOT)}")
    return target

def get_fashion_mnist() -> Path:
    import torchvision

    out_dir = DATA / "FashionMNIST"
    out_dir.mkdir(parents=True, exist_ok=True)
    present = list((out_dir / "raw").glob("*.gz")) if (out_dir / "raw").exists() else []
    if len(present) < 4:
        print("[get ] Fashion-MNIST (torchvision download)")
        torchvision.datasets.FashionMNIST(root=str(out_dir), train=True, download=True)
        torchvision.datasets.FashionMNIST(root=str(out_dir), train=False, download=True)
    print(f"[ok  ] {out_dir.relative_to(ROOT)}  (files: {len(list((out_dir/'raw').glob('*.gz')))})")
    return out_dir

def _cleanup(out_dir: Path, keep: set[str]) -> None:
    for p in out_dir.iterdir():
        if p.is_file() and p.name not in keep:
            p.unlink()

if __name__ == "__main__":
    get_bank_marketing()
    get_fashion_mnist()
    print("\nAll datasets ready under:", DATA)
