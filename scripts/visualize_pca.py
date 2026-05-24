#!/usr/bin/env python3
"""記事アイキャッチ用に 2D PCA 散布図を生成。

依存: numpy, matplotlib, japanize-matplotlib (or font fallback)
出力: data/pca_scatter.png
事前に run_experiment.py を実行して data/embeddings.json を生成しておくこと。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    # macOS: ヒラギノを試す
    matplotlib.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "sans-serif"]

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "embeddings.json"
OUT = ROOT / "data" / "pca_scatter.png"

GROUPS = {
    "果物": ["りんご", "みかん", "ぶどう", "バナナ", "いちご", "もも", "メロン", "スイカ"],
    "野菜": ["トマト", "ナス", "玉ねぎ", "キャベツ"],
    "Apple系": ["Apple", "apple", "アップル", "iPhone", "Mac", "Apple Inc."],
    "りんご表記": ["りんご", "林檎", "アップル"],
    "無関係": ["自動車", "椅子", "本", "パソコン", "鉛筆", "ビル", "机", "バット"],
}
COLORS = {
    "果物": "#e74c3c",
    "野菜": "#27ae60",
    "Apple系": "#2980b9",
    "りんご表記": "#f39c12",
    "無関係": "#7f8c8d",
}


def main():
    if not CACHE.exists():
        raise SystemExit("data/embeddings.json が無い。先に run_experiment.py を実行してください。")
    cache = json.loads(CACHE.read_text())

    plotted_words = []
    plotted_labels = []
    for label, words in GROUPS.items():
        for w in words:
            if w in cache:
                plotted_words.append(w)
                plotted_labels.append(label)

    X = np.array([cache[w] for w in plotted_words])
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(10, 7))
    for label in GROUPS:
        idx = [i for i, lab in enumerate(plotted_labels) if lab == label]
        if not idx:
            continue
        ax.scatter(coords[idx, 0], coords[idx, 1], s=120,
                   c=COLORS[label], label=label, alpha=0.75, edgecolors="white", linewidths=1.5)
        for i in idx:
            ax.annotate(plotted_words[i], coords[i], fontsize=10,
                        xytext=(6, 4), textcoords="offset points")

    ax.set_title("text-embedding-3-large (3072d) → PCA 2D 投影", fontsize=13)
    ax.set_xlabel(f"PC1 (var ratio {S[0] ** 2 / (S ** 2).sum() * 100:.1f}%)")
    ax.set_ylabel(f"PC2 (var ratio {S[1] ** 2 / (S ** 2).sum() * 100:.1f}%)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
