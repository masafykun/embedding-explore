#!/usr/bin/env python3
"""記事の実験を一気通貫で再現するスクリプト。

依存: 標準ライブラリのみ。OPENAI_API_KEY 環境変数が必要。
出力: data/embeddings.json と各実験のテーブル/ランキング。
"""
from __future__ import annotations
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

API_URL = "https://api.openai.com/v1/embeddings"
MODEL = "text-embedding-3-large"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CACHE = DATA_DIR / "embeddings.json"

FRUITS = ["りんご", "みかん", "ぶどう", "バナナ", "いちご", "もも"]
NON_FRUITS = ["自動車", "椅子", "本", "パソコン", "鉛筆", "ビル"]
APPLE_VARIANTS = ["りんご", "林檎", "アップル", "Apple", "apple",
                  "iPhone", "Mac", "果物", "fruit", "自動車"]
TEST = ["メロン", "スイカ", "キャベツ", "トマト", "ナス", "玉ねぎ",
        "バット", "椎名林檎", "アップルパイ", "リンゴジュース",
        "くだもの狩り", "Apple Inc.", "机", "花"]


def fetch_embeddings(words: list[str]) -> dict[str, list[float]]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    missing = [w for w in words if w not in cache]
    if missing:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("OPENAI_API_KEY が未設定です")
        req = urllib.request.Request(
            API_URL,
            data=json.dumps({"model": MODEL, "input": missing}).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            payload = json.load(r)
        for w, item in zip(missing, payload["data"]):
            cache[w] = item["embedding"]
        CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    return {w: cache[w] for w in words}


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def mean(vs):
    n = len(vs)
    dim = len(vs[0])
    return [sum(v[i] for v in vs) / n for i in range(dim)]


def normed(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def fmt_bar(score, width=40, threshold=0):
    n = int(abs(score) * width * 2)
    ch = "█" if score >= threshold else "·"
    return ch * n


def main():
    all_words = sorted(set(FRUITS + NON_FRUITS + APPLE_VARIANTS + TEST))
    emb = fetch_embeddings(all_words)
    print(f"# キャッシュ: {len(emb)} 語の embedding (model={MODEL}, dim={len(next(iter(emb.values())))})")

    # 1. りんご vs みかん
    a, b = emb["りんご"], emb["みかん"]
    print(f"\n## りんご <-> みかん  cos={cos(a, b):.4f}  angle={math.degrees(math.acos(cos(a, b))):.2f}°")

    # 2. 5-word matrix
    print("\n## 類似度マトリックス: りんご/みかん/ぶどう/バナナ/自動車")
    ws = ["りんご", "みかん", "ぶどう", "バナナ", "自動車"]
    print("       " + "  ".join(f"{w:>5}" for w in ws))
    for i, w in enumerate(ws):
        row = [f"{cos(emb[w], emb[ws[j]]):.3f}" for j in range(len(ws))]
        print(f"{w:>5}  " + "  ".join(row))

    # 3. Apple polysemy
    print("\n## Apple の多義性: ‘りんご’ と ‘Apple’ からの近さ")
    for base in ("りんご", "Apple"):
        print(f"\n  -- {base} 視点 --")
        items = sorted(
            ((cos(emb[base], emb[w]), w) for w in APPLE_VARIANTS if w != base),
            reverse=True,
        )
        for s, w in items:
            print(f"  {w:>8}  {s:+.4f}  {fmt_bar(s)}")

    # 4. fruit_direction
    fd_raw = [mf - mo for mf, mo in zip(mean([emb[w] for w in FRUITS]),
                                        mean([emb[w] for w in NON_FRUITS]))]
    fd = normed(fd_raw)

    print("\n## 集中度: 単一次元では果物概念を捉えられない")
    total = sum(x * x for x in fd_raw)
    top = sorted(range(len(fd_raw)), key=lambda i: -abs(fd_raw[i]))[:10]
    print(f"  全次元 ||diff||^2 = {total:.4f}")
    print(f"  top10 寄与         = {sum(fd_raw[i] ** 2 for i in top):.4f}  ({100 * sum(fd_raw[i] ** 2 for i in top) / total:.1f}%)")
    print(f"  top1  寄与         = {fd_raw[top[0]] ** 2:.4f}  ({100 * fd_raw[top[0]] ** 2 / total:.1f}%)")

    # 5. projection ranking
    print("\n## 果物方向への射影スコア (新規語)")
    scored = sorted(((cos(fd, emb[w]), w) for w in TEST), reverse=True)
    for s, w in scored:
        print(f"  {w:>12}  {s:+.4f}  {fmt_bar(s)}")


if __name__ == "__main__":
    main()
