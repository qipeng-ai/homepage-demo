# -*- coding: utf-8 -*-
"""构建本地向量知识库（方向A 第3步）。
把 chunks.jsonl 的 1545 个文本块用多语言 embedding 模型向量化，
存成 data/index.npy（行序与 chunks.jsonl 一致）。
用法: python scripts/build_index.py
"""
import json
import pathlib

import numpy as np
from sentence_transformers import SentenceTransformer

BASE = pathlib.Path(__file__).resolve().parent.parent
DATA = BASE / "data"
# 复用方向B已切好的块(同一套材料)
CHUNKS = BASE.parent / "nvidia-b" / "data" / "chunks.jsonl"
if not CHUNKS.exists():
    CHUNKS = DATA / "chunks.jsonl"

# 多语言模型: 中文提问 x 英文年报原文 的跨语言检索
# e5 系列对 query/passage 有前缀要求; 换模型时改 EMBED_MODEL 即可
EMBED_MODEL = "intfloat/multilingual-e5-base"


def is_e5(name: str) -> bool:
    return "e5" in name.lower()


def main() -> None:
    import sys
    args = sys.argv[1:]
    model_name = args[0] if args else EMBED_MODEL
    rows = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    prefix = "passage: " if is_e5(model_name) else ""
    texts = [f"{prefix}{r['company']} {r['fiscal_year']} {r['section_cn']}\n{r['text'][:1200]}" for r in rows]
    print(f"加载模型 {model_name} ...")
    model = SentenceTransformer(model_name)
    print(f"编码 {len(texts)} 个文本块 ...")
    vecs = model.encode(texts, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=True)
    np.save(DATA / "index.npy", np.asarray(vecs, dtype=np.float32))
    print(f"完成: index.npy 形状 {vecs.shape}, 保存在 {DATA}")


if __name__ == "__main__":
    main()
