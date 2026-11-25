# -*- coding: utf-8 -*-
"""方向A · 检索问答脚本（向量知识库 + 大模型）。
流程: 问题向量化 -> 余弦检索 top-k 原文块 -> 塞进提示词 -> 调大模型 -> 带出处回答。

用法:
  python scripts/ask.py "英伟达FY2026总收入是多少？"
  python scripts/ask.py "..." --k 6 --temperature 0.2
  python scripts/ask.py "..." --no-kb            # 无知识库对照(格1)
  python scripts/ask.py "..." --show-sources     # 打印召回的块
API 配置从 nvidia-a/.env 读取 (模板见 .env.example)。
"""
import argparse
import json
import pathlib
import os
import sys
import urllib.request

import numpy as np

BASE = pathlib.Path(__file__).resolve().parent.parent
DATA = BASE / "data"
CHUNKS = BASE.parent / "nvidia-b" / "data" / "chunks.jsonl"
if not CHUNKS.exists():
    CHUNKS = DATA / "chunks.jsonl"

# ---------------------------------------------------------------- .env
def load_env() -> dict:
    env = {}
    f = BASE / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    env.update({k: v for k, v in os.environ.items() if k.startswith("LLM_")})
    return env


# ---------------------------------------------------------------- 检索
class KnowledgeBase:
    def __init__(self, embed_model: str):
        from sentence_transformers import SentenceTransformer
        self.rows = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
        self.vecs = np.load(DATA / "index.npy")
        self.qprefix = "query: " if "e5" in embed_model.lower() else ""
        self.model = SentenceTransformer(embed_model)

    def retrieve(self, question: str, k: int = 6):
        qv = self.model.encode([self.qprefix + question], normalize_embeddings=True)
        sims = (self.vecs @ qv.T).ravel()
        top = sims.argsort()[::-1][:k]
        return [(self.rows[i], float(sims[i])) for i in top]


# ---------------------------------------------------------------- 大模型
def chat(env: dict, messages: list, temperature: float) -> str:
    url = env["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": env["LLM_MODEL"],
        "messages": messages,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {env['LLM_API_KEY']}",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


SYS_WITH_KB = (
    "你是英伟达公司的董秘。只依据下面提供的年报原文片段回答投资者问题。"
    "规范: 先结论、再细节; 回答中用到的每个事实后面标注出处, 格式为【FY20XX·章节·块ID】; "
    "如果片段里没有足够信息, 明确回答\"该问题在所提供材料中未覆盖\", 不要编造。"
)

SYS_NO_KB = (
    "你是英伟达公司的董秘。回答投资者问题。"
    "规范: 先结论、再细节; 回答中用到的每个事实后面标注出处, 格式为【报告财年·章节】。"
)


def answer(env, kb, question, k=6, temperature=0.2, use_kb=True, show_sources=False):
    if use_kb:
        hits = kb.retrieve(question, k)
        ctx = "\n\n".join(
            f"[{r['fiscal_year']}·{r['section_cn']}·{r['chunk_id']}] {r['text'][:900]}"
            for r, _ in hits)
        messages = [
            {"role": "system", "content": SYS_WITH_KB + "\n\n年报原文片段:\n" + ctx},
            {"role": "user", "content": question},
        ]
        if show_sources:
            print("--- 召回的块 ---")
            for r, s in hits:
                print(f"  {s:.3f} {r['chunk_id']} {r['section_cn']}")
    else:
        messages = [{"role": "system", "content": SYS_NO_KB},
                    {"role": "user", "content": question}]
    return chat(env, messages, temperature)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--no-kb", action="store_true")
    ap.add_argument("--show-sources", action="store_true")
    a = ap.parse_args()

    env = load_env()
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if not env.get(key):
            sys.exit(f"缺少 {key}: 请复制 .env.example 为 .env 并填入配置")
    kb = None if a.no_kb else KnowledgeBase(env.get("EMBED_MODEL",
        "intfloat/multilingual-e5-base"))
    print(answer(env, kb, a.question, a.k, a.temperature, not a.no_kb, a.show_sources))


if __name__ == "__main__":
    main()
