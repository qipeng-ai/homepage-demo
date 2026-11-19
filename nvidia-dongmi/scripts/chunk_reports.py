# -*- coding: utf-8 -*-
"""把 10-K HTML 切成 ~600 字的文本块，每块带公司 / 财年 / 章节(Item) / 块序号。
用法: python scripts/chunk_reports.py
产出: data/chunks.jsonl  每行一个块:
  {chunk_id, company, fiscal_year, report_date, section, section_cn, seq, text}
"""
import html
import json
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parent.parent
REPORTS = BASE / "data" / "reports"
OUT = BASE / "data" / "chunks.jsonl"

CHUNK_SIZE = 600  # 课件口径：每块约 600 字

SECTION_PATTERNS = [  # 按文档逻辑顺序排列，逐个向后定位
    ("Item 1. Business", "Item 1 业务", r"item\s+1\.\s*business"),
    ("Item 1A. Risk Factors", "Item 1A 风险因素", r"item\s+1a\.\s*risk\s+factors"),
    ("Item 7. MD&A", "Item 7 管理层讨论与分析", r"item\s+7\.\s*management"),
    ("Item 7A. Quantitative and Qualitative Disclosures About Market Risk", "Item 7A 市场风险定量披露", r"item\s+7a\."),
    ("Item 8. Financial Statements", "Item 8 财务报表", r"item\s+8\.\s*financial\s+statements"),
]


def html_to_text(raw: str) -> str:
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<br[^>]*>", "\n", raw, flags=re.I)
    raw = re.sub(r"</(p|div|tr|table|h[1-6])>", "\n", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\xa0]+", " ", raw)
    raw = re.sub(r"\n\s*\n+", "\n", raw)
    return raw.strip()


def find_sections(text: str):
    """定位各 Item 的正文起始位置，返回 [(start, section_en, section_cn)]。
    目录里也有同样的标题（后面跟着页码），要跳过；
    同时各 Item 必须按文档顺序递增出现，避免匹配到前文的交叉引用。"""
    marks = []
    lower = text.lower()
    cursor = 0
    for sec_en, sec_cn, pat in SECTION_PATTERNS:
        found = None
        for m in re.finditer(pat, lower):
            if m.start() <= cursor:
                continue
            after = text[m.end():m.end() + 400]
            # 目录特征: 标题后(一段标题文字之内)紧跟页码(数字换行)
            if re.search(r"\n\s*\d{1,3}\s*\n", after[:200]) or \
               re.search(r"\n\s*\d{1,3}\s*\n\s*Item\b", after):
                continue
            found = m.start()
            break
        if found is not None:
            marks.append((found, sec_en, sec_cn))
            cursor = found
    marks.sort()
    return marks


def chunk_text(text: str):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    buf, out = [], []
    cur = 0
    for p in paras:
        if cur + len(p) > CHUNK_SIZE and buf:
            out.append("\n".join(buf))
            buf, cur = [], 0
        buf.append(p)
        cur += len(p) + 1
    if buf:
        out.append("\n".join(buf))
    return out


def main() -> None:
    rows = []
    for f in sorted(REPORTS.glob("*.htm")):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(raw)
        # 财年从文件名推断: 20250126_10K -> FY2025 (截至 2025-01-26 的财年)
        fy = "FY" + f.name[:4]
        marks = find_sections(text)
        for i, (start, sec_en, sec_cn) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            for seq, blk in enumerate(chunk_text(text[start:end])):
                if len(blk) < 120:  # 太碎的块(目录残留等)丢弃
                    continue
                rows.append({
                    "chunk_id": f"{fy}_{sec_en.split('.')[0].replace(' ', '')}_{seq:03d}",
                    "company": "NVIDIA",
                    "fiscal_year": fy,
                    "report_date": f.name[5:13],
                    "section": sec_en,
                    "section_cn": sec_cn,
                    "seq": seq,
                    "text": blk,
                })
    with OUT.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 统计
    from collections import Counter
    c = Counter((r["fiscal_year"], r["section_cn"]) for r in rows)
    print(f"共 {len(rows)} 块 -> {OUT}")
    for k, v in sorted(c.items()):
        print(f"  {k[0]} {k[1]}: {v} 块")


if __name__ == "__main__":
    main()
