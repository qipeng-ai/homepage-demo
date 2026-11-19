# -*- coding: utf-8 -*-
"""下载 NVIDIA 近三年 10-K 年报全文（SEC EDGAR）。
用法: python scripts/download_reports.py
说明: 10-K 是英伟达（美股）的年报，对应作业里"近几年的年报"。
      财报 PDF/HTML 不进代码仓库，仓库里只保留本脚本。
"""
import json
import pathlib
import urllib.request

SEC_HEADERS = {"User-Agent": "homework research contact@example.com"}  # SEC 要求声明 UA
CIK = "0001045810"  # NVIDIA CIK

# 想要的财年: filingDate -> 输出文件名
WANT = 3  # 最近 3 份 10-K

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "data" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    with urllib.request.urlopen(req) as r:
        return r.read()


def main() -> None:
    subs = json.loads(fetch(f"https://data.sec.gov/submissions/CIK{CIK}.json"))
    recent = subs["filings"]["recent"]
    ten_ks = [
        (d, a, p)
        for f, d, a, p in zip(recent["form"], recent["filingDate"],
                              recent["accessionNumber"], recent["primaryDocument"])
        if f == "10-K"
    ][:WANT]
    for date, acc, doc in ten_ks:
        accn = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/1045810/{accn}/{doc}"
        out = OUT_DIR / f"{doc.replace('.htm', '')}_10K.htm"
        print(f"下载 {date} 10-K -> {out.name}")
        out.write_bytes(fetch(url))
    print(f"完成，共 {len(ten_ks)} 份，保存在 {OUT_DIR}")


if __name__ == "__main__":
    main()
