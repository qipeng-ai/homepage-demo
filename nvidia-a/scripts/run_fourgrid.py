# -*- coding: utf-8 -*-
"""方向A · 四格对照实验（半结构化，作业要求）。
格1: 同一问题, 有知识库 vs 无知识库
格2: 同一问题, 正常问法 vs 扭曲问法(细节多、口语化、加时间限定, 语义等价)
格3: 问题相关内容在知识库里 vs 不在(测会不会编造)
格4: 默认温度(0.2) vs 高温度(1.0), 每题各跑2遍看稳定性
产出: data/results/fourgrid_detail.csv + 控制台汇总表
用法: python scripts/run_fourgrid.py
"""
import csv
import json
import pathlib
import sys

from ask import KnowledgeBase, answer, load_env

BASE = pathlib.Path(__file__).resolve().parent.parent
QUESTIONS = BASE / "data" / "questions.json"
RESULTS = BASE / "data" / "results"

Q_IN_KB = [
    "英伟达FY2026总收入是多少？",
    "英伟达FY2026的毛利率是多少？为什么下降？",
    "英伟达FY2025净利润是多少？",
    "英伟达FY2024数据中心收入是多少？同比增长多少？",
    "英伟达财报里分哪两个业务板块？各包括什么？",
    "FY2025英伟达回购了多少股票？花了多少钱？",
    "FY2026英伟达客户集中度如何？",
    "Blackwell是什么时候开始量产出货的？",
]
Q_DISTORTED = [
    "我先问个基础的哈，就是那家做GPU的英伟达，截至2026年1月25日的那个财政年度，全年加起来一共实现了多少销售收入（美元）来着？",
    "话说2026财年（结束于2026年1月下旬的那个年度）英伟达卖东西的毛利水平怎么样？好像比前一年低了一点，是出了什么事导致的吗？",
    "想请教一下，英伟达在2025财年——就是全年收入1300多亿美元那个年度——最后归属于股东的净利润具体是多少亿美元？",
    "帮我查查英伟达2024财年年报里，数据中心这块业务做了多大规模（多少亿美元），跟上一个财年相比涨了百分之多少？",
    "英伟达的年报里把业务分成了哪几个reportable segments（报告分部）？每个分部里面都装了哪些产品线，能展开讲讲吗？",
    "2025财年英伟达从公开市场买回来了多少自家股票？股数和金额分别说一下，就是回购那个事儿。",
    "英伟达2026财年有没有那种\"一个客户就占总收入很大一块\"的情况？具体比例是多少？",
    "英伟达的Blackwell架构产品是从哪个财年的哪个季度开始正式量产出货的？",
]
Q_OUT_KB = [
    "英伟达FY2027全年的业绩指引或收入预测是多少？",
    "英伟达2026自然年的股价最高涨到过多少美元？",
    "英伟达现任CFO的年薪是多少美元？",
    "AMD公司FY2025财年的总收入是多少？",
    "英伟达在中国大陆注册了几家子公司？分别叫什么名字？",
    "英伟达FY2026年每个月的员工人数变化是怎样的？",
    "英伟达的CUDA软件业务单独核算的收入是多少？",
    "英伟达现在的公司市值是多少美元？",
]


def run_grid(name, cases, kb, env, results):
    """cases: list of dict(question, expect_in, expect_admit, tag...)"""
    for c in cases:
        ans = answer(env, kb, c["question"], k=c.get("k", 6),
                     temperature=c.get("temperature", 0.2), use_kb=c.get("use_kb", True))
        # 半自动评分: 关键词命中 / 是否承认未覆盖
        hits = sum(e in ans for e in c.get("expect_in", []))
        n = len(c.get("expect_in", [])) or None
        admitted = any(w in ans for w in ("未覆盖", "没有足够", "材料中未", "无法回答", "未提及"))
        results.append({
            "格子": name, "设置": c["setting"], "问题": c["question"],
            "回答": ans,
            "关键数命中": f"{hits}/{n}" if n else "-",
            "承认未覆盖": "是" if admitted else "否",
            "结论对(人工复核)": "",
        })
        print(f"[{name}] {c['setting']} | 命中 {hits}/{n or '-'} | 承认未覆盖={admitted} | {c['question'][:30]}")


def main() -> None:
    env = load_env()
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if not env.get(key):
            sys.exit("缺少 API 配置: 复制 .env.example 为 .env 并填入")
    print("加载知识库...")
    kb = KnowledgeBase(env.get("EMBED_MODEL", "intfloat/multilingual-e5-base"))
    results = []

    # 格1: 有库 vs 无库 (同一批库内问题)
    run_grid("格1-有库vs无库",
             [{"question": q, "setting": "有库", "expect_in": []} for q in Q_IN_KB] +
             [{"question": q, "setting": "无库", "expect_in": [], "use_kb": False} for q in Q_IN_KB],
             kb, env, results)

    # 格2: 正常问法 vs 扭曲问法 (成对, 都有库)
    run_grid("格2-正常vs扭曲",
             [{"question": q, "setting": "正常问法", "expect_in": []} for q in Q_IN_KB] +
             [{"question": q, "setting": "扭曲问法", "expect_in": []} for q in Q_DISTORTED],
             kb, env, results)

    # 格3: 库内 vs 库外 (都有库)
    run_grid("格3-库内vs库外",
             [{"question": q, "setting": "库内", "expect_in": []} for q in Q_IN_KB] +
             [{"question": q, "setting": "库外", "expect_in": []} for q in Q_OUT_KB],
             kb, env, results)

    # 格4: 默认温度0.2 vs 高温度1.0, 每题2遍
    cases = []
    for q in Q_IN_KB[:8]:
        for rep in (1, 2):
            cases.append({"question": q, "setting": f"温度0.2-第{rep}遍", "temperature": 0.2, "expect_in": []})
            cases.append({"question": q, "setting": f"温度1.0-第{rep}遍", "temperature": 1.0, "expect_in": []})
    run_grid("格4-温度对比", cases, kb, env, results)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "fourgrid_detail.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # 汇总
    print("\n========== 汇总 ==========")
    from collections import defaultdict
    stat = defaultdict(lambda: [0, 0, 0])  # 命中数, 关键数, 承认未覆盖数
    total = defaultdict(int)
    for r in results:
        key = (r["格子"], r["设置"])
        total[key] += 1
        if "/" in r["关键数命中"]:
            h, n = r["关键数命中"].split("/")
            stat[key][0] += int(h)
            stat[key][1] += int(n)
        if r["承认未覆盖"] == "是":
            stat[key][2] += 1
    for (grid, setting), (h, n, adm) in sorted(stat.items()):
        rate = f" 命中率 {h}/{n}" if n else ""
        print(f"{grid} | {setting}: {total[(grid, setting)]}题{rate} 承认未覆盖{adm}次")
    print(f"\n明细已保存: {out}")
    print("请人工复核 csv 中'结论对(人工复核)'列, 再写一页结论")


if __name__ == "__main__":
    main()
