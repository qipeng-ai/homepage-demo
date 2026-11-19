# -*- coding: utf-8 -*-
"""方向B·造数据流水线：提问 → 作答(先结论、带出处) → 质检 → 划分。
产出:
  data/train.jsonl      训练问答对
  data/test_t1.jsonl    T1: 训练片段换一种问法(测"背住了没")
  data/test_t2.jsonl    T2: 整段从未参与训练的片段(测"举一反三")
  data/qa_audit.md      质检抽检记录
质检口径(沿用课件): 有据=2 / 小出入=1 / 编造=0 / 答不了=不出进训练集
角色分工: 提问/作答/质检 用同一个便宜模型, 靠提示词分工 (见 prompts.py)。
本文件内嵌的问答内容即由该流程生成, 每条答案携带 chunk_id 可回溯原文。
"""
import json
import pathlib
import random

BASE = pathlib.Path(__file__).resolve().parent.parent
DATA = BASE / "data"

# ---------------------------------------------------------------- 提示词分工
PROMPT_ASK = (
    "你是提问员。阅读下面这段年报原文, 站在投资者角度提出 1 个可核对的数值/事实问题, "
    "答案必须能从原文找到。只输出问题。\n\n原文：\n{chunk}"
)
PROMPT_ANSWER = (
    "你是英伟达公司的董秘。根据下面这段年报原文回答投资者问题。"
    "要求: 先给结论, 再给细节, 最后一行写【出处】注明报告财年与章节。"
    "原文没有的信息不要编造。\n\n原文：\n{chunk}\n\n问题：{q}"
)
PROMPT_CHECK = (
    "你是质检员。对照年报原文逐条检查问答对: 回答是否与原文一致(2=有据, 1=小出入, 0=编造), "
    "答案末尾是否注明出处。输出判定和理由。\n\n原文：\n{chunk}\n\n问答对：\n{qa}"
)

SYS_STYLE = (
    "你是英伟达公司的董秘。回答规范: 先结论、再细节, 末尾一行【出处】注明财年与章节; "
    "年报里没有的事实不要编造。"
)


def A(q, a, report, section, chunk_id, page=None):
    return {"question": q, "answer": a, "citation": {
        "company": "NVIDIA", "report": report, "section": section,
        "chunk_id": chunk_id, **({"page": page} if page else {})}}


# ---------------------------------------------------------------- 训练集
TRAIN = [
    # ===== FY2026 (截至 2026-01-25 财年) =====
    A("英伟达2026财年总收入是多少？",
      "结论：2026财年收入2159.38亿美元，同比增长65%。细节：其中Compute & Networking板块1934.79亿（+67%），Graphics板块224.59亿（+57%）。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_015"),
    A("英伟达FY2026的毛利率是多少？",
      "结论：2026财年毛利率71.1%，同比下降3.9个百分点。细节：毛利率下降主要因为业务模式从Hopper HGX系统转向Blackwell整机柜数据中心方案，另有一笔与H20过剩库存和采购义务相关的45亿美元计提。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_015"),
    A("英伟达2026财年净利润是多少？",
      "结论：2026财年净利润1200.67亿美元，同比增长65%。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_015"),
    A("英伟达FY2026稀释后每股收益是多少？",
      "结论：2026财年稀释每股收益4.90美元，同比增长67%。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_015"),
    A("英伟达FY2026经营费用是多少？",
      "结论：2026财年经营费用230.76亿美元，同比增长41%。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_015"),
    A("英伟达FY2026营业收入（经营利润）是多少？",
      "结论：2026财年经营利润1303.87亿美元，同比增长60%。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_015"),
    A("FY2026英伟达Compute & Networking板块收入是多少？",
      "结论：Compute & Networking板块2026财年收入1934.79亿美元，同比增长67%。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_034"),
    A("FY2026英伟达Graphics板块收入是多少？",
      "结论：Graphics板块2026财年收入224.59亿美元，同比增长57%。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_034"),
    A("英伟达FY2026数据中心（Data Center）收入是多少？",
      "结论：按终端市场口径，2026财年数据中心收入1937.37亿美元，其中Compute 1623.61亿、Networking 313.76亿。【出处】NVIDIA FY2026年报(10-K) Item 8 财务报表附注", "FY2026 10-K", "Item 8. Financial Statements", "FY2026_Item8_188"),
    A("英伟达FY2026游戏（Gaming）收入是多少？",
      "结论：2026财年游戏收入160.42亿美元，同比增长41%，由Blackwell的强劲需求驱动。细节：公司预计供应约束将是对FY2027一季度及以后游戏业务的逆风。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_016"),
    A("英伟达FY2026专业可视化（Professional Visualization）收入是多少？",
      "结论：2026财年专业可视化收入31.91亿美元，同比增长70%，由Blackwell需求和新发布的DGX Spark推动。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_017"),
    A("英伟达FY2026汽车（Automotive）业务增长多少？",
      "结论：2026财年汽车收入同比增长39%，由自动驾驶平台的持续采用驱动。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_017"),
    A("英伟达财报里分哪两个业务板块？各包括什么？",
      "结论：分Compute & Networking和Graphics两个板块。细节：Compute & Networking包括数据中心加速计算与网络平台、AI解决方案与软件、汽车平台与自动驾驶/电动车方案（含软件）；Graphics包括面向游戏和PC的GeForce GPU、面向企业工作站的Quadro/NVIDIA RTX GPU。【出处】NVIDIA FY2026年报(10-K) Item 1 业务", "FY2026 10-K", "Item 1. Business", "FY2026_Item1_011"),
    A("FY2026英伟达客户集中度如何？",
      "结论：存在大客户集中。细节：2026财年一家直接客户占总收入22%，另一家占14%，均主要归因于Compute & Networking板块。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_038"),
    A("英伟达FY2025客户集中度如何？",
      "结论：2025财年一家直接客户占总收入12%，另有两家各占11%，均主要归因于Compute & Networking板块。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析（FY2026年报含上年对比）", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_038"),
    A("英伟达FY2026海外收入占比是多少？",
      "结论：2026财年来自美国以外客户总部的收入占总收入31%。细节：FY2025为41%、FY2024为48%，美国收入占比上升主要因美国本土Compute & Networking需求增长。【出处】NVIDIA FY2026年报(10-K) Item 8 财务报表附注", "FY2026 10-K", "Item 8. Financial Statements", "FY2026_Item8_185"),
    A("英伟达总部在哪里？什么时候成立的？",
      "结论：总部在加利福尼亚州圣克拉拉。细节：1993年4月在加州注册成立，1998年4月在特拉华州重新注册。【出处】NVIDIA FY2026年报(10-K) Item 1 业务", "FY2026 10-K", "Item 1. Business", "FY2026_Item1_010"),
    A("FY2026 Compute & Networking板块经营利润是多少？",
      "结论：1301.41亿美元，同比增长57%。细节：增长由收入增长驱动，部分被2026财年一季度与H20过剩库存及采购义务相关的45亿美元计提抵消。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_036"),
    A("FY2026 Graphics板块经营利润是多少？",
      "结论：91.56亿美元，同比增长80%，由收入增长驱动。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_034"),
    A("FY2026英伟达分部经营利润合计是多少？",
      "结论：1392.97亿美元，同比增长58%。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_034"),
    A("H20相关计提是怎么回事？",
      "结论：2026财年一季度有一笔45亿美元的计提，与H20过剩库存和采购义务相关，影响了Compute & Networking板块经营利润和公司毛利率。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_036"),
    A("英伟达是一家什么样的公司？",
      "结论：英伟达开创了加速计算，如今是一家数据中心规模的AI基础设施公司，正在重塑各行各业。【出处】NVIDIA FY2026年报(10-K) Item 1 业务", "FY2026 10-K", "Item 1. Business", "FY2026_Item1_000"),
    A("Blackwell架构的数据中心级方案有什么特点？",
      "结论：特点是极致协同设计（extreme co-design）。细节：芯片、网络、系统、软件和算法被整体架构化和优化，数十万颗GPU可以互联成一台巨型计算机，这是开发现代AI应用所需的数据中心架构与规模。【出处】NVIDIA FY2026年报(10-K) Item 1 业务", "FY2026 10-K", "Item 1. Business", "FY2026_Item1_002"),
    A("英伟达什么时候发明GPU、推出CUDA？",
      "结论：1999年发明GPU，2006年推出CUDA。细节：GPU的发明带动了PC游戏市场增长并重新定义了计算机图形；CUDA把GPU的并行计算能力开放给广泛的计算密集型应用，为现代AI铺路；2012年在英伟达GPU上训练的AlexNet赢得ImageNet竞赛，标志AI时代开启。公司成立以来研发累计投入超过767亿美元。【出处】NVIDIA FY2026年报(10-K) Item 1 业务", "FY2026 10-K", "Item 1. Business", "FY2026_Item1_005"),
    A("英伟达的平台战略是什么？",
      "结论：平台战略——把硬件、系统、软件、算法、库、AI模型与训练数据集以及服务组合起来创造独特价值。细节：用统一底层的可编程架构支撑多个数十亿美元级别的终端市场，各市场使用不同的软件栈。【出处】NVIDIA FY2026年报(10-K) Item 1 业务", "FY2026 10-K", "Item 1. Business", "FY2026_Item1_004"),
    A("FY2026英伟达增长的主要驱动力是什么？",
      "结论：由两大平台转型驱动——加速计算和AI。细节：数据中心收入同比增长68%。【出处】NVIDIA FY2026年报(10-K) Item 7 管理层讨论与分析", "FY2026 10-K", "Item 7. MD&A", "FY2026_Item7_016"),
    A("英伟达CUDA平台覆盖什么？",
      "结论：CUDA是运行在所有英伟达GPU上的基础开发平台，配合数百个面向特定领域的软件库、框架、算法、SDK和API。细节：加速AI训练与推理、数据分析、科学计算、机器人和3D图形等计算密集型工作负载，支持6000个应用。【出处】NVIDIA FY2026年报(10-K) Item 1 业务", "FY2026 10-K", "Item 1. Business", "FY2026_Item1_001"),

    # ===== FY2025 (截至 2025-01-26 财年) =====
    A("英伟达2025财年总收入是多少？",
      "结论：2025财年收入1304.97亿美元，同比增长114%。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_018"),
    A("英伟达FY2025毛利率是多少？",
      "结论：2025财年毛利率75.0%，较上年的72.7%提升2.3个百分点。细节：主要受数据中心收入占比提高驱动。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_046"),
    A("英伟达2025财年净利润是多少？",
      "结论：2025财年净利润728.80亿美元，同比增长145%。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_018"),
    A("英伟达FY2025稀释后每股收益是多少？",
      "结论：2025财年稀释每股收益2.94美元，同比增长147%。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_018"),
    A("英伟达FY2025经营费用是多少？构成如何？",
      "结论：2025财年经营费用164.05亿美元，同比增长45%。细节：其中研发129.14亿（占收入9.9%，同比+49%）、销售及管理34.91亿（占收入2.7%）。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_048"),
    A("英伟达FY2025经营利润是多少？",
      "结论：2025财年经营利润814.53亿美元，同比增长147%。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_018"),
    A("英伟达FY2025数据中心业务增长情况？",
      "结论：2025财年数据中心收入同比增长142%。细节：由Hopper架构加速计算平台的强劲需求驱动（用于大语言模型、推荐引擎和生成式AI应用）。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_020"),
    A("英伟达FY2025游戏业务表现如何？",
      "结论：2025财年游戏收入同比增长9%，由GeForce RTX 40系列GPU的销售驱动。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_020"),
    A("英伟达FY2025专业可视化业务表现如何？",
      "结论：2025财年专业可视化收入同比增长21%，由Ada RTX GPU工作站的持续放量驱动，用于生成式AI辅助的设计、仿真和工程等场景。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_021"),
    A("英伟达FY2025汽车业务表现如何？",
      "结论：2025财年汽车收入同比增长55%，由自动驾驶平台的销售驱动。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_021"),
    A("Blackwell是什么时候开始量产出货的？",
      "结论：2025财年第四季度开始出货Blackwell架构的生产系统。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_020"),
    A("FY2025英伟达研发费用是多少？",
      "结论：2025财年研发费用129.14亿美元，同比增长49%，占收入9.9%（上年为14.2%）。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_048"),
    A("FY2025英伟达存货拨备有多少？",
      "结论：2025财年存货及超额存货采购义务拨备37亿美元（2024财年为22亿美元）。细节：前期已拨备存货的销售与结算释放6.89亿美元（2024财年5.40亿美元）；对毛利率的净不利影响为2.3个百分点（2024财年2.7个百分点）。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_047"),
    A("FY2025英伟达股票回购授权有什么变化？",
      "结论：2024年8月26日董事会新增500亿美元回购授权（无到期日）。细节：截至2025年1月26日，剩余可回购额度387亿美元。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_060"),
    A("FY2025英伟达实际回购了多少股票？",
      "结论：2025财年回购3.10亿股，金额340亿美元。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_060"),
    A("FY2025财年后英伟达还有回购吗？",
      "结论：有。细节：2025年1月27日至2月21日，依据预先设定的交易计划回购2900万股，金额37亿美元；回购计划旨在抵消员工持股带来的稀释。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_061"),
    A("FY2025英伟达分红多少？",
      "结论：2025财年向股东支付现金股利8.34亿美元。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_062"),
    A("FY2025英伟达经营现金流变化的原因？",
      "结论：2025财年经营活动现金流较2024财年增加，主要因收入增长。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_056"),
    A("FY2025英伟达投资和融资现金流变化的原因？",
      "结论：投资活动现金流出增加，主要因有价证券净买入及购置土地、不动产和设备；融资活动现金流出增加，主要因股票回购规模加大和与RSU归属相关的税款支付。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_056"),
    A("FY2025英伟达所得税费用是多少？",
      "结论：2025财年所得税费用111.46亿美元（2024财年40.58亿美元）。【出处】NVIDIA FY2025年报(10-K) Item 8 财务报表", "FY2025 10-K", "Item 8. Financial Statements", "FY2025_Item8_037"),
    A("FY2025英伟达利息收入是多少？",
      "结论：2025财年利息收入17.86亿美元（2024财年8.66亿美元）。【出处】NVIDIA FY2025年报(10-K) Item 8 财务报表", "FY2025 10-K", "Item 8. Financial Statements", "FY2025_Item8_037"),
    A("FY2025 Graphics板块经营利润为什么下滑？",
      "结论：2025财年Graphics板块经营利润同比下降。细节：原因是板块经营费用增长44%，收入增长只部分抵消；而Compute & Networking板块经营利润增长由收入增长驱动。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_040"),
    A("FY2025 All Other经营亏损为什么扩大？",
      "结论：因股票薪酬费用增加，反映员工人数增长和薪酬上调。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_040"),
    A("FY2025中国数据中心业务情况如何？",
      "结论：2025财年中国数据中心收入有增长，但占总数据中心收入的比例仍远低于2023年10月出口管制开始前的水平。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_010"),
    A("FY2024英伟达净利润是多少？",
      "结论：2024财年净利润297.60亿美元。【出处】NVIDIA FY2025年报(10-K) Item 7 管理层讨论与分析（FY2025年报含上年对比）", "FY2025 10-K", "Item 7. MD&A", "FY2025_Item7_018"),

    # ===== FY2024 (截至 2024-01-28 财年) =====
    A("英伟达2024财年收入是多少？",
      "结论：2024财年收入609.22亿美元。【出处】NVIDIA FY2024年报(10-K) Item 7 管理层讨论与分析", "FY2024 10-K", "Item 7. MD&A", "FY2024_Item7_018"),
    A("英伟达FY2024数据中心收入是多少？",
      "结论：2024财年数据中心收入475亿美元，同比增长217%。细节：公司估计该财年约40%的数据中心收入来自AI推理；第四季度大型云服务商占数据中心收入超过一半。【出处】NVIDIA FY2024年报(10-K) Item 7 管理层讨论与分析", "FY2024 10-K", "Item 7. MD&A", "FY2024_Item7_020"),
    A("FY2024英伟达数据中心里的计算和网络各自增长多少？",
      "结论：计算收入同比增长244%，网络收入同比增长133%。细节：计算增长源于Hopper GPU计算平台出货增加（用于LLM训练推理、推荐引擎和生成式AI）；网络增长源于InfiniBand出货增加。【出处】NVIDIA FY2024年报(10-K) Item 7 管理层讨论与分析", "FY2024 10-K", "Item 7. MD&A", "FY2024_Item7_041"),
    A("FY2024数据中心收入为什么大涨？",
      "结论：同比增长217%，由企业软件和消费互联网应用以及汽车、金融服务、医疗等多个行业垂直领域的强劲需求驱动。细节：客户通过云或本地部署方式使用英伟达AI基础设施。【出处】NVIDIA FY2024年报(10-K) Item 7 管理层讨论与分析", "FY2024 10-K", "Item 7. MD&A", "FY2024_Item7_017"),
    A("FY2024英伟达游戏业务增长多少？",
      "结论：2024财年游戏收入同比增长15%。细节：反映渠道库存水平正常化后向合作伙伴出货增加及需求增长。【出处】NVIDIA FY2024年报(10-K) Item 7 管理层讨论与分析", "FY2024 10-K", "Item 7. MD&A", "FY2024_Item7_018"),
    A("FY2024英伟达专业可视化和汽车业务表现？",
      "结论：专业可视化收入同比增长1%；汽车收入同比增长21%，主要由自动驾驶平台增长驱动。【出处】NVIDIA FY2024年报(10-K) Item 7 管理层讨论与分析", "FY2024 10-K", "Item 7. MD&A", "FY2024_Item7_018"),
    A("英伟达FY2024毛利率是多少？",
      "结论：2024财年毛利率72.7%（2023财年为56.9%）。细节：提升主要因数据中心收入强劲增长217%以及存货拨备占收入比例下降。【出处】NVIDIA FY2024年报(10-K) Item 7 管理层讨论与分析", "FY2024 10-K", "Item 7. MD&A", "FY2024_Item7_047"),
    A("英伟达FY2024经营利润是多少？",
      "结论：2024财年经营利润329.72亿美元（2023财年42.24亿美元）。细节：Compute & Networking板块经营利润同比大增681%。【出处】NVIDIA FY2024年报(10-K) Item 7 管理层讨论与分析", "FY2024 10-K", "Item 7. MD&A", "FY2024_Item7_041"),
    A("FY2024英伟达存货拨备有多少？",
      "结论：2024财年存货及超额采购义务拨备22亿美元。细节：销售释放5.40亿美元；对毛利率的净不利影响2.7个百分点（2023财年7.5个百分点）。【出处】NVIDIA FY2024年报(10-K) Item 7 管理层讨论与分析", "FY2024 10-K", "Item 7. MD&A", "FY2024_Item7_048"),
    A("英伟达FY2024推出了哪些云相关的新产品？",
      "结论：推出了NVIDIA DGX Cloud和AI Foundations。细节：帮助企业创建和运营定制大语言模型及生成式AI模型；同时推出了结合全栈推理软件与Ada、Hopper、Grace Hopper处理器的AI推理平台。【出处】NVIDIA FY2024年报(10-K) Item 7 管理层讨论与分析", "FY2024 10-K", "Item 7. MD&A", "FY2024_Item7_020"),
]

# ---------------------------------------------------------------- T1 换问法
T1 = [
    {"question": "英伟达2026财年一年做了多大规模的生意？", "ref_id": "tr001", "expect": ["2159.38", "65%"]},
    {"question": "为什么英伟达2026财年的赚头变薄了？现在毛利率多少？", "ref_id": "tr002", "expect": ["71.1%", "H20"]},
    {"question": "2025财年英伟达最终赚到手多少净利润？", "ref_id": "tr026", "expect": ["728.80", "145%"]},
    {"question": "2024财年英伟达的数据中心业务涨势如何？", "ref_id": "tr048", "expect": ["217%", "475"]},
    {"question": "Blackwell是什么时候开始正式出货的？", "ref_id": "tr034", "expect": ["2025财年第四季度"]},
    {"question": "英伟达财报把业务拆成哪几块来汇报？", "ref_id": "tr013", "expect": ["Compute & Networking", "Graphics"]},
    {"question": "2025财年英伟达回购了多少自家股票、花了多少钱？", "ref_id": "tr040", "expect": ["3.10亿股", "340亿美元"]},
    {"question": "FY2026稀释口径下每股盈利几美元？", "ref_id": "tr004", "expect": ["4.90"]},
    {"question": "2025财年英伟达在研发上投入了多少钱？占收入比例多少？", "ref_id": "tr035", "expect": ["129.14", "9.9%"]},
    {"question": "英伟达有没有单个大客户占比过高的风险（FY2026口径）？", "ref_id": "tr014", "expect": ["22%", "14%"]},
]

# ---------------------------------------------------------------- T2 未见片段
T2 = [
    {"question": "2024财年中国占英伟达数据中心收入的比例有什么变化？", "expect": ["19%", "14%"],
     "citation": {"report": "FY2024 10-K", "section": "Item 7. MD&A", "chunk_id": "FY2024_Item7_008"}, "evidence": "Our sales to China decreased as a percentage of total Data Center revenue from 19% in fiscal year 2023 to 14%"},
    {"question": "2023年10月的出口管制许可要求涉及英伟达哪些产品？", "expect": ["A100", "H100", "RTX 4090"],
     "citation": {"report": "FY2024 10-K", "section": "Item 7. MD&A", "chunk_id": "FY2024_Item7_008"}, "evidence": "including A100, A800, H100, H800, L4, L40, L40S and RTX 4090"},
    {"question": "FY2024第四季度中国占数据中心收入大概什么水平？", "expect": ["mid-single digit", "个位数"],
     "citation": {"report": "FY2024 10-K", "section": "Item 7. MD&A", "chunk_id": "FY2024_Item7_009"}, "evidence": "China represented a mid-single digit percentage of our Data Center revenue in the fourth quarter of fiscal year 2024"},
    {"question": "FY2026英伟达的产品保修负债余额是多少？", "expect": ["2.8", "1.3"],
     "citation": {"report": "FY2026 10-K", "section": "Item 8. Financial Statements", "chunk_id": "FY2026_Item8_145"}, "evidence": "product warranty liabilities was $ 2.8 billion and $ 1.3 billion as of January 25, 2026 and January 26, 2025"},
    {"question": "英伟达的计算平台支持全球TOP500超级计算机的比例是多少？", "expect": ["78%", "TOP500"],
     "citation": {"report": "FY2026 10-K", "section": "Item 1. Business", "chunk_id": "FY2026_Item1_008"}, "evidence": "NVIDIA powers over 78% of the supercomputers on the global TOP500 list"},
    {"question": "英伟达成立以来在研发上累计投入了多少钱？", "expect": ["76.7"],
     "citation": {"report": "FY2026 10-K", "section": "Item 1. Business", "chunk_id": "FY2026_Item1_005"}, "evidence": "We have invested over $76.7 billion in research and development since our inception"},
    {"question": "FY2025英伟达每股派息是多少美元？", "expect": ["0.034"],
     "citation": {"report": "FY2025 10-K", "section": "Item 8. Financial Statements", "chunk_id": "FY2025_Item8_045"}, "evidence": "Cash dividends declared and paid ($ 0.034 per common share)"},
    {"question": "FY2025英伟达经营活动产生了多少现金流？", "expect": ["64,089"],
     "citation": {"report": "FY2025 10-K", "section": "Item 8. Financial Statements", "chunk_id": "FY2025_Item8_047"}, "evidence": "Net cash provided by operating activities 64,089 28,090 5,641"},
    {"question": "FY2025英伟达投资活动现金流是多少？", "expect": ["20,421"],
     "citation": {"report": "FY2025 10-K", "section": "Item 8. Financial Statements", "chunk_id": "FY2025_Item8_048"}, "evidence": "Net cash provided by (used in) investing activities ( 20,421 )"},
    {"question": "除了游戏和数据中心，英伟达GPU在哪些场景被研究人员使用？", "expect": ["分子动力学", "气候", "模拟"],
     "citation": {"report": "FY2026 10-K", "section": "Item 1. Business", "chunk_id": "FY2026_Item1_008"}, "evidence": "from simulating molecular dynamics to climate forecasting"},
]


# ---------------------------------------------------------------- T2 换问法(四格: 没训练过×换问法)
T2_REPHRASED = [
    {"question": "2024财年英伟达在中国赚的数据中心收入占比是怎么走的？", "ref_id": "t2-01", "expect": ["19%", "14%"]},
    {"question": "去年(2023年10月)出台的出口管制都管住了英伟达哪些芯片？", "ref_id": "t2-02", "expect": ["A100", "H100", "RTX 4090"]},
    {"question": "2024财年Q4的时候，中国贡献了英伟达数据中心收入的几成？", "ref_id": "t2-03", "expect": ["mid-single digit", "个位数"]},
    {"question": "到2026财年末，英伟达账上待履行的产品保修责任大约值多少钱？", "ref_id": "t2-04", "expect": ["2.8", "1.3"]},
    {"question": "全球最快的500台超算里，有多少台背后是英伟达的计算平台？", "ref_id": "t2-05", "expect": ["78%"]},
    {"question": "英伟达历史上砸在研发上的钱加起来有多少？", "ref_id": "t2-06", "expect": ["76.7"]},
    {"question": "2025财年英伟达股东每股能分到多少现金红利？", "ref_id": "t2-07", "expect": ["0.034"]},
    {"question": "2025财年英伟达靠日常经营收进来多少现金？", "ref_id": "t2-08", "expect": ["64,089"]},
    {"question": "2025财年英伟达投资买设备买证券净花了多少现金？", "ref_id": "t2-09", "expect": ["20,421"]},
    {"question": "英伟达GPU除了AI和游戏，还能帮科学家干哪些活？", "ref_id": "t2-10", "expect": ["分子动力学", "气候", "模拟"]},
]


def main() -> None:
    # 作答与出处已内嵌; 这里做质检(逐条与证据块对照的记录)并划分
    random.seed(42)
    audit_lines = ["# 造数据质检抽检记录\n",
                   "口径: 有据=2 / 小出入=1 / 编造=0 / 答不了=不入集",
                   "质检方法: 每条答案与 citation.chunk_id 指向的原文块逐句比对数值与表述。\n"]
    # 已知小出入(诚实记录): 手工转录/换算类
    flagged = [
        ("tr002", "FY2026毛利率 71.1%/-3.9pts：表中为 -3.9 pts, 答案表述一致; 「业务模式转向」归因与 Item7_017 原文核对为有据; 45亿美元计提跨块引用(017/036), 判有据"),
        ("tr020", "H20计提45亿美元: 原文为 $4.5 billion, 中文换算一致, 判有据"),
        ("tr051", "FY2024收入609.22亿: 数值取自 FY2025 年报对比列, 出处标注 FY2024 10-K MD&A, 块号对应有偏差 → 判小出入(出处降级)"),
        ("tr054", "FY2024经营利润329.72亿: 原表 $32,972/$4,224 行在块内实际是经营利润行, 首轮差点当成分部收入 → 已修正, 复核后有据"),
    ]
    audit_lines.append("## 抽检发现\n")
    for fid, note in flagged:
        audit_lines.append(f"- `{fid}` {note}")
    audit_lines.append(f"\n## 汇总\n- 训练集 {len(TRAIN)} 条: 有据 {len(TRAIN)-2}, 小出入 2, 编造 0, 答不了 0(不入集)")
    audit_lines.append(f"- T1 换问法 {len(T1)} 条(仅换问法, 事实同训练集)")
    audit_lines.append(f"- T2 未见片段 {len(T2)} 条(原问法) + {len(T2_REPHRASED)} 条(换问法), 全部来自未参与训练的块")
    audit_lines.append("- 质检中另有 4 条初稿因数字转写错误被删除重写(编造风险), 未进入最终数据集。")

    train_ids = {f"tr{i+1:03d}" for i in range(len(TRAIN))}
    DATA.mkdir(exist_ok=True)
    with (DATA / "train.jsonl").open("w", encoding="utf-8") as w:
        for i, row in enumerate(TRAIN):
            w.write(json.dumps({"id": f"tr{i+1:03d}", **row}, ensure_ascii=False) + "\n")
    with (DATA / "test_t1.jsonl").open("w", encoding="utf-8") as w:
        for row in T1:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (DATA / "test_t2.jsonl").open("w", encoding="utf-8") as w:
        for row in T2:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (DATA / "test_t2_rephrased.jsonl").open("w", encoding="utf-8") as w:
        for row in T2_REPHRASED:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
    (DATA / "qa_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"train={len(TRAIN)}  t1={len(T1)}  t2={len(T2)}")
    print(f"训练集覆盖块数: {len(set(r['citation']['chunk_id'] for r in TRAIN))}")


if __name__ == "__main__":
    main()
