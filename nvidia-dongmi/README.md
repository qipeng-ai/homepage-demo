# 方向 B · 用英伟达财报训练一个「董秘」

《学会使用大模型》第三课作业，方向 B：把知识写进模型权重（深度四）。

仓库：https://github.com/qipeng-ai/homepage-demo （本项目位于 `nvidia-dongmi/` 子目录）

## 材料

NVIDIA（英伟达）近三年 10-K 年报（SEC EDGAR）：

| 财年 | 财年截止 | 文件 |
|---|---|---|
| FY2024 | 2024-01-28 | `data/reports/20240128_10K.htm` |
| FY2025 | 2025-01-26 | `data/reports/20250126_10K.htm` |
| FY2026 | 2026-01-25 | `data/reports/20260126_10K.htm` |

财报全文（约 6MB HTML）**不进仓库**（已被 `.gitignore` 排除），仓库只保留下载脚本，重新执行即可获取：

```bash
python scripts/download_reports.py
```

## 流水线（课件「深度四」口径）

```
下载 10-K ──> 切块(~600字/块, 带公司/财年/章节) ──> 提问 → 作答(先结论+带出处) → 质检 → 划分
                                                                        │
        Colab QLoRA 微调 (unsloth, r=16, lr=2e-4)  <────  train.jsonl ──┘
                                                                        │
        四格对照 + RAG 对比  <────  test_t1/t2/t2_rephrased.jsonl ────────┘
```

## 文件结构

```
scripts/
  download_reports.py   # 下载 NVIDIA 近三年 10-K (SEC EDGAR)
  chunk_reports.py      # 切块 -> data/chunks.jsonl (1545 块)
  build_qa_dataset.py   # 造数据: 提问/作答/质检/划分 -> 4 个 jsonl + 抽检记录
data/
  chunks.jsonl              # 1545 个文本块(知识库)
  train.jsonl               # 62 条训练问答(带出处)
  test_t1.jsonl             # T1: 训练片段换问法 10 条
  test_t2.jsonl             # T2: 未见片段原问法 10 条
  test_t2_rephrased.jsonl   # T2: 未见片段换问法 10 条
  qa_audit.md               # 质检抽检记录(有据/小出入/编造判定)
notebook/
  nvidia_dongmi_colab.ipynb # Colab 训练笔记本(T4 GPU, 开箱即跑)
```

## 复现步骤

1. `python scripts/download_reports.py && python scripts/chunk_reports.py && python scripts/build_qa_dataset.py`
2. 打开 `notebook/nvidia_dongmi_colab.ipynb` → Colab T4 GPU → 把仓库地址填进第一个单元格 → 全部运行
3. 笔记本自动输出：微调前/后四格对照表（CSV）+ RAG 对照（CSV）+ LoRA 适配器

## 安全说明

- API key 只放 `.env`（已写进 `.gitignore`），仓库和截图不出现任何密钥、密码、个人账号。
- 本作业造数据阶段使用的是课件同款「便宜模型 + 提示词分工」思路；问答内容由大模型生成并经人工抽检（见 `data/qa_audit.md`）。
