# 方向 A · 英伟达财报向量知识库（RAG 问答）

《学会使用大模型》第三课作业，方向 A：把知识放在检索库里（课上"半年报的切块"那一套做法）。

复用方向 B 已备好的材料：NVIDIA FY2024–FY2026 三份 10-K，切块 1545 块（每块带公司/财年/章节，见 `../nvidia-dongmi/scripts/chunk_reports.py`）。

## 流水线

```
chunks.jsonl (1545块) ──多语言embedding──> index.npy 本地向量库
                                              │
问题 ──向量化──> 余弦检索 top-6 ──塞进提示词──> 大模型(glm-4.7-flash) ──> 带出处回答
                                              │
                        四格对照实验: 有库vs无库 / 正常vs扭曲问法 / 库内vs库外 / 温度对比
```

## 文件结构

```
scripts/
  build_index.py    # 建向量索引 -> data/index.npy
  ask.py            # 检索问答 CLI (支持 --no-kb / --temperature / --k / --show-sources)
  run_fourgrid.py   # 四格对照实验, 输出 data/results/fourgrid_detail.csv
data/
  index.npy         # 1545 x 384 向量 (不进仓库, build_index.py 重建)
  results/          # 实验结果 CSV
.env.example        # API 配置模板(复制为 .env, key 不进仓库)
```

## 复现步骤

```bash
pip install sentence-transformers
python scripts/build_index.py          # 约3-5分钟(含模型下载)
cp .env.example .env                   # 填入你的大模型 API key
python scripts/ask.py "英伟达FY2026总收入是多少？" --show-sources
python scripts/run_fourgrid.py         # 四格实验, 约10分钟
```

## 四格实验设计（对应作业要求）

| 格子 | 对照 | 问题集 |
|---|---|---|
| 格1 | 有知识库 vs 无知识库 | 8 个库内问题 |
| 格2 | 正常问法 vs 扭曲问法（口语化+加时间限定+细节干扰，语义等价） | 8 对 |
| 格3 | 相关内容在库里 vs 不在库里（测编造） | 8 + 8 |
| 格4 | 默认温度 0.2 vs 高温度 1.0（每题 2 遍） | 8 题 × 2 × 2 |

## 安全说明

API key 只放 `.env`（已被 .gitignore 排除），截图与仓库不出现任何密钥。
