# 数据字典

| 字段 | 原始类型 | 输出类型 | 含义 |
|---|---|---|---|
| `Review` | UTF-8 文本 | 字符串 | 餐厅评论文本 |
| `Liked` | 0 或 1 | 整数 | 0 = 负面，1 = 正面 |

## 本次分析的数据版本

- 实际使用的输入为 `Restaurant_Reviews.tsv`，含表头，分隔符为制表符，下载于 2026 年 11 月 12 日。下载页标注 Apache-2.0。[固定版本文件](https://huggingface.co/datasets/KarthikaRajagopal/Restaurant_Reviews.tsv/blob/ef6e2c7abbdf201ee732a2cb90ac394de6f5bb00/Restaurant_Reviews.tsv)
- 原计划的 Kaggle 竞赛页面当前列出 8 个 CSV，下载需要登录并接受竞赛规则。本次分析使用上述公开 TSV。[原计划 Kaggle 入口](https://www.kaggle.com/competitions/restaurant-reviews/data)
- 与 UCI 原始语料的 `yelp_labelled.txt` 按行对齐，去掉首尾空白后有 995 条文本及标签一致，其余 5 条的文本有差异。两份文件应各自记录版本。[UCI 数据集及许可](https://doi.org/10.24432/C57604)
- TSV 的 SHA-256 为 `2b432f36bc7c7ed77312187a9afd6531dfa3d16be48b9738131dd5141195b469`。

## 检查结果

实际输入 1,000 条，负面和正面各 500 条。移除 4 条同标签重复记录后保留 996 条。固定分层划分（随机种子 8017）得到训练集 796 条、测试集 200 条。详情见[数据质量报告](../docs/data_quality.md)。

## 本地输出

`cleaned_reviews.csv`、`train.csv` 和 `test.csv` 的列为 `review_id,Review,Liked`。`review_id` 是从 1 开始的清洗后行号，用于关联同一次运行的输出。`split_manifest.csv` 保存每个 `review_id` 对应的 `Liked` 和 `split`。

请将原始和处理后的评论文件保留在本地 `data/raw/` 和 `data/processed/`。
