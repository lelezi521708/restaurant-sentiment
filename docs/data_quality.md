# 数据质量报告 · 成员 2

本次运行使用 [数据字典](../data/data_dictionary.md) 中记录的公开 TSV 快照。所有计数可由 `results/tables/data_quality.json` 和配套 CSV 复核。

## 质量检查与处理

| 检查项 | 涉及记录数 | 处理 |
|---|---:|---|
| 原始记录 | 1,000 | 负面 500 条，正面 500 条 |
| 缺失评论、纯空白评论、缺失标签 | 各 0 | 对应记录会被移除 |
| 非 ASCII 文本 | 4 | 保留 Unicode 字符 |
| 控制字符、嵌入 BOM 或零宽空格 | 0 | 将控制字符和零宽空格转为空格，去掉嵌入 BOM |
| Unicode 替换字符 | 0 | 单独计数，留待核查 |
| 标签冲突的同文组 | 0 | 若出现，移除该组的全部记录 |
| 重复评论中标签一致的多余记录 | 4 | 保留每组第一条，移除 3 条负面和 1 条正面 |
| 清洗后记录 | 996 | 负面 497 条，正面 499 条 |

清洗先进行 Unicode NFC 和空白规范化，再处理标签与重复，保留原文本的大小写、标点和否定表达。审计表的诊断项可能重叠，各项 `removed_*` 是互斥的实际删除计数。

## 统一划分

| 样本 | 负面 | 正面 | 合计 |
|---|---:|---:|---:|
| Train | 397 | 399 | 796 |
| Test | 100 | 100 | 200 |

分层比例为 80/20，种子为 `8017`。两个集合的规范化文本交集为 0 条，忽略大小写后交集仍为 0 条。按行划分无法控制同一餐厅或作者的样本分布，因为源文件没有这两类标识。

## EDA 摘要

负面评论长度中位数为 10 词（四分位区间 6–16），正面为 9 词（6–14），两类长度分布重叠。训练集中，`not` 出现在 20.4% 的负面评论中，`good` 出现在 14.8% 的正面评论及 4.5% 的负面评论中。词频按包含该词的同类训练评论占比计算。图表和英文图注见 [英文报告小节](../report/data_eda.tex) 及下表。

| 文件 | English caption |
|---|---|
| `results/figures/eda_class_distribution.png` | Class counts in the cleaned dataset and its stratified partitions. |
| `results/figures/eda_review_lengths.png` | Word counts by sentiment in the cleaned corpus. Dashed lines mark class medians. |
| `results/figures/eda_training_terms.png` | Share of training reviews in each class containing each displayed term. |
