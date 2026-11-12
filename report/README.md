# 英文报告说明

本目录用于保存 ICML 格式的英文报告源文件和参考文献。

**每位成员提交的报告小节必须使用英文，最终课程报告也必须全部使用英文。**

建议报告结构：

1. Introduction
2. Dataset and Exploratory Data Analysis
3. Text Preprocessing and Feature Representation
4. Methods and Experimental Setup
5. Results and Error Analysis
6. Limitations and Impact Statement
7. Conclusion

正文必须遵守课程页数限制。报告中的每一个数字都必须能够追溯到 results/ 中的文件或可复现命令。

## 成员 2 · Dataset and Exploratory Data Analysis

从 `report/` 编译主文档时，使用 `\input{data_eda}` 引入小节。把 `data_eda.bib` 的两条文献合并到主文献库，或将该文件添加到主文档的 BibTeX 文件列表。运行 EDA 命令后，正文所需的长度分布图位于 `../results/figures/eda_review_lengths.png`。

本小节的数字对应数据字典中记录的公开 TSV 快照。若全组采用其他文件，先用最终文件重跑 EDA，再按输出更新正文、表格与引用。
