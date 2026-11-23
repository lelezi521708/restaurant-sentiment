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

## 成员 3 Text Preprocessing and Feature Representation

使用 `\input{text_features}` 引入 `text_features.tex`，把 `text_features.bib` 加入主文献库。先运行 `python -m restaurant_sentiment.preprocessing`，对应数值见 `results/tables/tfidf_cv_comparison.csv` 和 `tfidf_feature_summary.csv`。本节仅报告训练集内五折特征对比，不是最终测试集评估。

英文图 `results/figures/tfidf_ngram_cv_f1.png` 为补充交接图；小节已经含有紧凑表格，主报告可按 8 页限制决定是否再插入该图。

## 成员 4 Linear SVM

使用 `\input{svm}` 引入 `svm.tex`，并把 `svm.bib` 合并到主文献库。先运行 `python -m restaurant_sentiment.svm_experiment`，表中数值可追溯到 `results/tables/svm_cv_results.csv` 和 `svm_test_metrics.csv`。统一模型比较合并后，应以成员 5 使用 `--svm-c 0.1` 重跑的最终结果为准。
