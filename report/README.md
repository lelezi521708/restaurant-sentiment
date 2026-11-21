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

## 成员 5 Naive Bayes、Results、Error Analysis 和 Discussion

使用 `\input{nb_evaluation}` 引入 `report/nb_evaluation.tex`。该小节包含四个英文 section，数字由 `python -m restaurant_sentiment.final_evaluation` 生成，对应 `results/tables/final_model_comparison.csv`、`final_nb_cv.csv`、`final_error_summary.csv` 和 `results/figures/final_roc_curves.png`。当前 SVM 使用共享默认 `C=1`；若成员 4 在训练集内选出不同的 `C`，必须先以 `--svm-c` 重跑评估再替换数字，不要用测试集调参。

`nb_evaluation.tex` 引用的 `pedregosa2011scikit`、`manning2008ir` 和 `rajagopal-restaurant-reviews` 已分别出现在成员 2/3 的 bib 文件中，主报告合并时不要重复添加相同 key。主报告建议保留 `tab:final-comparison`、`fig:final-roc`、`tab:error-summary`；其余表和图作为补充。报告不包含原始评论文本，错误案例仅以本地语料中的 review ID 指代。
