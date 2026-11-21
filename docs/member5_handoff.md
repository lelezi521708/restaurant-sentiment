# 成员 5 提交前检查与交接

本交付提交至 `member-5/nb-evaluation` 分支。负责模块：Multinomial Naive Bayes、统一最终评估、错误分析和英文报告小节。

## 交付清单

- NB 平滑参数网格与训练折内调参：`src/restaurant_sentiment/models.py`、`src/restaurant_sentiment/final_evaluation.py`。
- 评估与解释工具：`src/restaurant_sentiment/evaluation.py`（ROC、NB 对数几率关键词、错误分型、汇总表和英文图）。
- 可从头运行的 notebook：`notebooks/04_naive_bayes.ipynb`、`notebooks/05_evaluation_error_analysis.ipynb`。
- 自动测试：`tests/test_evaluation.py`；全仓库共 22 项测试。
- 中文说明：本文件、README 更新。
- 英文报告：`report/nb_evaluation.tex`（Method、Results、Error Analysis、Discussion 四节）。
- 汇总结果表和图：见下方“提交到仓库的 results 文件”。
- 仅本地：`data/processed/final_errors_*.csv`、`data/processed/final_predictions.csv`、`models/nb/final_nb_pipeline.joblib`。

## 输入与前置条件

1. 按 README 下载固定 TSV 并校验 SHA-256；
2. 运行 `python -m restaurant_sentiment.eda --data data/raw/Restaurant_Reviews.tsv`；
3. 运行 `python -m restaurant_sentiment.preprocessing`；
4. 确认 `data/processed/train.csv`（796 条）、`test.csv`（200 条）和 `split_manifest.csv` 存在。

最终评估通过 `load_fixed_partitions()` 再次校验 review_id、文本重叠和 manifest 一致性，不重新清洗或划分数据。

## 运行命令

```bash
python -m restaurant_sentiment.final_evaluation
python -m pytest -q
```

如果成员 4 在训练集内选出不同的 SVM `C`，必须重新生成全部结果：

```bash
python -m restaurant_sentiment.final_evaluation --svm-c <成员4选出的C>
```

notebook 从头执行：

```bash
python -m nbconvert --to notebook --execute notebooks/04_naive_bayes.ipynb \
  --output 04_naive_bayes.ipynb --output-dir notebooks
python -m nbconvert --to notebook --execute notebooks/05_evaluation_error_analysis.ipynb \
  --output 05_evaluation_error_analysis.ipynb --output-dir notebooks
```

本分支在 Python 3.12.14、`scikit-learn 1.9.1`、`pandas 2.3.3` 下完成 22 项测试，两个 notebook 所有代码单元格执行成功。若本机 pytest 临时目录受沙箱限制，可加 `--basetemp=/tmp/stat8017-pytest-base`。

## 参数与选择规则

- 随机种子统一 `8017`；训练/测试为 796/200 分层划分，测试集 100 条正类、100 条负类。
- 共享特征为成员 3 锁定的 unigram TF-IDF：`sublinear_tf=True`、`smooth_idf=True`、L2 归一化、不删停用词。
- NB 网格 `alpha=[0.1, 0.5, 1.0, 2.0]`；五折分层交叉验证，每折重新拟合完整 TF-IDF + NB pipeline，以平均正类 F1 选参。
- 训练折结果：`alpha=0.5` 的平均 F1 为 0.8052，选为最终值。与成员 3 的 `alpha=1` 探针结果 0.7962 一致。
- Linear SVM 当前使用共享默认 `C=1`。**成员 4 的最终选择若不同，必须改参数重跑，不能用测试集选 C。**
- 测试集只在所有参数锁定后使用一次；没有在测试集上做特征、参数或阈值选择。

## 最终结果（C=1 版本）

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy (majority) | 0.500 | 0.500 | 1.000 | 0.667 | 0.500 |
| Multinomial NB (alpha=0.5) | 0.750 | 0.766 | 0.720 | 0.742 | 0.837 |
| Linear SVM (C=1) | 0.755 | 0.793 | 0.690 | 0.738 | 0.842 |

NB 的 F1 略高，SVM 的 accuracy、precision 和 ROC-AUC 略高；差距很小，不作显著性结论。混淆矩阵计数：NB 为 TN=78、FP=22、FN=28、TP=72；SVM 为 TN=82、FP=18、FN=31、TP=69。

错误分析为描述性启发式，不代表因果：NB 50 个错误中 22 个 FP、28 个 FN、2 个高置信度错误；16 个为短评论，10 个带否定词，2 个带转折词。完整明细和评论原文只保留在本地 `data/processed/`，不提交到 GitHub。

## 提交到仓库的 results 文件

```text
results/tables/final_nb_cv.csv
results/tables/final_model_comparison.csv
results/tables/final_nb_top_terms.csv
results/tables/final_svm_top_terms.csv
results/tables/final_error_summary.csv
results/tables/final_evaluation.json
results/figures/final_nb_alpha_cv.png
results/figures/final_model_comparison.png
results/figures/final_roc_curves.png
results/figures/final_confusion_matrix_dummy.png
results/figures/final_confusion_matrix_nb.png
results/figures/final_confusion_matrix_svm.png
results/figures/final_error_categories.png
```

以上文件和两个 notebook 均不含原始评论文本。元数据 `final_evaluation.json` 记录输入 SHA-256、依赖版本、参数来源和测试集使用状态。

## 成员 1 整合提示

- 英文报告用 `\input{nb_evaluation}` 引入四节内容；`nb_evaluation.tex` 引用的 `pedregosa2011scikit`、`manning2008ir` 和 `rajagopal-restaurant-reviews` 已存在于成员 2/3 的 bib 文件，无需新增重复条目。
- 主报告按需挑选图和表：建议保留 `tab:final-comparison`、`fig:final-roc`、`tab:error-summary`，其余作为补充。
- 若成员 4 改动 SVM `C`，请让成员 5 先重跑评估，再统一替换报告数字；不要手工编辑 CSV 或 LaTeX。
- 不要把 `data/processed/`、`models/`、`results/` 的其他生成文件强制加入 Git。
