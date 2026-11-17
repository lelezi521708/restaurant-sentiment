# 成员 3 提交前检查与交接

本交付已提交至 `member-3/tfidf-features` 分支，等待组内审阅和合并。以下清单同时作为后续成员的复现与交接记录。

## 交付清单

- 共享参数：`src/restaurant_sentiment/config.py`、`features.py`、`pipeline.py`。
- 实验入口：`src/restaurant_sentiment/preprocessing.py`。
- 可从头运行的 notebook：`notebooks/02_preprocessing.ipynb`。
- 特征与数据隔离测试：`tests/test_features.py`；整个测试集共 14 项。
- 中文说明：`docs/member3_tfidf.md`、本文件、仓库和报告目录 README。
- 英文报告：`report/text_features.tex`、`report/text_features.bib`。
- 已验证核心依赖：`requirements-member3.txt`。
- 已提交的汇总结果：`tfidf_cv_comparison.csv`、`tfidf_feature_summary.csv`、`tfidf_experiment.json` 和 `tfidf_ngram_cv_f1.png`。
- 其余逐折结果仅在运行实验后保留本地。
- 仅本地交接：`data/processed/tfidf/`、`models/tfidf/`。

## 验证

从仓库根目录、已激活环境后运行：

```bash
python -m restaurant_sentiment.preprocessing
python -m pytest -q
python -m nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb \
  --output 02_preprocessing.ipynb --output-dir notebooks
```

本次在 Python 3.12.14 验证，测试全部通过，notebook 所有代码单元格运行成功。`results/tables/tfidf_experiment.json` 保存实际运行版本和输入哈希。原已提交的成员 2 数据审计文件保留其原环境记录，不改写为成员 3 的环境。

检查 `selected_variant` 为 `unigram`，`test_metrics_computed` 为 `false`。五折验证 F1 为 unigram SVM 0.8016、unigram + bigram SVM 0.7998，差距很小，不作显著性结论。

## 提交范围

先检查差异，避免把其他成员或后续编辑混入本次提交。以下命令只暂存明确列出的源文件，不会自动 commit 或 push：

```bash
git diff --check
git diff
git add README.md requirements-member3.txt \
  src/restaurant_sentiment/config.py \
  src/restaurant_sentiment/features.py \
  src/restaurant_sentiment/pipeline.py \
  src/restaurant_sentiment/preprocessing.py \
  tests/test_features.py notebooks/02_preprocessing.ipynb \
  docs/member3_tfidf.md docs/member3_submission.md \
  report/README.md report/text_features.tex report/text_features.bib
```

`results/` 默认被忽略。本分支仅明确提交了以下不含原始评论的报告汇总结果，没有强制添加整个结果目录：

```bash
git add -f results/tables/tfidf_cv_comparison.csv \
  results/tables/tfidf_feature_summary.csv \
  results/tables/tfidf_experiment.json \
  results/figures/tfidf_ngram_cv_f1.png
git diff --cached --stat
```

不要上传原始 TSV、完整评论 CSV、词汇表、稀疏矩阵、joblib 模型、虚拟环境或桌面说明文档。不要使用 `git add -f data/`、`git add -f models/` 或整个目录的强制暂存。

## Pull Request 说明草稿

标题建议：`Add Member 3 training-only TF-IDF comparison and feature handoff`

本次完成共享文本预处理、unigram 与 unigram + bigram 的训练集内五折比较、稀疏特征及向量化器导出、notebook、测试和英文报告小节。沿用成员 2 的 796/200 固定划分，每折内部重新拟合 TF-IDF。按固定 SVM C=1 的平均正类 F1 推荐 unigram，保留两种表示的完整结果，不计算最终测试集模型指标。

验证：14 项测试通过；notebook 从头运行成功；矩阵行清单与固定划分顺序一致；输入哈希及依赖版本已记录。

## 成员 4 和 5 接续

成员 4 使用统一 unigram 设置，从训练文本构建完整 pipeline 后调优 SVM C；不要对完整训练集拟合后的矩阵进行内部交叉验证。原 `tune_svm` 仍包含 n-gram 联合搜索，锁定特征时请采用交接文档中的 C-only 网格。

成员 5 使用相同特征、划分与评估规则，完成 NB 和统一最终评估。200 条测试评论留到模型与设置确定后使用。成员 1 将英文小节和参考文献合并进 ICML 主报告，并核对全篇 8 页限制。本次未代做这些后续任务。
