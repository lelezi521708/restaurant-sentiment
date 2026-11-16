# 成员 3 文本预处理与 TF-IDF 交接

## 完成范围

完成共享 TF-IDF 设置、训练集内 unigram/bigram 对比、稀疏特征导出、特征总结、可从头运行的 notebook、自动测试与英文报告小节。成员 4 的 SVM C 参数调优和成员 5 的最终测试评估不在本次范围内。

数据沿用成员 2 的固定版本。原始 TSV 的 SHA-256 为 `2b432f36bc7c7ed77312187a9afd6531dfa3d16be48b9738131dd5141195b469`；清洗后 996 条，训练 796 条，测试 200 条。运行器直接读取已生成的划分及清单，不重新清洗或随机划分，并检查 ID、文本重叠及清单一致性。

## 运行

在仓库根目录、已激活虚拟环境且安装项目后：

```bash
python -m restaurant_sentiment.eda --data data/raw/Restaurant_Reviews.tsv
python -m restaurant_sentiment.preprocessing
python -m pytest -q
```

也可以在 Jupyter 中从头运行 `notebooks/02_preprocessing.ipynb`。该 notebook 调用同一个实验函数，不维护另一份实验逻辑。Python 3.11 及以上可用；本次实际验证环境为 Python 3.12.14，精确核心依赖见 `requirements-member3.txt` 和实验元数据。

## 预处理设置

保留成员 2 已清洗的文本，在向量化阶段转小写并执行 Unicode 去音调（例如 café 转为 cafe）。使用 scikit-learn 的默认词级模式 `(?u)\b\w\w+\b`，单字符词不进入词汇表，数字词不另行删除。标点作为词边界，不作为独立特征。

不移除停用词，保留 `not`、`no`、`never` 等否定词；不做词干化或词形还原，避免额外依赖和语义合并。默认分词不完整保留缩写否定：例如 `isn't` 分成 `isn` 和被丢弃的 `t`。这是真实限制，不能把保留否定词等同于理解所有否定表达。

TF 使用 `1 + ln(count)`；平滑 IDF 使用 `ln((1 + n)/(1 + df)) + 1`，之后按文档 L2 归一化。固定 `min_df=1`、`max_df=1.0`、`max_features=None`、`sublinear_tf=True`、`smooth_idf=True`、`use_idf=True`。TF-IDF 非负，可同时供 NB 与 SVM 使用。

## 训练集内特征实验

使用完全相同的 `StratifiedKFold(5, shuffle=True, random_state=8017)`。每折重新拟合完整 TF-IDF 与分类器 pipeline，防止验证折词汇或 IDF 泄漏。Linear SVM 固定 `C=1`；NB 固定 `alpha=1`。选择规则在实验前固定：按 SVM 的平均正类 F1 选特征，同分选 unigram。NB 仅作敏感性检查。

| Features | Probe | Accuracy | Precision | Recall | F1 mean ± SD | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Unigram | Linear SVM | 0.8015 | 0.8054 | 0.7996 | 0.8016 ± 0.0282 | 0.8953 |
| Unigram + bigram | Linear SVM | 0.8041 | 0.8177 | 0.7846 | 0.7998 ± 0.0388 | 0.8972 |
| Unigram | Multinomial NB | 0.7965 | 0.8008 | 0.7972 | 0.7962 ± 0.0484 | 0.8802 |
| Unigram + bigram | Multinomial NB | 0.7852 | 0.7893 | 0.7821 | 0.7845 ± 0.0394 | 0.8809 |

以上全部为验证折结果，不是测试集指标。SD 为五个折得分的样本标准差，不能当作置信区间或显著性检验。SVM bigram 的 Accuracy 和 AUC 略高，但主选择指标 F1 略低。

推荐统一采用 unigram（1781 维），而不是 unigram + bigram（7345 维）。bigram 令特征维度增至约 4.12 倍，未改善本次主指标；F1 差仅约 0.0018，远小于折间波动，不能宣称 unigram 普遍更优。小数据集中的稀疏短语可能限制 bigram 收益。`config.DEFAULT_NGRAM_RANGE=(1,1)` 是共享默认值；两种表示的实验结果均保留。

| Features | Partition | Shape | Nonzero | Density | Zero rows |
|---|---|---|---:|---:|---:|
| Unigram | Train | 796 × 1781 | 7836 | 0.5527% | 0 |
| Unigram | Test | 200 × 1781 | 1653 | 0.4641% | 0 |
| Unigram + bigram | Train | 796 × 7345 | 15269 | 0.2612% | 0 |
| Unigram + bigram | Test | 200 × 7345 | 2342 | 0.1594% | 0 |

测试集的这些数字仅检查转换后的矩阵结构，不使用测试标签评价模型。最终完整训练集向量化器只调用 `fit_transform(train.Review)`，测试集只调用 `transform(test.Review)`。

## 输出与下游用法

`results/tables/tfidf_cv_comparison.csv` 保存所有五项指标的均值和标准差；`tfidf_cv_folds.csv` 保存逐折指标；`tfidf_cv_assignments.csv` 保存每条训练评论的验证折编号。`tfidf_feature_summary.csv` 保存矩阵结构，`tfidf_top_training_terms.csv` 保存按训练集平均 TF-IDF 排序的前 30 项。后者是频率与表示摘要，不是情感重要性或因果解释。

英文对比图位于 `results/figures/tfidf_ngram_cv_f1.png`，误差棒表示折间 SD。`results/tables/tfidf_experiment.json` 记录完整向量化参数、核心依赖版本、固定输入文件 SHA-256 和选择规则。

本地 `data/processed/tfidf/` 保存两种表示的训练/测试 `.npz`、按特征列顺序排列的词汇表与 IDF，以及 `train_rows.csv`、`test_rows.csv`。矩阵行顺序与对应的行清单及成员 2 的 CSV 完全一致。`models/tfidf/` 保存拟合后的 joblib 向量化器及 `feature_config.json`。

成员 4 和 5 在交叉验证时应从原始训练文本构造完整 pipeline：

```python
from restaurant_sentiment.pipeline import build_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(5, shuffle=True, random_state=8017)
search = GridSearchCV(build_pipeline("svm"),
                      {"classifier__C": [0.1, 1.0, 10.0]},
                      scoring="f1", cv=cv)
# train is the unmodified data/processed/train.csv DataFrame
search.fit(train["Review"], train["Liked"])
```

此处是供成员 4 使用的接口示例，不是本次已经完成的调参实验。仓库原 `tune_svm` 仍包含 n-gram 联合搜索；若要遵守本次锁定的表示，应使用上面只搜索 C 的网格。需要联合搜索时，必须仅在训练折内进行并重新记录选择。

最终训练可直接加载已拟合特征：

```python
from scipy.sparse import load_npz
import pandas as pd

x_train = load_npz("data/processed/tfidf/unigram_train.npz")
y_train = pd.read_csv("data/processed/tfidf/train_rows.csv")["Liked"]
```

不要对这些在全部训练集上拟合过的矩阵直接执行内部交叉验证。joblib 仅加载本机生成或可信来源的文件，且需要匹配依赖版本。

## 英文报告与提交范围

`report/text_features.tex` 可通过 `\input{text_features}` 引入主报告；将 `text_features.bib` 合并至主参考文献库。正文约一个小节，包含训练集内对比结果和限制，不冒充最终测试结论。

原始文本、处理后的评论、矩阵、词汇表、拟合模型和虚拟环境仅保留本地。遵守仓库规则，不强制添加数据。代码、notebook、中文说明、英文小节及少量报告所需汇总表/图可以由组内确认后提交。完整文件清单和提交前检查见 `docs/member3_submission.md`。
