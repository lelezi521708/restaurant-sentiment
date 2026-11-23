# 成员 4：Linear SVM 主模型交接

## 完成内容

本模块严格复用成员 2 的固定 796/200 划分和成员 3 选定的 unigram TF-IDF。训练集内使用五折分层交叉验证，仅比较 `C = 0.1, 1, 10`。每一折都从原始评论文本重新拟合完整的 TF-IDF + Linear SVM pipeline，测试集不参与参数选择。

训练集交叉验证按正类 F1 选择 `C=0.1`：

| C | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.8104 | 0.8224 | 0.7922 | **0.8060** | 0.8850 |
| 1.0 | 0.8015 | 0.8054 | 0.7996 | 0.8016 | **0.8953** |
| 10.0 | 0.8015 | 0.8051 | 0.8021 | 0.8022 | 0.8915 |

锁定 `C=0.1` 后，对 200 条固定测试集只评估一次：Accuracy 0.7400、Precision 0.7791、Recall 0.6700、F1 0.7204、ROC-AUC 0.8061。较低的测试 Recall 表明模型漏判正面评论较多，后续统一错误分析应重点检查表达含蓄、带否定结构或正负信息混合的评论。

## 运行方式

```bash
python -m restaurant_sentiment.svm_experiment
python -m pytest -q
python -m nbconvert --to notebook --execute notebooks/03_svm.ipynb \
  --output 03_svm.ipynb --output-dir notebooks
```

运行前必须已经生成：

```text
data/processed/train.csv
data/processed/test.csv
data/processed/split_manifest.csv
```

## 交付文件

- 实验代码：`src/restaurant_sentiment/svm_experiment.py`
- Notebook：`notebooks/03_svm.ipynb`
- 测试：`tests/test_svm_experiment.py`
- CV 表：`results/tables/svm_cv_results.csv`
- 测试指标：`results/tables/svm_test_metrics.csv`
- 可追踪预测：`results/tables/svm_test_predictions.csv`（不含评论原文）
- 关键词权重：`results/tables/svm_top_terms.csv`
- 元数据与输入哈希：`results/tables/svm_experiment.json`
- 图：`results/figures/svm_c_cv_f1.png`、`svm_test_confusion_matrix.png`
- 英文报告：`report/svm.tex`、`report/svm.bib`
- 本地模型：`models/svm/linear_svm_pipeline.joblib`（不上传）

## 给成员 5 的参数交接

统一最终评估应将 SVM 参数从临时默认值更新为：

```text
svm_c = 0.1
```

成员 5 可运行：

```bash
python -m restaurant_sentiment.final_evaluation --svm-c 0.1
```

最终报告应以成员 5 在合并分支后重新生成的统一比较表为准，避免同时引用 `C=1` 的旧结果。
