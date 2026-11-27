# Member 1 Reproduction and Integration Record

## English

### Scope

Member 1 independently reproduced the project from the fixed raw dataset on
2026-11-27 (Asia/Hong_Kong). The review started from remote `main` commit
`540cfea` and ran the work of Members 2--5 in dependency order. No raw review
text, processed dataset, fitted model, secret, or personal path is included in
this record.

### Verification environment

- macOS (Apple Silicon)
- Python 3.11.18
- NumPy 2.4.6
- pandas 2.3.3
- SciPy 1.17.1
- scikit-learn 1.9.1
- joblib 1.6.0
- Dataset SHA-256:
  `2b432f36bc7c7ed77312187a9afd6531dfa3d16be48b9738131dd5141195b469`

### Commands

From the repository root, after installing `requirements.txt` and the editable
package:

```bash
python scripts/reproduce_project.py
```

The script verifies the dataset hash, runs data preparation, feature selection,
SVM tuning, NB tuning, unified final evaluation, and `pytest`. It writes the
machine-readable manifest to
`results/tables/member1_reproduction.json`.

The five notebooks were also executed from first cell to last cell with the
project Python 3.11 kernel:

```bash
python -m ipykernel install --user \
  --name restaurant-sentiment \
  --display-name "Restaurant Sentiment (Python 3.11)"

jupyter nbconvert --to notebook --execute notebooks/01_data_eda.ipynb \
  --ExecutePreprocessor.kernel_name=restaurant-sentiment
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb \
  --ExecutePreprocessor.kernel_name=restaurant-sentiment
jupyter nbconvert --to notebook --execute notebooks/03_svm.ipynb \
  --ExecutePreprocessor.kernel_name=restaurant-sentiment
jupyter nbconvert --to notebook --execute notebooks/04_naive_bayes.ipynb \
  --ExecutePreprocessor.kernel_name=restaurant-sentiment
jupyter nbconvert --to notebook --execute notebooks/05_evaluation_error_analysis.ipynb \
  --ExecutePreprocessor.kernel_name=restaurant-sentiment
```

### Reproduced outcomes

- Input reviews: 1,000
- Removed same-label duplicate rows: 4
- Retained reviews: 996
- Fixed split: 796 training / 200 test
- Exact and case-folded train/test review overlap: 0
- Selected feature representation: word unigram TF-IDF
- Selected Linear SVM parameter: `C=0.1` (training-fold F1 0.8060)
- Selected Multinomial NB parameter: `alpha=0.5` (training-fold F1 0.8052)
- Automated tests: 26 passed
- Notebooks: 5/5 executed successfully with the project kernel

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy majority baseline | 0.500 | 0.500 | 1.000 | 0.667 | 0.500 |
| Multinomial NB (`alpha=0.5`) | 0.750 | 0.766 | 0.720 | 0.742 | 0.837 |
| Linear SVM (`C=0.1`) | 0.740 | 0.779 | 0.670 | 0.720 | 0.806 |

The regenerated CSV values matched the committed results except for
machine-level floating-point rounding at approximately `1e-16`, dependency
version metadata, runtime measurements, and PNG encoding differences. Model
selection, predictions, confusion matrices, and report-level metrics were
unchanged.

### Integration corrections

The review identified stale SVM error-analysis numbers in
`report/nb_evaluation.tex`. The report was corrected to match the unified final
output: 52 errors, 19 false positives, 33 false negatives, and a 26.0% error
rate. The Jupyter kernel instructions, final README, and unified report entry
point were also added during integration.

## 中文

### 复现范围

成员 1 于 2026-11-27（香港时间）使用固定原始数据独立复现了完整项目。检查从远程
`main` 的 `540cfea` 开始，按照成员 2 到成员 5 的依赖顺序重新运行。本文档不包含原始
评论、处理后的完整数据、拟合模型、密钥或个人绝对路径。

### 验证环境

- macOS（Apple Silicon）
- Python 3.11.18
- NumPy 2.4.6
- pandas 2.3.3
- SciPy 1.17.1
- scikit-learn 1.9.1
- joblib 1.6.0
- 数据集 SHA-256：
  `2b432f36bc7c7ed77312187a9afd6531dfa3d16be48b9738131dd5141195b469`

### 复现命令

在仓库根目录安装依赖和项目包后运行：

```bash
python scripts/reproduce_project.py
```

该脚本会检查数据哈希，并依次运行数据处理、特征选择、SVM 调参、NB 调参、统一最终
评估和 `pytest`。机器可读的复现清单保存于
`results/tables/member1_reproduction.json`。

5 个 Notebook 也使用项目 Python 3.11 kernel 从第一格运行到最后一格。具体 kernel
安装与批量执行命令见上方英文部分。

### 复现结果

- 原始评论：1,000 条
- 删除同标签重复记录：4 条
- 最终保留：996 条
- 固定划分：796 条训练数据 / 200 条测试数据
- 训练集和测试集评论重叠：0
- 最终特征：word unigram TF-IDF
- Linear SVM：`C=0.1`，训练折 F1 为 0.8060
- Multinomial NB：`alpha=0.5`，训练折 F1 为 0.8052
- 自动测试：26 项全部通过
- Notebook：5/5 使用项目 kernel 完整运行成功

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy 多数类基线 | 0.500 | 0.500 | 1.000 | 0.667 | 0.500 |
| Multinomial NB（`alpha=0.5`） | 0.750 | 0.766 | 0.720 | 0.742 | 0.837 |
| Linear SVM（`C=0.1`） | 0.740 | 0.779 | 0.670 | 0.720 | 0.806 |

重新生成的结果与仓库已提交结果一致。差异仅包括约 `1e-16` 的浮点舍入、依赖版本记录、
运行时间和 PNG 编码；模型选择、预测、混淆矩阵及报告所用指标均未改变。

### 整合修正

复现过程中发现 `report/nb_evaluation.tex` 的 SVM 错误分析仍使用旧数字。现已按照统一
最终结果修正为：52 个错误、19 个假阳性、33 个假阴性、错误率 26.0%。本次整合还补充
了 Jupyter kernel 说明、最终双语 README 和统一报告入口。
