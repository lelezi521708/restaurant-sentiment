# Restaurant Review Sentiment Classification

STAT8017 Data Mining Techniques group project

> English documentation appears first. The complete Chinese version follows.
> 英文说明在前，完整中文说明在后。

## English

### Project overview

This project builds a reproducible binary sentiment classifier for short
restaurant reviews (`0` negative, `1` positive). A majority-class baseline,
Multinomial Naive Bayes (NB), and Linear SVM use the same fixed split and
TF-IDF representation.

Research questions:

1. Does Linear SVM outperform NB on the fixed test set?
2. Does unigram + bigram TF-IDF improve over unigram TF-IDF?
3. Which terms are most associated with positive and negative predictions?
4. What types of reviews are most frequently misclassified?

```text
Fixed public dataset
        ↓
Validation, deduplication, EDA, and one fixed split
        ↓
Training-only TF-IDF representation comparison
        ↓
Training-fold SVM and NB tuning
        ↓
Unified evaluation on the locked test set
        ↓
Interpretation, error analysis, and English report
```

### Final results

The fixed dataset contains 1,000 labelled reviews. Four redundant same-label
rows are removed, leaving 996 reviews and one stratified split of 796 training
and 200 test reviews. Feature and parameter selection use training folds only.

Selected settings:

- word-unigram TF-IDF with lowercase, Unicode accent stripping, sublinear TF,
  smoothed IDF, and L2 normalization;
- Linear SVM `C=0.1`;
- Multinomial NB `alpha=0.5`;
- random seed `8017`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy majority baseline | 0.500 | 0.500 | 1.000 | 0.667 | 0.500 |
| Multinomial NB (`alpha=0.5`) | **0.750** | 0.766 | **0.720** | **0.742** | **0.837** |
| Linear SVM (`C=0.1`) | 0.740 | **0.779** | 0.670 | 0.720 | 0.806 |

NB has higher accuracy, recall, F1, and ROC-AUC on this fixed sample, while SVM
has higher precision. With only 200 test reviews, this is not evidence of a
statistically significant or universal ranking.

### Dataset

The fixed `Restaurant_Reviews.tsv` contains `Review` text and binary `Liked`.

- [Download the fixed snapshot](https://huggingface.co/datasets/KarthikaRajagopal/Restaurant_Reviews.tsv/resolve/ef6e2c7abbdf201ee732a2cb90ac394de6f5bb00/Restaurant_Reviews.tsv?download=true)
- Required filename: `Restaurant_Reviews.tsv`
- SHA-256: `2b432f36bc7c7ed77312187a9afd6531dfa3d16be48b9738131dd5141195b469`
- Local path: `data/raw/Restaurant_Reviews.tsv`

The eight CSV files from the original Kaggle competition plan are not used.
Raw and processed review text is excluded from Git. See
[`data/data_dictionary.md`](data/data_dictionary.md) and
[`docs/data_quality.md`](docs/data_quality.md).

### Setup

Python 3.11 is recommended.

```bash
git clone https://github.com/lelezi521708/restaurant-sentiment.git
cd restaurant-sentiment
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify the data with `shasum -a 256 data/raw/Restaurant_Reviews.tsv`.

### Complete reproduction

Preferred command:

```bash
python scripts/reproduce_project.py
```

It verifies the dataset hash; runs Member 2 data preparation, Member 3 feature
selection, Member 4 SVM tuning, Member 5 NB tuning and final evaluation; runs
all tests; and writes `results/tables/member1_reproduction.json`.

Equivalent individual commands:

```bash
python -m restaurant_sentiment.eda --data data/raw/Restaurant_Reviews.tsv
python -m restaurant_sentiment.preprocessing
python -m restaurant_sentiment.svm_experiment
python -m restaurant_sentiment.final_evaluation --svm-c 0.1
python -m pytest -q
```

Expected checks: 796 training rows, 200 test rows, unigram TF-IDF, SVM
`C=0.1`, NB `alpha=0.5`, and 26 passing tests.

Member 1 independently repeated the full workflow. See
[`docs/member1_reproduction.md`](docs/member1_reproduction.md) and the
[machine-readable manifest](results/tables/member1_reproduction.json).

### Notebooks and Jupyter kernel

| Notebook | Owner and purpose |
|---|---|
| `01_data_eda.ipynb` | Member 2: quality checks, split, and EDA |
| `02_preprocessing.ipynb` | Member 3: TF-IDF and n-gram comparison |
| `03_svm.ipynb` | Member 4: Linear SVM tuning |
| `04_naive_bayes.ipynb` | Member 5: NB smoothing selection |
| `05_evaluation_error_analysis.ipynb` | Member 5: final evaluation and errors |

Register the project kernel so Jupyter does not silently use another Python:

```bash
python -m pip install ipykernel
python -m ipykernel install --user \
  --name restaurant-sentiment \
  --display-name "Restaurant Sentiment (Python 3.11)"
```

Select `Restaurant Sentiment (Python 3.11)` in JupyterLab or PyCharm.

### Main outputs

Key tables include `data_quality.json`, `tfidf_cv_comparison.csv`,
`svm_cv_results.csv`, `final_nb_cv.csv`, `final_model_comparison.csv`,
`final_error_summary.csv`, and `member1_reproduction.json` under
`results/tables/`.

Key figures include review-length EDA, TF-IDF n-gram comparison, SVM C
selection, NB alpha selection, final model comparison, ROC curves, confusion
matrices, and error categories under `results/figures/`.

### Member contributions

| Member | Responsibility | Main deliverables |
|---|---|---|
| Member 1 | Repository, integration, independent reproduction, final report | Bilingual README, reproduction evidence, consistency corrections, integrated LaTeX report |
| Member 2 | Data cleaning, EDA, fixed split | Data audit, `01_data_eda.ipynb`, EDA figures, Dataset/EDA section |
| Member 3 | Preprocessing and TF-IDF | Feature pipeline, n-gram comparison, `02_preprocessing.ipynb`, feature section |
| Member 4 | Linear SVM | C selection, `03_svm.ipynb`, predictions, figures, SVM section |
| Member 5 | NB and unified evaluation | NB tuning, `04`/`05` notebooks, model comparison, ROC/confusion matrices, errors, Results/Discussion |

Detailed handoffs are in `docs/member3_tfidf.md`, `docs/member4_svm.md`,
`docs/member5_handoff.md`, and `docs/member1_reproduction.md`.

### Experimental safeguards

- one split with `random_state=8017` and no train/test review overlap;
- the same partitions and metrics for all models;
- complete TF-IDF + classifier refitting inside every validation fold;
- no test-set use for n-gram, SVM C, or NB alpha selection;
- code-generated metrics with hashes and dependency metadata;
- no raw text, fitted models, credentials, or personal paths in Git.

### English report

The integrated ICML-format entry point is `report/main.tex`. Official ICML
2026 style files are included. Before submission, replace the five member and
email placeholders with actual names, student IDs, and contact details.

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Check the compiled main body against the course eight-page limit.

### Repository structure

```text
restaurant-sentiment/
├── README.md
├── CONTRIBUTING.md
├── data/                         # dictionary; raw/processed text stays local
├── docs/                         # handoffs and reproduction evidence
├── notebooks/                    # five notebooks
├── scripts/reproduce_project.py  # one-command independent reproduction
├── src/restaurant_sentiment/     # tested implementation
├── tests/                         # 26 automated tests
├── results/{figures,tables}/      # report-ready aggregate artifacts
└── report/                        # integrated English ICML LaTeX report
```

### Optional prediction demo

An interactive website is not required. The model comparison, ROC, confusion
matrices, and error plots already provide the academic visual evidence. For a
live presentation, one notebook cell predicting a user-entered review is
sufficient and should be described as a demo, not additional test evidence.

---

## 中文

### 项目简介

本项目建立一套可复现的餐厅评论二分类流程（`0` 负面，`1` 正面）。多数类基线、
Multinomial Naive Bayes（NB）和 Linear SVM 使用同一固定划分和 TF-IDF 表示。

研究问题：

1. Linear SVM 在固定测试集上是否优于 NB？
2. unigram + bigram 是否优于 unigram？
3. 哪些词语与正面或负面预测关系最强？
4. 模型最容易错误判断哪些类型的评论？

```text
固定公开数据集
        ↓
数据检查、去重、EDA 和一次性固定划分
        ↓
只使用训练集比较 TF-IDF 表示
        ↓
训练折内调节 SVM 和 NB
        ↓
在同一固定测试集上统一评价
        ↓
关键词、错误分析和英文报告
```

### 最终实验结果

原始数据包含 1,000 条评论，删除 4 条同标签重复记录后保留 996 条，固定分层划分为
796 条训练数据和 200 条测试数据。特征和参数选择只使用训练折。

最终设置：word unigram TF-IDF、SVM `C=0.1`、NB `alpha=0.5`、随机种子
`8017`。

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy 多数类基线 | 0.500 | 0.500 | 1.000 | 0.667 | 0.500 |
| Multinomial NB（`alpha=0.5`） | **0.750** | 0.766 | **0.720** | **0.742** | **0.837** |
| Linear SVM（`C=0.1`） | 0.740 | **0.779** | 0.670 | 0.720 | 0.806 |

在这份固定样本上，NB 的 Accuracy、Recall、F1 和 ROC-AUC 更高，SVM 的 Precision
更高。测试集只有 200 条，不能据此声称 NB 在一般情况下显著优于 SVM。

### 数据集

- [下载固定 Restaurant_Reviews.tsv](https://huggingface.co/datasets/KarthikaRajagopal/Restaurant_Reviews.tsv/resolve/ef6e2c7abbdf201ee732a2cb90ac394de6f5bb00/Restaurant_Reviews.tsv?download=true)
- 文件名：`Restaurant_Reviews.tsv`
- SHA-256：`2b432f36bc7c7ed77312187a9afd6531dfa3d16be48b9738131dd5141195b469`
- 本地路径：`data/raw/Restaurant_Reviews.tsv`

原计划 Kaggle competition 的 8 个 CSV 已停用，不得与当前 `Review`/`Liked` 数据
混合。原始数据和包含完整评论的处理后数据不上传 GitHub。字段和质量说明见
`data/data_dictionary.md` 与 `docs/data_quality.md`。

### 环境安装

建议 Python 3.11：

```bash
git clone https://github.com/lelezi521708/restaurant-sentiment.git
cd restaurant-sentiment
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 完整复现

推荐一条命令：

```bash
python scripts/reproduce_project.py
```

脚本会检查数据哈希，依次执行成员 2 数据处理、成员 3 特征选择、成员 4 SVM、成员 5
NB 与统一评价，运行全部测试，并生成 `member1_reproduction.json`。

也可以逐步运行：

```bash
python -m restaurant_sentiment.eda --data data/raw/Restaurant_Reviews.tsv
python -m restaurant_sentiment.preprocessing
python -m restaurant_sentiment.svm_experiment
python -m restaurant_sentiment.final_evaluation --svm-c 0.1
python -m pytest -q
```

预期：796 条训练、200 条测试、unigram、SVM `C=0.1`、NB `alpha=0.5`、
26 项测试通过。成员 1 的独立复现记录见
[`docs/member1_reproduction.md`](docs/member1_reproduction.md) 和
[`results/tables/member1_reproduction.json`](results/tables/member1_reproduction.json)。

### Notebook 和 Jupyter kernel

| Notebook | 负责人和内容 |
|---|---|
| `01_data_eda.ipynb` | 成员 2：质量检查、固定划分和 EDA |
| `02_preprocessing.ipynb` | 成员 3：TF-IDF 和 n-gram 比较 |
| `03_svm.ipynb` | 成员 4：Linear SVM 调参 |
| `04_naive_bayes.ipynb` | 成员 5：NB 参数选择 |
| `05_evaluation_error_analysis.ipynb` | 成员 5：统一评价和错误分析 |

避免 Jupyter 错用其他环境：

```bash
python -m pip install ipykernel
python -m ipykernel install --user \
  --name restaurant-sentiment \
  --display-name "Restaurant Sentiment (Python 3.11)"
```

在 JupyterLab 或 PyCharm 中选择该 kernel 后运行所有单元格。

### 成员分工与交付

| 成员 | 负责内容 | 主要交付物 |
|---|---|---|
| 成员 1 | 仓库、整合、独立复现、最终报告 | 双语 README、复现证明、数字修正、统一 LaTeX 报告 |
| 成员 2 | 数据清洗、EDA、固定划分 | 数据审计、`01_data_eda.ipynb`、EDA 图、Dataset/EDA 英文小节 |
| 成员 3 | 文本预处理和 TF-IDF | 特征 pipeline、n-gram 对比、`02_preprocessing.ipynb`、特征英文小节 |
| 成员 4 | Linear SVM | C 参数实验、`03_svm.ipynb`、预测、图表、SVM 英文小节 |
| 成员 5 | NB 和统一评估 | NB 调参、`04`/`05` Notebook、模型比较、ROC/混淆矩阵、错误分析、Results/Discussion |

### 实验规范

- `random_state=8017` 固定划分一次，训练集和测试集无评论重叠；
- 所有模型使用相同划分和指标；
- 每个验证折重新拟合完整 TF-IDF + classifier pipeline；
- 测试集不参与 n-gram、SVM `C` 或 NB `alpha` 的选择；
- 指标由代码生成，并记录哈希和依赖版本；
- 不上传完整评论、拟合模型、密钥或个人路径。

### 英文报告

统一 ICML 格式入口为 `report/main.tex`，仓库包含 ICML 2026 样式和统一参考文献。
提交前必须把 5 个成员和邮箱占位符替换为真实姓名、学号及联系方式。

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

最终必须检查正文是否符合课程规定的 8 页限制。

### 是否需要输入评论的演示界面

课程统计分析部分不要求网页。当前模型对比、ROC、混淆矩阵和错误分析图已经构成完整
可视化证据。如需现场展示，在 Notebook 最后增加一个单条评论预测单元格即可；该功能
属于演示，不能当作额外测试结果。
