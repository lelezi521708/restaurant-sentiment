# 餐厅评论情感分析与分类

STAT8017 Data Mining Techniques 小组项目

本项目使用带有正面/负面标签的餐厅评论，建立一套可复现的文本分类流程。我们会使用相同的 TF-IDF 特征和训练/测试集，对比多数类基线、Multinomial Naive Bayes 和 Linear SVM，并进一步分析关键词和错误案例。

## 一、项目要做什么

简单来说，我们要让计算机根据一条餐厅评论的文字内容，判断这条评论属于正面还是负面。

整体流程：

~~~text
下载带标签的餐厅评论
        ↓
检查并清洗数据
        ↓
使用 TF-IDF 把文本转换成数字特征
        ↓
训练多数类基线、Naive Bayes 和 Linear SVM
        ↓
在同一测试集上比较模型表现
        ↓
分析重要关键词和错误预测
        ↓
完成英文课程报告
~~~

主要研究问题：

1. Linear SVM 是否比 Multinomial Naive Bayes 表现更好？
2. unigram + bigram 是否优于只使用 unigram？
3. 哪些词或短语与正面、负面评价最相关？
4. 模型最容易在哪些类型的评论上判断错误？

最终交付物包括：

- 可复现的 Python 代码和 notebook；
- 模型结果、表格和图表；
- ICML 格式英文报告，正文不超过课程规定的 8 页；
- 完整的 README、数据说明和协作记录。

## 二、语言约定

- 本 README 和组内协作文档使用中文，方便组员理解。
- 代码变量名、函数名和文件名使用英文。
- 代码注释可使用简洁英文，必要时可补充中文解释。
- Git commit message 建议使用简洁英文。
- 最终课程报告、图表标题和表格标题统一使用英文。

## 三、数据集

### 最终采用的数据方案（所有成员必须遵守）

本项目最终统一使用含 `Review`（评论文本）和 `Liked`（0 负面、1 正面）的公开 `Restaurant_Reviews.tsv`：

- [下载固定版本 Restaurant_Reviews.tsv](https://huggingface.co/datasets/KarthikaRajagopal/Restaurant_Reviews.tsv/resolve/ef6e2c7abbdf201ee732a2cb90ac394de6f5bb00/Restaurant_Reviews.tsv?download=true)
- 文件名必须为 `Restaurant_Reviews.tsv`；
- SHA-256 必须为 `2b432f36bc7c7ed77312187a9afd6531dfa3d16be48b9738131dd5141195b469`；
- 原始样本为 1,000 条，负面和正面各 500 条；
- Member 2 的清洗代码会删除 4 条同标签重复记录，保留 996 条。

> **重要：原计划 Kaggle competition 下载的 8 个 CSV 已停用，不再用于本项目。**这些 CSV 的结构与当前 `Review`/`Liked` 情感二分类任务不一致。请不要读取、合并或上传这些旧 CSV；如已下载，可以移到项目外的备份目录。原 Kaggle 入口仅保留为选题过程记录，不是当前数据源。

原始 TSV 和处理后的完整评论不会上传 GitHub。每位成员需要自行下载固定版本，并在本地运行 Member 2 的代码生成完全一致的训练集和测试集。来源、字段、版本和统计见[数据字典](data/data_dictionary.md)。

### Member 3–5 如何接续 Member 2 的工作

每位成员按以下顺序操作：

1. 从上面的固定链接下载 `Restaurant_Reviews.tsv`；
2. 放到自己克隆仓库的 `data/raw/Restaurant_Reviews.tsv`；
3. 在仓库根目录安装依赖和项目包；
4. 运行 Member 2 的 EDA 命令；
5. 确认本地生成 `data/processed/train.csv` 和 `data/processed/test.csv`；
6. 后续 TF-IDF、SVM 和 Naive Bayes 只使用这两个固定划分文件。

~~~text
下载 Restaurant_Reviews.tsv
            ↓
放入 data/raw/Restaurant_Reviews.tsv
            ↓
运行 python -m restaurant_sentiment.eda
            ↓
生成 data/processed/train.csv（796 条）
     data/processed/test.csv（200 条）
            ↓
成员 3–5 开始各自任务
~~~

`data/raw/` 和 `data/processed/` 已被 Git 忽略，所以每位成员都要在自己的电脑上执行以上步骤。不要把原始 TSV、`train.csv` 或 `test.csv` 强制上传到 GitHub。

## 四、开发环境

建议环境：

- Python 3.11
- pandas
- NumPy
- scikit-learn
- Matplotlib
- seaborn
- JupyterLab
- pytest

以下命令默认使用 `python`。如果电脑无法识别，可以尝试 `python3`。

## 五、快速开始

### 1. 克隆仓库

~~~bash
git clone https://github.com/lelezi521708/restaurant-sentiment.git
cd restaurant-sentiment
~~~

### 2. 创建虚拟环境

macOS/Linux：

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
~~~

Windows PowerShell：

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

PyCharm 用户也可以在 Settings → Project → Python Interpreter 中选择：

~~~text
.venv/bin/python
~~~

### 3. 安装依赖

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
~~~

### 4. 下载并放置统一的原始数据

从[固定版本链接](https://huggingface.co/datasets/KarthikaRajagopal/Restaurant_Reviews.tsv/resolve/ef6e2c7abbdf201ee732a2cb90ac394de6f5bb00/Restaurant_Reviews.tsv?download=true)下载文件，并确保目录如下：

~~~text
restaurant-sentiment/
└── data/
    └── raw/
        └── Restaurant_Reviews.tsv
~~~

macOS/Linux 可选检查：

~~~bash
shasum -a 256 data/raw/Restaurant_Reviews.tsv
~~~

输出应以以下哈希开始：

~~~text
2b432f36bc7c7ed77312187a9afd6531dfa3d16be48b9738131dd5141195b469
~~~

### 5. 运行 Member 2 的清洗、划分与 EDA

在仓库根目录执行：

~~~bash
python -m restaurant_sentiment.eda --data data/raw/Restaurant_Reviews.tsv
~~~

也可以打开 [01_data_eda.ipynb](notebooks/01_data_eda.ipynb)，从头运行所有单元格。全组已经固定数据版本和列名，不应再自行更换 CSV、列名或重新随机划分。

- 后续特征与建模 notebook 读取 `data/processed/train.csv` 和 `data/processed/test.csv`，以 `Review` 为输入、`Liked` 为标签。划分按标签分层，比例为 80/20，随机种子为 `8017`。
- 同目录保存完整清洗表 `cleaned_reviews.csv` 和对应划分清单 `split_manifest.csv`。
- `results/tables/` 包含质量审计（`data_quality.json` 和 `data_quality.csv`）、类别分布、长度和训练集词频。三张英文图保存到 `results/figures/`。
- 质量结论见[数据质量报告](docs/data_quality.md)。英文报告小节及引用见 [report 说明](report/README.md)。

运行完成后，可以检查文件是否生成：

~~~bash
wc -l data/processed/train.csv data/processed/test.csv
~~~

预期为 797 行和 201 行，其中各包含一行表头，即实际数据分别为 796 条和 200 条。

### 6. 后续成员使用统一数据

- Member 3：读取 `train.csv` 和 `test.csv`，只在训练集拟合 TF-IDF，再转换测试集；
- Member 4：使用 Member 3 的统一特征设置和固定划分训练 Linear SVM；
- Member 5：使用同一划分训练 Naive Bayes，并对所有模型进行统一评估；
- 所有人不得各自重新执行不同的随机划分，也不得混入原来的 8 个 Kaggle CSV。

### 7. 运行成员 3 的特征实验

```bash
python -m restaurant_sentiment.preprocessing
```

直接读取成员 2 的固定划分，在训练集内使用相同的五折比较 unigram 与 unigram + bigram。向量化器在每个验证折内重新拟合，测试集不参与选参或评价。

#### Member 3 已完成的内容

Member 3 的代码会先检查 Member 2 的固定交接文件：

```text
data/processed/train.csv
data/processed/test.csv
data/processed/split_manifest.csv
```

检查内容包括：

- `review_id` 是否有效且唯一；
- train/test 的 ID 和评论文本是否重叠；
- 标签是否只包含 0 和 1；
- train/test 是否与 `split_manifest.csv` 完全一致；
- 是否错误地重新清洗、随机打乱或划分数据。

共享 TF-IDF 设置包括：转小写、Unicode 去音调、保留否定词、不删除停用词、不做词干化或词形还原、sublinear TF、平滑 IDF 和 L2 归一化。详细参数见[成员 3 TF-IDF 交接](docs/member3_tfidf.md)。

特征比较只使用训练集，并采用：

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=8017)
```

每一个验证折都会重新拟合完整的 TF-IDF + classifier pipeline，避免验证折词汇或 IDF 泄漏。比较阶段固定使用 Linear SVM `C=1` 和 Multinomial NB `alpha=1`，测试集不参与特征选择。

| Features | Probe model | Mean F1 | SD |
|---|---|---:|---:|
| Unigram | Linear SVM | 0.8016 | 0.0282 |
| Unigram + bigram | Linear SVM | 0.7998 | 0.0388 |
| Unigram | Multinomial NB | 0.7962 | 0.0484 |
| Unigram + bigram | Multinomial NB | 0.7845 | 0.0394 |

最终按固定 SVM 的平均正类 F1 选择 **unigram** 为共享默认。unigram 有 1,781 个特征，unigram + bigram 有 7,345 个特征；F1 差异约为 0.0018，远小于折间波动，因此不能宣称 unigram 在一般情况下显著更好。

- 可运行 notebook：`notebooks/02_preprocessing.ipynb`；
- 中文参数、输出、复现与交接说明：[成员 3 交接](docs/member3_tfidf.md)；
- 汇总表：`results/tables/tfidf_cv_comparison.csv`；
- 特征统计：`results/tables/tfidf_feature_summary.csv`；
- 实验配置和输入哈希：`results/tables/tfidf_experiment.json`；
- 对比图：`results/figures/tfidf_ngram_cv_f1.png`；
- 特征矩阵及行清单：`data/processed/tfidf/`，仅保留本地；
- 拟合向量化器及参数：`models/tfidf/`，仅保留本地；
- 英文报告小节：`report/text_features.tex`。

#### Member 1 的审查与修改

Member 1 已在项目 Python 3.11 环境完成代码审查和复现，并进行了以下修改：

1. 将 unigram 锁定为 `DEFAULT_NGRAM_RANGE=(1, 1)`；
2. 将共享 `svm_parameter_grid()` 改为只搜索 `C=[0.1, 1.0, 10.0]`，不再重新搜索 n-gram；
3. 增加回归测试，确保后续不会意外恢复 n-gram 搜索；
4. 提交四个不含原始评论的汇总表和图；
5. 更新 Member 3 的交接状态和过期说明；
6. 确认原始 TSV、train/test、稀疏矩阵、词汇表和 joblib 模型没有上传 GitHub。

复现结果：14 个测试全部通过，`02_preprocessing.ipynb` 可在 Python 3.11 环境从头执行，测试集未参与特征或参数选择。

#### Member 4 接下来如何使用

合并后先更新本地 `main`，重新生成 Member 2 和 Member 3 的本地文件：

```bash
git switch main
git pull origin main
python -m restaurant_sentiment.eda --data data/raw/Restaurant_Reviews.tsv
python -m restaurant_sentiment.preprocessing
```

Member 4 使用 unigram 训练和调优 Linear SVM。只搜索 `C`，不能重新搜索 unigram/bigram：

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from restaurant_sentiment.pipeline import build_pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=8017)
search = GridSearchCV(
    estimator=build_pipeline("svm"),
    param_grid={"classifier__C": [0.1, 1.0, 10.0]},
    scoring="f1",
    cv=cv,
    n_jobs=-1,
)
search.fit(train["Review"], train["Liked"])
```

Member 4 必须注意：

- 只能根据训练集交叉验证选择 `C`；
- 每折必须重新拟合完整 TF-IDF + SVM pipeline；
- 不能使用已在完整训练集上拟合的 `.npz` 特征矩阵进行内部交叉验证；
- 不能根据测试集选择参数；
- 最终测试集评价要与 Member 5 协调后统一进行；
- 需要提交代码/notebook、各个 C 的 CV 结果、最佳参数、中文交接说明和英文 Methodology/Experiments 小节。

#### Member 5 接下来如何使用

Member 5 使用相同数据划分和 unigram 特征：

```python
from restaurant_sentiment.pipeline import build_pipeline

nb_pipeline = build_pipeline("nb")
```

如需调节 Multinomial NB，可以只在训练集交叉验证中比较：

```python
{"classifier__alpha": [0.1, 0.5, 1.0, 2.0]}
```

SVM 和 NB 的设置全部确定后，Member 5 再使用固定的 200 条测试集完成统一最终评估，包括 Accuracy、Precision、Recall、F1、ROC-AUC、混淆矩阵、重要词语和错误案例。所有模型必须使用相同测试集和指标，不能重新划分数据。

### 8. 运行成员 4 的 Linear SVM 实验

```bash
python -m restaurant_sentiment.svm_experiment
```

该命令只在 796 条训练评论内进行五折分层交叉验证，比较 `C=[0.1, 1.0, 10.0]`，并在每一折重新拟合完整 unigram TF-IDF + Linear SVM pipeline。按正类 F1 选择的最终参数为 **`C=0.1`**（CV F1 0.8060）。参数锁定后，模型在固定 200 条测试集上的 Accuracy 为 0.7400、F1 为 0.7204、ROC-AUC 为 0.8061。

主要输出：

- `results/tables/svm_cv_results.csv`：每个 C 的五折指标；
- `results/tables/svm_test_metrics.csv`：锁定模型的测试指标；
- `results/tables/svm_test_predictions.csv`：不含评论原文的逐条预测；
- `results/tables/svm_top_terms.csv`：正负权重最高的词；
- `results/tables/svm_experiment.json`：版本、哈希和实验参数；
- `results/figures/svm_c_cv_f1.png` 和 `svm_test_confusion_matrix.png`；
- `docs/member4_svm.md`：中文交接；
- `report/svm.tex` 和 `report/svm.bib`：英文报告小节与引用。

成员 5 的统一评估应使用：

```bash
python -m restaurant_sentiment.final_evaluation --svm-c 0.1
```

Member 5 需要提交 NB 代码/notebook、统一模型比较、图表、关键词与错误分析、中文交接说明，以及英文 Results、Error Analysis 和 Discussion 小节。

#### 下游共同约束

- `build_pipeline()` 默认统一使用 unigram；
- 内部交叉验证必须从原始训练文本和完整 pipeline 开始；
- 不能先在全部训练集拟合 TF-IDF，再对该矩阵做交叉验证；
- 测试集只在所有模型和参数锁定后使用一次；
- Member 3 的本地 `.npz` 和 joblib 文件可以用于最终拟合后的交接，但不能代替训练折内的特征拟合；
- 最终报告、图表标题、表格标题和图注全部使用英文。

### 9. 运行完整 baseline

如果文件名是 `Restaurant_Reviews.tsv`：

~~~bash
python -m restaurant_sentiment.cli \
  --data data/raw/Restaurant_Reviews.tsv \
  --models dummy nb svm
~~~

结果会自动保存到：

- `results/tables/`：模型指标和关键词表；
- `results/figures/`：混淆矩阵等图表。

### 10. 运行测试

~~~bash
pytest
~~~

### 11. 启动 JupyterLab

~~~bash
jupyter lab
~~~

notebook 应尽量调用 `src/restaurant_sentiment/` 中已经测试的函数，不要在多个 notebook 中重复复制数据清洗或建模代码。

## 六、仓库结构

~~~text
restaurant-sentiment/
├── README.md                         # 中文项目说明
├── CONTRIBUTING.md                   # GitHub 协作规范
├── requirements.txt                  # Python 依赖
├── pyproject.toml                    # 包和测试配置
├── data/
│   ├── raw/                          # 本地原始数据，不上传
│   ├── processed/                    # 本地处理数据，不默认上传
│   └── data_dictionary.md            # 数据字典
├── notebooks/
│   ├── 01_data_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_svm.ipynb
│   ├── 04_naive_bayes.ipynb
│   └── 05_evaluation_error_analysis.ipynb
├── src/restaurant_sentiment/
│   ├── config.py                     # 随机种子和输出路径
│   ├── data.py                       # 读取、清洗和划分数据
│   ├── eda.py                        # 数据审计、划分文件和 EDA 图表
│   ├── features.py                   # TF-IDF 特征
│   ├── models.py                     # Dummy、NB 和 SVM
│   ├── evaluation.py                 # 指标、图表和关键词
│   ├── pipeline.py                   # 完整训练管线
│   └── cli.py                        # 命令行入口
├── tests/                            # 单元测试和合成测试数据
├── results/
│   ├── figures/
│   └── tables/
├── report/                           # 英文报告源文件
└── .github/                          # PR 和 Issue 模板
~~~

## 七、成员分工

| 成员 | 负责模块 | 主要交付物 | 建议完成时间 |
|---|---|---|---|
| 成员 1 | 仓库、数据说明和项目整合 | 仓库结构、README、数据字典、整合、复现检查和最终提交 | 11 月 9 日完成框架；11 月 25 日初步整合；11 月 30 日最终版 |
| 成员 2 | 数据清洗、EDA 和数据质量 | 清洗代码、固定数据划分、数据质量总结、EDA 图表和英文报告小节 | 11 月 13 日 |
| 成员 3 | 文本预处理和 TF-IDF | TF-IDF pipeline、unigram/bigram 对比、特征总结和英文报告小节 | 11 月 16 日 |
| 成员 4 | Linear SVM | C 参数实验、交叉验证、预测结果、结果表和英文报告小节 | 11 月 20 日 |
| 成员 5 | Naive Bayes、统一评估和错误分析 | NB、模型比较、结果图表、关键词/错误分析和英文报告小节 | 11 月 23 日 |

每位成员都需要提交：

1. 可复现的代码或 notebook；
2. 自己负责的英文报告小节；
3. 必要的图表、表格和引用；
4. README 或其他文档更新，说明输入、运行方式、输出和参数。

## 八、任务依赖和并行关系

~~~text
成员 1：仓库和数据说明
              ↓
成员 2：数据清洗、EDA、固定数据划分
              ↓
成员 3：TF-IDF 和特征实验
              ↓
       ┌──────┴──────┐
       ↓             ↓
成员 4：SVM      成员 5：Naive Bayes
       └──────┬──────┘
              ↓
成员 5：统一评估和错误分析
              ↓
成员 1：整合、复现检查和最终提交
~~~

成员 2–5 可以提前并行准备代码框架，但最终模型比较必须使用全组统一的数据划分和 TF-IDF pipeline。

## 九、实验约定

- 所有支持随机种子的操作统一使用 `random_state=8017`。
- 数据只进行一次分层划分，所有模型使用同一测试集。
- TF-IDF 和特征选择只能在训练集上拟合。
- 测试集只用于最终评估，不能用于选参数。
- 最终至少报告 Accuracy、Precision、Recall、F1 和 ROC-AUC。
- 最终图表和表格保存在 `results/`。
- 记录模型参数、依赖版本和输出路径。
- 不允许手动修改实验指标。

## 十、GitHub 开发流程

首次框架可以提交到 `main`。之后所有成员通过自己的分支开发。

例如成员 2：

~~~bash
git switch main
git pull --ff-only origin main
git switch -c member-2/data-eda

# 完成修改后
git add .
git commit -m "add data quality checks"
git push -u origin member-2/data-eda
~~~

然后在 GitHub 上创建 Pull Request，请至少一位成员检查后再合并进 `main`。

分支建议：

- `member-1/project-setup`
- `member-2/data-eda`
- `member-3/tfidf-features`
- `member-4/svm`
- `member-5/nb-evaluation`

完整规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 十一、提交代码前检查

~~~bash
pytest
~~~

提交 notebook 前还需要确认：

- 可以从第一格运行到最后一格；
- 没有使用个人电脑绝对路径；
- 没有提交原始数据或密钥；
- 没有在测试集上拟合 TF-IDF；
- 结果图表与英文报告中的数字一致；
- README 中的运行方式仍然有效。

## 十二、预期输出

运行完整 CLI 后会生成：

- `results/tables/model_comparison.csv`；
- `results/tables/svm_top_terms.csv`；
- Dummy、NB 和 SVM 的混淆矩阵。

生成结果默认被 Git 忽略，避免多位成员产生冲突。全组确认最终版本后，只提交报告需要的最终表格和图表。

## 十三、英文报告结构

建议最终英文报告采用以下结构：

1. Introduction
2. Dataset and Exploratory Data Analysis
3. Text Preprocessing and Feature Representation
4. Methods and Experimental Setup
5. Results and Error Analysis
6. Limitations and Impact Statement
7. Conclusion

报告中的每一个数字，都应能够追溯到代码、notebook 或生成结果。

## 十四、常见问题

### 找不到数据文件

检查 `--data` 参数以及数据是否放在 `data/raw/`。

### 提示缺少 Review 或 Liked

先查看真实列名，然后使用 `--text-column` 和 `--label-column` 指定。

### 无法导入 restaurant_sentiment

确认虚拟环境已激活，然后执行：

~~~bash
python -m pip install -e .
~~~

### notebook 出现合并冲突

避免两个人同时修改同一个 notebook。每个 notebook 由对应成员负责，其他成员通过 PR 提出意见。

## 十五、数据和许可说明

本仓库只保存项目代码、测试和文档，不代表获得重新发布 Kaggle 竞赛数据的许可。数据的下载和使用仍需遵守 Kaggle 页面及相关竞赛规则。
