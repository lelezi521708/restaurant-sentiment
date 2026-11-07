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

- 数据来源：[Kaggle Restaurant Reviews](https://www.kaggle.com/competitions/restaurant-reviews/data)
- 预计文本列：`Review`
- 预计标签列：`Liked`
- 标签含义：`0` 表示负面，`1` 表示正面
- 使用条件：受 Kaggle 竞赛规则约束

### 数据下载方式

原始数据不会通过本仓库重新发布。每位成员需要：

1. 登录 Kaggle；
2. 从上面的链接下载同一份数据；
3. 将文件放入本地 `data/raw/`；
4. 根据实际文件更新 [数据字典](data/data_dictionary.md)；
5. 不要把原始数据、Kaggle token 或 `kaggle.json` 上传到 GitHub。

例如：

~~~text
data/raw/Restaurant_Reviews.tsv
~~~

`data/raw/` 和 `data/processed/` 中的数据文件已经被 `.gitignore` 排除。

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

### 4. 下载并放置数据

从 Kaggle 下载后，将实际 CSV 或 TSV 文件放入：

~~~text
data/raw/
~~~

### 5. 运行完整 baseline

如果文件名是 `Restaurant_Reviews.tsv`：

~~~bash
python -m restaurant_sentiment.cli \
  --data data/raw/Restaurant_Reviews.tsv \
  --models dummy nb svm
~~~

如果实际文件名或列名不同：

~~~bash
python -m restaurant_sentiment.cli \
  --data data/raw/your_file.csv \
  --text-column Review \
  --label-column Liked
~~~

结果会自动保存到：

- `results/tables/`：模型指标和关键词表；
- `results/figures/`：混淆矩阵等图表。

### 6. 运行测试

~~~bash
pytest
~~~

### 7. 启动 JupyterLab

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
