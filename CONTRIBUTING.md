# 项目协作规范

本文档规定五位成员如何共同开发，避免 notebook 冲突、实验设置不一致、结果无法复现等问题。

## 一、语言要求

请严格遵守以下语言约定：

- README、CONTRIBUTING、数据说明、任务说明、PR 和 Issue 等组内协作文档使用中文。
- 每位成员提交的课程报告小节必须使用英文。
- 最终课程报告、图表标题、表格标题和图注统一使用英文。
- Python 变量名、函数名、类名和文件名使用英文。
- Git commit message 建议使用简洁英文。
- 代码注释可以使用简洁英文；复杂逻辑可以补充中文解释。

## 二、开始任务前

1. 拉取最新 `main` 分支。
2. 阅读 README.md 和 data/data_dictionary.md。
3. 从 Kaggle 自行下载数据集。
4. 建立并激活 Python 3.11 虚拟环境。
5. 安装依赖并运行测试。
6. 确认任务负责人、输入、输出和截止时间。

~~~bash
git switch main
git pull --ff-only origin main
python -m pip install -r requirements.txt
python -m pip install -e .
pytest
~~~

## 三、分支命名

每个任务单独建立分支：

- `member-1/project-setup`
- `member-2/data-eda`
- `member-3/tfidf-features`
- `member-4/svm`
- `member-5/nb-evaluation`

后续任务使用 `member-N/short-task-name` 格式。已经合并的分支不要继续用于无关任务。

建立分支示例：

~~~bash
git switch main
git pull --ff-only origin main
git switch -c member-3/tfidf-features
~~~

## 四、每次提交必须包含什么

每位成员的 Pull Request 应同时包含：

1. 自己负责模块的代码或 notebook；
2. 自己负责的英文报告小节；
3. 必要的英文图表标题、表格标题和图注；
4. 复现实验所需的 README 或说明文档更新；
5. 输入文件、输出文件、参数和验证结果说明；
6. `pytest` 通过的结果，或者对已知问题的解释。

不要在一个 Pull Request 中混合多个无关任务。

## 五、Commit message

使用简短、明确的英文 commit message，例如：

- `add data quality checks`
- `implement tfidf feature pipeline`
- `tune linear svm`
- `add model comparison figures`
- `document experiment commands`

不要使用 `update`、`final`、`fix stuff`、`new version` 等无法说明内容的名称。

## 六、Python 代码规范

- 可复用逻辑放在 `src/restaurant_sentiment/`。
- 函数名称应清楚表达用途。
- 对重要函数添加 docstring 和必要注释。
- 优先使用 scikit-learn Pipeline，避免手工重复转换。
- 所有支持随机种子的步骤使用 `random_state=8017`。
- 文件不存在、列缺失或标签错误时，应给出清楚的错误信息。
- 不要在代码中使用某位成员电脑上的绝对路径。
- 不要在多个 notebook 中复制相同的数据清洗代码。

## 七、Notebook 规范

- 每个编号 notebook 由对应成员负责。
- notebook 按编号顺序组织。
- 开头用中文写明负责人、任务、输入和预期输出。
- 从 `src/restaurant_sentiment/` 导入可复用函数。
- 不能在测试集上拟合 TF-IDF 或选择参数。
- 提交前删除废弃代码和大量调试输出。
- 重启 kernel 后，从第一格到最后一格完整运行一次。
- 最终图表和表格保存到 `results/`，不能只保留在 cell 输出中。
- notebook 内用于组内协作的说明使用中文。
- 将要放入最终报告的图表标题、坐标轴、图例和表格标题使用英文。

## 八、数据规范

- 不提交 `data/raw/` 或 `data/processed/` 中的数据文件。
- 不提交 Kaggle token、API key 或 `kaggle.json`。
- 在数据字典中记录来源、下载日期、文件名、列、标签和清洗数量。
- 所有人使用统一的数据划分和随机种子。
- 不能无说明地删除数据或修改标签。
- 如果 Kaggle 使用条件发生变化，应停止传播数据并更新 README。

## 九、实验结果和英文报告

- 模型指标必须由代码生成，不能手工修改。
- 图表必须有英文标题、坐标轴、图例和图注。
- 表格标题和列名应适合直接放入英文报告。
- 英文报告必须说明模型参数和数据划分。
- 不仅报告最高分，也要讨论错误案例和局限性。
- 数据集、方法和参考代码需要正确引用。
- 每位成员提交的报告内容必须是英文，不接受中文报告段落后再由成员 1 全部翻译。

生成的 `results/` 文件默认被 Git 忽略。全组确认后，只强制添加最终报告需要的文件：

~~~bash
git add -f results/tables/final_model_comparison.csv
git add -f results/figures/final_confusion_matrix.png
~~~

## 十、申请审阅前运行测试

~~~bash
pytest
python -m restaurant_sentiment.cli \
  --data data/raw/Restaurant_Reviews.tsv \
  --models dummy nb svm
~~~

如果审阅人本地没有原始数据，至少必须确保单元测试全部通过。

## 十一、提交 Pull Request

1. 从最新 `main` 更新自己的分支。
2. 在本地解决冲突。
3. 推送自己的分支。
4. 使用仓库中的中文 PR 模板填写说明。
5. 请求至少一位成员审阅。
6. 根据意见继续提交修改。
7. 确认代码、英文报告小节和中文说明文档都完整后再合并。

~~~bash
git fetch origin
git rebase origin/main
git push -u origin member-3/tfidf-features
~~~

如果一个分支由多人使用，不要在未通知其他人的情况下 force push。

## 十二、代码审阅检查

审阅人需要确认：

- 修改内容符合成员负责范围；
- 没有提交原始数据、密钥或个人绝对路径；
- 代码可以理解和复现；
- 使用统一的数据划分和随机种子；
- 没有发生测试集数据泄漏；
- 指标和图表与保存结果一致；
- 英文报告小节准确描述代码和实验；
- 中文 README 和运行说明仍然有效。

## 十三、沟通和记录

临时沟通可以使用群聊，但重要决定应整理到 GitHub Issue、README 或 Pull Request 中，避免信息丢失。
