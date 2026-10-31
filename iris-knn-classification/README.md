# Iris Flower Classification with KNN

一个面向机器学习初学者的中文案例教程：使用 Scikit-learn 内置鸢尾花（Iris）数据集，完成从数据探索到 KNN 分类、模型评估和参数调优的完整流程。

## 项目简介

本项目将鸢尾花分类作为一个监督学习多分类任务。模型根据花萼与花瓣的 4 个测量特征，预测样本属于 `setosa`、`versicolor` 或 `virginica` 中的哪一类。

教程以 **K-近邻算法（K-Nearest Neighbors, KNN）** 为主线，重点解释机器学习入门阶段最常见的概念和步骤。

## 学习内容

- 使用 `load_iris()` 加载 Scikit-learn 内置数据集
- 使用 Pandas 检查数据结构、统计信息和类别分布
- 使用 Matplotlib / Seaborn 进行探索性数据分析（EDA）
- 使用 `train_test_split()` 划分训练集与测试集
- 理解归一化（Min-Max Scaling）和标准化（Standardization）
- 使用 `StandardScaler` 进行特征标准化
- 使用 `KNeighborsClassifier` 训练 KNN 分类模型
- 使用准确率、混淆矩阵和分类报告评估模型
- 使用 `Pipeline`、交叉验证和 `GridSearchCV` 搜索 K 值

## 项目结构

```text
iris-knn-classification/
├── iris_knn_tutorial.ipynb   # 完整中文案例教程（含运行结果和图表）
├── README.md                 # 项目说明
├── requirements.txt          # Python 依赖
└── .gitignore                # Git 忽略规则
```

## 环境要求

- Python 3.10 或更高版本
- Jupyter Notebook / JupyterLab
- 依赖包见 `requirements.txt`

## 如何运行

### 1. 克隆仓库

```bash
git clone <你的 GitHub 仓库地址>
cd iris-knn-classification
```

### 2. 创建并激活虚拟环境（推荐）

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows：

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 启动 Jupyter

```bash
jupyter notebook
```

在浏览器中打开 `iris_knn_tutorial.ipynb`，选择 Python 3 内核，然后按顺序运行所有单元格。

> 也可以直接在 GitHub 仓库页面预览 `.ipynb` 文件中的 Markdown、代码、运行结果和图表。

## 案例流程

```text
加载 Iris 数据集
       ↓
数据检查与可视化
       ↓
划分训练集和测试集
       ↓
特征标准化（只在训练集上 fit）
       ↓
训练 KNN（初始 K=3）
       ↓
预测与模型评估
       ↓
交叉验证 + GridSearchCV 搜索 K 值
       ↓
使用独立测试集进行最终评估
```

## 关键知识点

### 1. 特征与标签

- `X`：模型输入的特征数据，包括花萼长度、花萼宽度、花瓣长度和花瓣宽度。
- `y`：模型需要预测的类别标签。

### 2. 为什么要标准化？

KNN 通过样本之间的距离进行分类。当特征量纲或数值范围不一致时，数值范围较大的特征可能对距离产生更大影响。`StandardScaler` 将特征转换为均值约为 0、标准差约为 1 的尺度。

### 3. 如何避免数据泄漏？

正确顺序是：

- 先划分训练集和测试集。
- 标准化器只在训练集上执行 `fit` 或 `fit_transform`。
- 测试集和新样本只执行 `transform`。

参数搜索部分使用 `Pipeline`，使标准化在每个交叉验证训练折内部完成。

### 4. 如何评价分类模型？

教程包含：

- Accuracy：整体预测正确率
- Confusion Matrix：查看不同类别被错分的情况
- Precision：精确率
- Recall：召回率
- F1-score：精确率与召回率的综合指标

## 复现实验说明

Notebook 固定了随机种子，并使用分层抽样，使实验结果更容易复现。运行后可在 Notebook 中查看实际准确率、分类报告、混淆矩阵和交叉验证结果。

模型结果会受到数据划分、算法参数和软件版本影响。请以自己实际运行的输出为准，不要把某一次运行的准确率理解为模型在所有数据上的固定表现。

## 后续可以尝试

1. 比较 KNN、决策树、逻辑回归和 SVM 的分类效果。
2. 修改 `test_size`，观察训练集和测试集划分对评估结果的影响。
3. 修改 K 值搜索区间，比较不同参数的交叉验证表现。
4. 尝试 `MinMaxScaler`，与 `StandardScaler` 的结果进行比较。
5. 尝试保存训练好的模型，并编写独立的预测脚本。

---

**关键词：** Machine Learning · Supervised Learning · Classification · KNN · Iris Dataset · Scikit-learn · Feature Scaling · Model Evaluation · Cross Validation
