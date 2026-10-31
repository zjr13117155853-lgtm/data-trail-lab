# 28×28 手写数字识别：KNN 入门项目

一个适合机器学习初学者复习和继续扩展的 Python 小项目。项目以你修改后的 Notebook 为基础，保留 **KNN（K-Nearest Neighbors，K 近邻）** 作为分类算法，按“读取数据 → 查看图片 → 预处理 → 划分数据集 → 训练 → 评估 → 保存模型 → 识别自定义图片”的顺序组织。

> 当前版本对应你上传的 Notebook：`KNeighborsClassifier(n_neighbors=3)`、测试集比例 `0.2`、`random_state=1`、`stratify=y`，并将训练像素除以 255。原 Notebook 最近一次记录的测试集准确率为 **0.9661904762（约 96.62%）**。本仓库没有收到训练 CSV，因此不能在这里独立复算该数值；实际结果以你运行自己的 CSV 后显示的结果为准。

## 1. 项目目标

输入一张 28×28 像素的灰度数字图片，让 KNN 模型预测图片表示的数字（0～9）。每张图片有 28×28=784 个像素，因此模型输入是一个包含 784 个特征的一维向量。

本项目是经典机器学习入门案例，不使用 CNN 或深度学习框架。重点是理解：

1. 特征（X）与标签（y）是什么；
2. 为什么要把 28×28 图片展开成 784 个特征；
3. 为什么训练图片和预测图片必须采用一致的预处理；
4. 训练集、测试集、准确率分别意味着什么；
5. 如何保存模型，并对自己准备的图片进行预测。

## 2. 项目目录

```text
handwritten-digit-recognition-knn/
├── README.md                         # 项目说明、环境配置、运行步骤
├── requirements.txt                  # Python 依赖
├── .gitignore                        # 忽略数据集、模型和临时文件
├── data/
│   ├── README.md                      # 训练数据格式说明
│   ├── demo.png                       # 28×28 演示图片
│   └── 手写数字识别.csv                # 需要你自行放入，不随项目提供
├── models/
│   └── knn_digit_model.joblib         # 训练后自动生成，不提交 GitHub
├── notebooks/
│   └── 01_手写数字识别_KNN完整教程.ipynb
└── src/
    ├── __init__.py
    ├── project_utils.py               # 数据加载、检查和图片预处理
    ├── train.py                       # 模型训练与评估
    └── predict.py                     # 单张图片预测
```

## 3. 环境准备（macOS）

建议使用 Python 3.10 或更新版本。你可以在终端进入项目根目录后执行：

```bash
# 1) 创建虚拟环境（只需创建一次）
python3 -m venv .venv

# 2) 激活虚拟环境
source .venv/bin/activate

# 3) 更新 pip
python -m pip install --upgrade pip

# 4) 安装项目依赖
python -m pip install -r requirements.txt
```

之后每次重新打开终端，进入项目目录后先执行：

```bash
source .venv/bin/activate
```

如果你已经在 Jupyter Notebook 中使用了其他 Python 环境，请确认 Notebook 内核和安装依赖的 Python 是同一个环境。

## 4. 准备训练数据

本次上传的文件中没有训练 CSV。请将你原项目中的 `手写数字识别.csv` 复制到：

```text
data/手写数字识别.csv
```

数据格式必须是：

- 第 1 列：真实数字标签，取值 0～9；
- 后面 784 列：28×28 图片展开后的像素值；
- 像素值范围：0～255；
- 不要多出一列 Pandas 行索引。

例如，某一行可以理解为：`标签 + pixel_0 + pixel_1 + ... + pixel_783`。列名可不同，但列的顺序必须是第一列标签、其余 784 列像素。

**数据集是否可以公开上传，需要以你获取该 CSV 时的许可或数据集页面条款为准。** 本项目的 `.gitignore` 默认忽略 `data/*.csv`，避免不小心把较大的数据集或未经确认授权的数据推送到 GitHub。

## 5. 方式一：运行 Notebook（推荐初学者）

从项目根目录执行：

```bash
jupyter lab notebooks/01_手写数字识别_KNN完整教程.ipynb
```

按照 Notebook 从上到下依次运行。教程中包含数据观察、图片展示、预处理、训练集/测试集划分、KNN 训练、评估、模型保存和自定义图片预测。

Notebook 会自动尝试从项目根目录的 `data/` 文件夹查找 CSV 和演示图片。请不要把 Notebook 单独移动到其他目录，否则相对路径可能需要调整。

## 6. 方式二：用 Python 脚本运行

### 第一步：训练模型

```bash
python -m src.train
```

如果 CSV 不在默认位置，可以指定路径：

```bash
python -m src.train --data "data/手写数字识别.csv" --neighbors 3 --test-size 0.2
```

训练脚本会：

1. 检查 CSV 是否包含 1 列标签和 784 列像素；
2. 将像素值除以 255，转换到 0～1；
3. 按 80%/20% 拆分训练集和测试集；
4. 使用 `stratify=y` 保持训练集和测试集中的类别比例；
5. 训练 KNN 模型；
6. 输出准确率、分类报告和混淆矩阵；
7. 将模型保存到 `models/knn_digit_model.joblib`。

### 第二步：识别演示图片

```bash
python -m src.predict --image data/demo.png --show
```

识别你自己的图片：

```bash
python -m src.predict --image "data/my_number.png" --show
```

图片要求：**28×28、单通道灰度图**。默认像素方向是黑色背景、白色笔画，且会按照训练时的规则除以 255。如果你的图片为白色背景、黑色笔画，可显式反转：

```bash
python -m src.predict --image "data/my_number.png" --invert --show
```

> `predict_proba` 输出的是 K 个近邻中的类别投票比例，不是经过概率校准的真实正确率，不应把它直接理解为模型可靠性保证。

## 7. 核心代码理解

### 7.1 一张图片为什么有 784 个特征？

图片大小是 28×28，每一个像素都可以看作一个数值特征：

```python
image_array = image_array.reshape(1, 784)
```

原来的二维矩阵形状是 `(28, 28)`；`reshape(1, -1)` 将其转换成模型需要的单行输入，形状为 `(1, 784)`。这里的 `1` 表示一张待预测图片，`784` 表示特征数量。

### 7.2 X 和 y 分别是什么？

```python
X = df.iloc[:, 1:]  # 第 2 列到最后：784 个像素特征
y = df.iloc[:, 0]   # 第 1 列：数字标签
```

`X` 是模型用来判断图片的输入信息；`y` 是每张训练图片对应的正确答案。监督学习通过训练样本中的 `(X, y)` 关系学习分类规则。

### 7.3 为什么除以 255？

原始灰度像素一般为 0～255。除以 255 后缩放到 0～1：

```python
X = X / 255.0
```

KNN 通过计算样本之间的距离寻找近邻。训练和预测必须使用同一种像素范围、颜色方向、尺寸和特征顺序。本项目中的图片预测也执行相同的 `/255` 操作。

### 7.4 KNN 的 k=3 是什么？

```python
KNeighborsClassifier(n_neighbors=3)
```

对于一张新图片，KNN 计算它与训练集中图片的距离，找到最近的 3 个样本，再根据这 3 个邻居的标签投票，决定预测类别。`k` 不是数字类别数量，而是参与投票的邻居数量。

### 7.5 为什么要划分测试集？

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y
)
```

训练集用于拟合模型；测试集用于检验模型对未参与训练的数据的预测表现。`test_size=0.2` 表示 20% 数据用于测试；`random_state=1` 固定随机拆分结果；`stratify=y` 尽量维持拆分前后各数字类别的占比。

## 8. 常见问题排查

| 现象 | 常见原因 | 检查方式 / 处理办法 |
|---|---|---|
| `FileNotFoundError` | CSV、图片或模型路径不正确 | 检查文件是否放在 `data/`、`models/` 目录；先训练再预测 |
| 提示 CSV 列数不对 | 不是“1 列标签 + 784 列像素”，或多出索引列 | 检查 `df.shape`，确认是否有额外的 `Unnamed: 0` 列 |
| 图片尺寸报错 | 输入图片不是 28×28 | 先将数字裁剪、缩放到 28×28；缩放时尽量保持数字比例 |
| 预测经常错误 | 图片背景/笔画黑白方向与训练集不同，或书写风格差异明显 | 检查灰度图；必要时使用 `--invert`；比较居中程度、笔画粗细和像素范围 |
| `X does not have valid feature names` | 训练时输入 DataFrame，预测时输入 NumPy 数组 | 本项目会按模型保存的 `feature_names_in_` 创建 DataFrame，保持特征名和顺序一致 |
| 测试准确率高，但自写数字识别差 | 测试集与自写图片分布不同 | 尝试收集更多自写样本；统一裁剪、居中、缩放、线条粗细和反色规则 |

### 特别说明：当前 Notebook 的预处理

你修改后的 Notebook 已经在训练数据上执行 `X = X / 255`。对于普通 PNG，`plt.imread()` 常会返回 0～1 的浮点像素值；如果图像确实为 28×28 灰度图且黑白方向一致，这与训练后的数值尺度相符。因此不能再机械地对已经是 0～1 的 PNG 除以 255，否则会把数值缩小 255 倍。整理版改用 Pillow 读取原始 0～255 灰度值，再统一除以 255，避免不同图片读取方式带来的歧义。

Notebook 原先的特征名称警告是因为训练输入是带列名的 DataFrame，而预测输入是 NumPy 数组。这个警告通常不是识别错误的直接原因，但应修正；本项目预测脚本会对齐列名。

## 9. 上传到 GitHub

在项目根目录打开终端：

```bash
git init
git add README.md requirements.txt .gitignore data/README.md data/demo.png notebooks/ src/
git status
git commit -m "Add KNN handwritten digit recognition tutorial"
git branch -M main
```

然后在 GitHub 网页创建一个空仓库（不要勾选自动生成 README），将下面命令中的用户名和仓库名换成你自己的：

```bash
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

推送前请查看 `git status`，确认没有意外加入训练 CSV、虚拟环境或模型文件。`.gitignore` 会默认忽略这些文件，但仍建议检查一次。

## 10. 可以继续练习的方向

- 把 `n_neighbors` 从 1、3、5、7、9 分别运行一次，比较测试集准确率；
- 比较不同距离度量方式，例如欧氏距离与曼哈顿距离；
- 绘制混淆矩阵，找出最容易互相混淆的数字；
- 收集自己的手写数字，统一处理成 28×28 后，建立一份独立测试集；
- 在理解 KNN 后，再尝试决策树、支持向量机（SVM）或卷积神经网络（CNN），比较不同模型的优缺点。

---

**学习建议：** 第一次运行时先从 Notebook 开始，逐个单元格执行并观察 `shape`、像素范围和预测结果；熟悉流程后，再尝试通过 `src/train.py` 和 `src/predict.py` 独立运行项目。
