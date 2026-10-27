# RFM 会员价值分析项目

基于 Python、Pandas、NumPy 与 Pyecharts 的会员价值分析项目，可用于数据分析学习、项目复现和 GitHub 作品集展示。

## 项目简介

本项目以 2015—2018 年订单数据为例，使用 RFM 模型从最近消费时间（Recency）、消费频率（Frequency）和消费金额（Monetary）三个维度构建会员年度画像，完成数据清洗、RFM 指标计算、评分分群、年度对比、交互式可视化和结果导出。

## 主要工作

1. 读取 Excel 中的年度订单工作表并检查数据质量。
2. 转换日期、金额字段，处理分析必要字段缺失记录并筛选有效订单。
3. 按“年份 + 会员 ID”计算 R、F、M。
4. 将 R、F、M 划分为三档评分并生成三位分群编码。
5. 比较年度会员记录数与各 RFM 分群的结构变化。
6. 使用 Pyecharts 输出年度趋势、分群人数对比、最近一年分群占比图。
7. 导出 CSV、Excel 和 HTML 图表；MySQL 导出为可选扩展。

## RFM 指标

| 指标 | 含义 | 评分方向 |
|---|---|---|
| R | 距离最近一次消费的天数 | 越短越好 |
| F | 年度订单号非空记录数 | 越多越好 |
| M | 年度消费金额总和 | 越高越好 |

项目使用三档评分。`333` 表示三项评分均为 3，`111` 表示三项评分均为 1。评分阈值沿用案例中的设置，应根据真实数据分布和业务目标复核。

## 数据准备

将 Excel 文件放在 `data/sales.xlsx`，也可放在项目根目录。文件应包含 `2015`、`2016`、`2017`、`2018` 四个工作表，每个订单工作表至少包含：

- `提交日期`
- `会员ID`
- `订单号`
- `订单金额`

原案例中的 `会员等级` 工作表不参与主分析流程。仓库不附带原始数据，请仅使用自己有权使用的数据。

## 环境配置

建议 Python 3.10—3.12。

macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Windows：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

打开 `RFM会员价值分析.ipynb`，从上到下运行单元格。

## 项目结构

```text
RFM_GitHub_Project/
├── RFM会员价值分析.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── outputs/                 # 运行后生成，不提交
```

## 输出结果

运行后会在 `outputs/` 生成：

- `rfm_member_scores.csv`：会员年度 RFM 指标和评分。
- `rfm_group_year_summary.csv`：年度分群统计。
- `year_summary.csv`：年度总体指标。
- `RFM分析结果.xlsx`：汇总工作簿。
- `01_年度会员记录数趋势.html`
- `02_RFM分群人数对比.html`
- `03_最近一年RFM分群占比.html`

HTML 图表可使用浏览器打开。

## 分析口径与局限

- 每年的统计节点为该年度清洗后订单中的最大提交日期，不一定是自然年 12 月 31 日。
- 年度记录按“年份 + 会员 ID”统计，同一会员可能在多个年份重复出现。
- F 指标当前按订单号非空记录数计算。如果源表是一张订单多行商品明细，应检查数据粒度，必要时改为订单号去重计数。
- RFM 仅描述历史交易特征，不能单独证明营销措施的因果效果。

## GitHub 公开前安全检查

- 不上传原始 Excel、真实会员信息或未授权业务数据。
- 不在 Notebook 中硬编码或展示数据库密码。
- 检查 Notebook 输出，避免暴露会员 ID、订单明细等敏感信息。
- 确认有权公开所有提交的材料。
