# A 股市场风险与因子实证（Python 复现）

三个课程作业的 Python 重写与扩展。原始代码用 Stata 写成（`use` / `append` / `statsby` / `ttest` 那套），
这里改成可读、可复现的 pandas + statsmodels 流程，并把每个作业**往前推了一步**：
作业停在「检验出结论」，这里继续问「这个结论对风险计量意味着什么」。

| 项目 | 回答的问题 | 主要方法 |
|---|---|---|
| `p1_market_heat.py` | 2015 / 2021 / 2026 三轮行情，这一轮到底有多热？ | 6 个跨期可比的倍数指标，统一以行情起点归一 |
| `p2_distribution.py` | 收益率不服从正态，对 VaR 意味着什么？ | 偏度-峰度检验、Jarque-Bera、Shapiro-Wilk；正态法 / 历史模拟法 / Cornish-Fisher 三种 VaR 对比 |
| `p2_capm.py` | 个股 Beta 能不能直接用来设风险限额？ | 个股 β 估计 → 50 只等权组合 → 组合 β 重估 → 截面 SML 检验 |
| `p3_factors.py` | 规模和动量因子在 A 股是否成立？ | 滞后市值十分组、滞后收益五分组；市场模型与 FF3 回归 |

## 目录

```
config.py            路径、样本窗口、分组数等所有可调参数
data_io.py           数据读取层（CSMAR dta → DataFrame），含文件名映射
demo_data.py         模拟数据生成器，没有真实数据也能跑通全流程
p1_market_heat.py    项目一
p2_distribution.py   项目二·上
p2_capm.py           项目二·下
p3_factors.py        项目三
run_all.py           一键运行
output/              真实数据跑出来的表格（csv）与图（png）
output/demo/         模拟数据的输出（仅用于验证流程，数值没有经济含义）
```

## 快速开始

```bash
pip install -r requirements.txt

# 1) 先不接数据，用模拟数据确认环境没问题（约 3 秒跑完）
python run_all.py --demo

# 2) 把 CSMAR 导出的 dta 放进 config.DATA_DIR，然后跑真实数据
python run_all.py
```

只想跑其中一个：

```bash
python p2_capm.py --demo
python p3_factors.py
```

## 数据准备

把 CSMAR 导出的 dta 文件放进 `config.py` 里的 `DATA_DIR`（默认 `C:\Users\ASUS\Desktop\data`）。
`data_io.py` 用通配符找文件，**文件名含下列关键字即可**，不用改名，多个同类文件会自动按行拼接：

| 数据 | 文件名关键字 | 需要的列 |
|---|---|---|
| 全市场日成交额 | `日交易额` | `Trddt`, `Dnvaltrdtl` |
| 融资融券余额 | `余额` | `Mtdate`, `Mtbalance` |
| 新发基金份额 | `fund shares` | `Date`, `Scale` |
| 沪深 300 估值 | `CSI300` + `PE` | `Date`, `CSI300_PE`, `CSI300_DividendYield` |
| 国债收益率 | `国债` | `Trddt`, `Yeartomatu`, `Yield` |
| 指数日行情 | `指数` | `Indexcd`, `Idxtrd01`, `Idxtrd05` |
| 个股周收益 | `周个股` | `Stkcd`, `Trdwnt`, `Wretnd` |
| 周度无风险利率 | `riskfree` | `tw`, `riskfree_w` |
| 个股月行情 | `月个股` | `Stkcd`, `Trdmnt`, `Msmvosd`, `Mretnd` |
| 三因子 | `因子` | `TradingMonth`, `RiskPremium1`, `SMB1`, `HML1` |
| 个股日价格 | `representative price` | `Date`, `stk_code`, `Price` |

> **不要把原始数据提交到公开仓库。** CSMAR 数据受授权协议约束，公开上传属于违约。
> 仓库里只需要代码和生成的结果，数据放在本地并加进 `.gitignore`。

## 结果示意

以下图片来自 `--demo` 模拟数据，只用于展示输出形态：

![CAPM 截面检验](output/demo/p2_CAPM%E6%88%AA%E9%9D%A2%E6%A3%80%E9%AA%8C.png)

![收益率分布](output/demo/p2_%E6%94%B6%E7%9B%8A%E7%8E%87%E5%88%86%E5%B8%83.png)

## 口径说明

写这类实证最容易出问题的地方不在回归，而在口径。这里几个关键选择：

- **分组只用滞后一期信息。** 用当月市值决定当月持仓，会引入前视偏差，规模效应会被系统性高估。
  因此 `lag_mv` / `lag_ret` 都用 `shift(1)`，并且要求月份**连续**（隔月断档的样本直接丢弃，
  不能用 shift 跨断档补齐）。
- **剔除创业板与科创板。** 涨跌幅限制、上市门槛、行业结构都与主板差异太大，混在一起会污染因子方向。
- **周收益剔除 |r| > 30%。** 这个阈值挡的是数据错误和停复牌衔接问题，不是真实的极端行情。
- **CAPM 的估计期与检验期各占一半。** 用同一段样本估 β 再在同一段样本上检验 SML，
  等于让模型自己给自己打分。
- **VaR 用月度数据时，99% 分位只由约 2 个观测决定。** 稳健性弱于 95%；
  实务上做 99% VaR 要用日频或周频数据。

## 已知局限

- `p2_capm.py` 的市场收益率用全样本个股等权平均代理，没有用流通市值加权，
  小盘股权重偏高，β 会有系统性偏移。
- `p3_factors.py` 用普通 OLS 标准误，没有做 Newey-West 调整；
  月度因子收益存在异方差和自相关时 t 值会偏大。
- 组合是等权，不是市值加权，因此多空收益里含一部分小盘股流动性溢价。
- `p1` 的 2021 轮窗口在 `config.PERIODS` 里是估计值，用的时候替换成你自己报告里的口径。

## 环境

Python 3.10+，依赖见 `requirements.txt`。
