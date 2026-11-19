"""
模拟数据生成器（仅在 demo=True 时使用）。

目的：让你在没有把数据拷到电脑上、或者换了台电脑的时候，
也能把整条流程跑通、看到输出长什么样。

生成的数据结构与 CSMAR 导出的 dta 完全一致（列名、日期格式都对齐），
所以真实数据接进来时，分析代码一行都不用改。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

RNG = np.random.default_rng(20260418)

# 月度因子序列单独用一条随机流生成，个股收益再引用它，
# 这样市场模型/FF3 回归出来的 β 才会接近 1（否则模拟数据里 β≈0，容易以为代码错了）
_MONTHS = pd.date_range("2001-01-31", periods=300, freq="ME")
_FRNG = np.random.default_rng(7)
_MKTRF = _FRNG.normal(0.006, 0.07, len(_MONTHS))
_SMB = _FRNG.normal(0.002, 0.04, len(_MONTHS))
_HML = _FRNG.normal(0.001, 0.035, len(_MONTHS))


def _daily(start="2013-01-04", end="2026-03-31") -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _date_str(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.strftime("%Y-%m-%d"))


# ---------------------------------------------------------------- 市场级日频
def _turnover() -> pd.DataFrame:
    idx = _daily()
    # 成交额：缓升趋势 + 两轮牛市脉冲 + 对数正态噪声
    t = np.arange(len(idx)) / len(idx)
    level = 3000 * (1 + 1.5 * t)
    bull = (
        4.0 * np.exp(-((np.arange(len(idx)) - int(len(idx) * 0.18)) ** 2) / (2 * 60 ** 2))
        + 2.0 * np.exp(-((np.arange(len(idx)) - int(len(idx) * 0.72)) ** 2) / (2 * 80 ** 2))
    )
    value = level * (1 + bull) * RNG.lognormal(0, 0.25, len(idx))
    return pd.DataFrame({"Trddt": _date_str(idx), "Dnvaltrdtl": value})


def _margin() -> pd.DataFrame:
    idx = _daily()
    t = np.arange(len(idx)) / len(idx)
    level = 8000 * (1 + 0.8 * t)
    bull = 2.5 * np.exp(-((np.arange(len(idx)) - int(len(idx) * 0.18)) ** 2) / (2 * 70 ** 2))
    value = level * (1 + bull) * RNG.lognormal(0, 0.08, len(idx))
    return pd.DataFrame({"Mtdate": _date_str(idx), "Mtbalance": value})


def _fund_shares() -> pd.DataFrame:
    idx = _daily()
    t = np.arange(len(idx)) / len(idx)
    bull = 1 + 3.0 * np.exp(-((np.arange(len(idx)) - int(len(idx) * 0.20)) ** 2) / (2 * 90 ** 2))
    value = 200 * (1 + 0.5 * t) * bull * RNG.lognormal(0, 0.4, len(idx))
    return pd.DataFrame({"Date": _date_str(idx), "Scale": value})


def _csi300_val() -> pd.DataFrame:
    idx = _daily()
    n = len(idx)
    t = np.arange(n) / n
    # PE 在 9–30 之间波动，牛市段被推高
    pe = 12 + 4 * np.sin(2 * np.pi * t * 1.5) + 8 * np.exp(
        -((np.arange(n) - int(n * 0.72)) ** 2) / (2 * 120 ** 2)
    ) + RNG.normal(0, 0.4, n)
    div_yield = 3.5 - 0.09 * (pe - 12) + RNG.normal(0, 0.05, n)
    return pd.DataFrame(
        {"Date": _date_str(idx), "CSI300_PE": np.clip(pe, 5, None), "CSI300_DividendYield": np.clip(div_yield, 0.2, None)}
    )


def _bond() -> pd.DataFrame:
    idx = _daily()
    rows = []
    for tenor in (1, 3, 5, 10):
        base = 2.2 + 0.06 * tenor
        yield_ = base + 0.35 * np.sin(np.arange(len(idx)) / 400) + RNG.normal(0, 0.02, len(idx))
        rows.append(pd.DataFrame({"Trddt": _date_str(idx), "Yeartomatu": tenor, "Yield": yield_}))
    return pd.concat(rows, ignore_index=True)


def _index() -> pd.DataFrame:
    idx = _daily()
    n = len(idx)
    # 用 t 分布造日收益，让模拟数据带上真实的尖峰厚尾特征
    df_t = 4
    innov = RNG.standard_t(df_t, n) * np.sqrt((df_t - 2) / df_t)
    ret = 0.0003 + 0.011 * innov
    close = 3000 * np.exp(np.cumsum(ret))
    return pd.DataFrame(
        {"Indexcd": "000300", "Idxtrd01": _date_str(idx), "Idxtrd05": close}
    )


def _price() -> pd.DataFrame:
    idx = _daily()
    rows = []
    for code, drift in (("300059", 0.0012), ("600570", 0.0008), ("002670", 0.0006), ("600030", 0.0005)):
        ret = RNG.normal(drift, 0.022, len(idx))
        rows.append(
            pd.DataFrame(
                {"Date": _date_str(idx), "stk_code": code, "Price": 20 * np.exp(np.cumsum(ret))}
            )
        )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 个股面板
def _stock_weekly(n_stocks: int = 400, n_weeks: int = 600) -> pd.DataFrame:
    weeks = pd.date_range("2013-01-04", periods=n_weeks, freq="W-FRI")
    week_id = pd.Series(weeks.strftime("%Y-%m-%d"))
    r_m = RNG.normal(0.0015, 0.032, n_weeks)

    betas = RNG.lognormal(0, 0.28, n_stocks)
    alphas = RNG.normal(0.0002, 0.0004, n_stocks)

    ret = (
        alphas[:, None]
        + betas[:, None] * r_m[None, :]
        + RNG.normal(0, 0.030, (n_stocks, n_weeks))
    )

    codes = [("60%04d" % i) if i % 2 == 0 else ("00%04d" % i) for i in range(n_stocks)]
    df = pd.DataFrame(
        {
            "Stkcd": np.repeat(codes, n_weeks),
            "Trdwnt": np.tile(week_id.values, n_stocks),
            "Wretnd": ret.ravel(),
        }
    )
    # 模拟少量异常值（真实 CSMAR 里也会有数据错误，作业口径是剔除 |r|>30%）
    bad = RNG.random(len(df)) < 0.001
    df.loc[bad, "Wretnd"] = RNG.choice([-1.5, 1.8], bad.sum())
    return df


def _riskfree_weekly(n_weeks: int = 600) -> pd.DataFrame:
    weeks = pd.date_range("2013-01-04", periods=n_weeks, freq="W-FRI")
    return pd.DataFrame(
        {
            "tw": weeks.strftime("%Y-%m-%d"),
            "riskfree_w": RNG.normal(0.0006, 0.00005, n_weeks),
        }
    )


def _stock_monthly(n_stocks: int = 800, n_months: int = 300) -> pd.DataFrame:
    months = _MONTHS
    month_id = pd.Series(months.strftime("%Y-%m"))

    size = RNG.lognormal(mean=13.5, sigma=1.4, size=(n_stocks, 1))
    size = np.tile(size, (1, n_months)) * RNG.lognormal(0, 0.06, (n_stocks, n_months))

    # 让数据里同时含规模效应与短期反转，方便验证两种因子的方向；
    # 收益里带上市场因子，这样回归出来的 β 才接近 1，不会让人误以为代码写错
    log_size = np.log(size)
    z_size = (log_size - log_size.mean()) / log_size.std()
    beta = RNG.normal(1.0, 0.25, (n_stocks, 1))
    eps = RNG.normal(0, 0.09, (n_stocks, n_months))
    # 加一点短期反转（上个月的特异冲击这个月部分回吐），让动量分组跑出负的多空收益
    prev_eps = np.concatenate([np.zeros((n_stocks, 1)), eps[:, :-1]], axis=1)
    ret = beta * _MKTRF[None, :] - 0.004 * z_size + eps - 0.06 * prev_eps

    codes = [("60%04d" % i) if i % 2 == 0 else ("00%04d" % i) for i in range(n_stocks)]
    return pd.DataFrame(
        {
            "Stkcd": np.repeat(codes, n_months),
            "Trdmnt": np.tile(month_id.values, n_stocks),
            "Msmvosd": size.ravel(),
            "Mretnd": ret.ravel(),
        }
    )


def _factors(n_months: int = 300) -> pd.DataFrame:
    months = _MONTHS
    return pd.DataFrame(
        {
            "TradingMonth": months.strftime("%Y-%m"),
            "RiskPremium1": _MKTRF,
            "SMB1": _SMB,
            "HML1": _HML,
        }
    )


_BUILDERS = {
    "turnover": _turnover,
    "margin": _margin,
    "fund_shares": _fund_shares,
    "csi300_val": _csi300_val,
    "bond": _bond,
    "index": _index,
    "price": _price,
    "stock_weekly": _stock_weekly,
    "riskfree_weekly": _riskfree_weekly,
    "stock_monthly": _stock_monthly,
    "factors": _factors,
}

_CACHE: dict[str, pd.DataFrame] = {}


def make(key: str) -> pd.DataFrame:
    """返回模拟数据；同一进程内重复调用会复用缓存，避免重复生成大表。"""
    if key not in _BUILDERS:
        raise KeyError(f"没有为 {key} 准备模拟数据")
    if key not in _CACHE:
        _CACHE[key] = _BUILDERS[key]()
    return _CACHE[key].copy()
