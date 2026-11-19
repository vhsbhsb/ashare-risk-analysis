"""
项目一：A 股三轮牛市的市场过热与杠杆指标监测

对应作业：HW2（Q1–Q7）。把 2015 / 2021 / 2026 三轮行情放在同一把尺子上，
用 6 个可比的倍数指标回答一个问题——这一轮到底有多热，杠杆和资金追高到什么程度。

指标口径（全部以「行情起点」为基准归一，跨期才可比）：
    1. 成交额倍数       期末成交额 / 期初成交额
    2. 成交额峰值倍数   区间最大成交额 / 期初成交额
    3. 两融余额倍数     期末融资融券余额 / 期初余额
    4. 新发基金扩张倍数 期末当月新发权益基金份额 / 期初当月
    5. ERP 压缩幅度     1/PE − 10 年期国债收益率的区间变化
    6. 股债利差压缩幅度 沪深 300 股息率 − 10 年期国债收益率的区间变化

运行：python p1_market_heat.py [--demo]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data_io

plt.rcParams["font.sans-serif"] = [config.PLOT_FONT, "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

STOCK_LABELS = {"300059": "东方财富", "600570": "恒生电子", "002670": "国盛金控", "600030": "中信证券"}


# ---------------------------------------------------------------- 数据准备
def daily_series(demo: bool) -> dict[str, pd.Series]:
    """把各张日频表整理成「按日期索引的单变量序列」。"""
    turnover = data_io.load("turnover", demo)
    turnover["date"] = pd.to_datetime(turnover["Trddt"], format="%Y-%m-%d", errors="coerce")
    ts_turnover = turnover.dropna(subset=["date"]).groupby("date")["Dnvaltrdtl"].sum().sort_index()

    margin = data_io.load("margin", demo)
    margin["date"] = pd.to_datetime(margin["Mtdate"], format="%Y-%m-%d", errors="coerce")
    ts_margin = margin.dropna(subset=["date"]).groupby("date")["Mtbalance"].sum().sort_index()

    fund = data_io.load("fund_shares", demo)
    fund["date"] = pd.to_datetime(fund["Date"], format="%Y-%m-%d", errors="coerce")
    fund = fund.dropna(subset=["date"])
    ts_fund = fund.groupby(fund["date"].dt.to_period("M"))["Scale"].sum().sort_index()
    # 统一成时间戳（取月末），后面才能和日频序列用同一套窗口函数
    ts_fund.index = ts_fund.index.to_timestamp(how="end").normalize()

    return {"turnover": ts_turnover, "margin": ts_margin, "fund": ts_fund}


def index_close(demo: bool) -> pd.Series:
    idx = data_io.load("index", demo)
    idx = idx[idx["Indexcd"].astype(str).str.zfill(6) == "000300"].copy()
    idx["date"] = pd.to_datetime(idx["Idxtrd01"], format="%Y-%m-%d", errors="coerce")
    idx = idx.dropna(subset=["date", "Idxtrd05"]).drop_duplicates("date", keep="last")
    return idx.set_index("date")["Idxtrd05"].sort_index()


def valuation_spreads(demo: bool) -> pd.DataFrame:
    """ERP 与股债利差，都是日频。"""
    bond = data_io.load("bond", demo)
    bond["date"] = pd.to_datetime(bond["Trddt"], format="%Y-%m-%d", errors="coerce")
    bond = bond.dropna(subset=["date"])
    bond = bond[bond["Yeartomatu"] == config.BOND_TENOR]
    ts_bond = bond.groupby("date")["Yield"].mean() / 100.0  # 百分数 -> 小数

    val = data_io.load("csi300_val", demo)
    val["date"] = pd.to_datetime(val["Date"], format="%Y-%m-%d", errors="coerce")
    val = val.dropna(subset=["date"]).drop_duplicates("date")
    val = val.set_index("date")[["CSI300_PE", "CSI300_DividendYield"]]

    df = val.join(ts_bond.rename("bond_yield"), how="inner").sort_index()
    df["erp"] = 1.0 / df["CSI300_PE"] - df["bond_yield"]
    df["spread"] = df["CSI300_DividendYield"] - df["bond_yield"] * 100  # 股息率是百分数
    return df


# ---------------------------------------------------------------- 单个指标
def _window(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return series.loc[(series.index >= start) & (series.index <= end)]


def _at(series: pd.Series, date: pd.Timestamp, side: str = "start") -> float:
    """取 date 当天的值；当天不是交易日时，起点往后找、终点往前找最近的一个。"""
    if side == "start":
        hit = series.loc[series.index >= date]
        return float(hit.iloc[0]) if len(hit) else np.nan
    hit = series.loc[series.index <= date]
    return float(hit.iloc[-1]) if len(hit) else np.nan


def _start_end(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float]:
    return _at(series, start, "start"), _at(series, end, "end")


def turnover_stats(ts: pd.Series, start, end) -> dict:
    win = _window(ts, start, end)
    if win.empty:
        return {}
    v0, v1 = float(win.iloc[0]), float(win.iloc[-1])
    return {
        "成交额倍数(期末/期初)": v1 / v0 if v0 else np.nan,
        "成交额峰值倍数(峰值/期初)": float(win.max()) / v0 if v0 else np.nan,
        "成交额峰值日": win.idxmax().strftime("%Y-%m-%d"),
    }


def peak_response(ts: pd.Series, px: pd.Series, start, end) -> dict:
    """成交额峰值前后，指数各自涨了多少。"""
    win = _window(ts, start, end)
    if win.empty:
        return {}
    peak = win.idxmax()
    p0 = _at(px, start, "start")
    ppk = _at(px, peak, "end")
    p1 = _at(px, end, "end")
    if not p0 or np.isnan(p0) or np.isnan(ppk):
        return {"峰值前涨幅": np.nan, "峰值后涨幅": np.nan}
    return {
        "峰值前涨幅": ppk / p0 - 1,
        "峰值后涨幅": (p1 / ppk - 1) if (not np.isnan(p1)) else np.nan,
    }


def ratio_stat(ts: pd.Series, start, end, name: str) -> dict:
    v0, v1 = _start_end(ts, start, end)
    if np.isnan(v0) or np.isnan(v1) or v0 == 0:
        return {name: np.nan}
    return {name: v1 / v0}


def change_stat(df: pd.DataFrame, col: str, start, end, label: str) -> dict:
    s = df[col].dropna()
    v0, v1 = _start_end(s, start, end)
    if np.isnan(v0) or v1 != v1:
        return {f"{label}变化": np.nan, f"{label}百分比变化": np.nan}
    return {
        f"{label}变化": v1 - v0,
        f"{label}百分比变化": (v1 - v0) / abs(v0) if v0 else np.nan,
    }


def stock_returns(demo: bool, start, end) -> pd.Series:
    px = data_io.load("price", demo)
    px["date"] = pd.to_datetime(px["Date"], format="%Y-%m-%d", errors="coerce")
    px = px.dropna(subset=["date", "Price"])
    px["code"] = px["stk_code"].astype(str).str.zfill(6)
    px = px.drop_duplicates(["code", "date"])

    out = {}
    for code, g in px.groupby("code"):
        s = g.set_index("date")["Price"].sort_index()
        v0, v1 = _start_end(s, start, end)
        if not np.isnan(v0) and v0:
            out[code] = v1 / v0 - 1
    return pd.Series(out).sort_values(ascending=False)


# ---------------------------------------------------------------- 主流程
def run(demo: bool = False) -> pd.DataFrame:
    data_io.set_demo(demo)
    print("=" * 68)
    print("项目一：三轮牛市的市场过热与杠杆指标监测" + ("（模拟数据）" if demo else ""))
    print("=" * 68)

    ts = daily_series(demo)
    px = index_close(demo)
    spreads = valuation_spreads(demo)

    rows = {}
    for label, (s, e) in config.PERIODS.items():
        if s is None or e is None:
            print(f"\n[{label}] 窗口未设置，跳过（在 config.PERIODS 里填上即可）")
            continue
        start, end = pd.Timestamp(s), pd.Timestamp(e)
        r = {}
        r.update(turnover_stats(ts["turnover"], start, end))
        r.update(peak_response(ts["turnover"], px, start, end))
        r.update(ratio_stat(ts["margin"], start, end, "两融余额倍数"))
        r.update(ratio_stat(ts["fund"], start, end, "新发基金扩张倍数"))
        r.update(change_stat(spreads, "erp", start, end, "ERP"))
        r.update(change_stat(spreads, "spread", start, end, "股债利差"))

        sr = stock_returns(demo, start, end)
        for code in ("300059", "600570"):
            if code in sr.index:
                r[f"个股涨幅[{STOCK_LABELS.get(code, code)}]"] = sr[code]
        if len(sr):
            r["个股涨幅[最佳个股]"] = sr.iloc[0]

        rows[label] = r
        print(f"\n[{label}] {s} → {e}")
        for k, v in r.items():
            if isinstance(v, str):
                print(f"  {k:<26s} {v}")
            else:
                print(f"  {k:<26s} {v:.4f}")

    table = pd.DataFrame(rows).T
    if not table.empty:
        data_io.save_table(table.reset_index().rename(columns={"index": "行情"}), "p1_过热指标汇总")

    _plot_latest(ts, px, demo)
    return table


def _plot_latest(ts: dict, px: pd.Series, demo: bool):
    """把时间上最新的一轮行情画出来：成交额、两融余额、新发基金。"""
    labels = [k for k, (s, e) in config.PERIODS.items() if s and e]
    if not labels:
        return
    latest = labels[-1]
    s, e = config.PERIODS[latest]
    start, end = pd.Timestamp(s), pd.Timestamp(e)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))

    for ax, key, title, unit in (
        (axes[0], "turnover", "全市场成交额", "亿元"),
        (axes[1], "margin", "融资融券余额", "亿元"),
    ):
        w = _window(ts[key], start, end)
        ax.plot(w.index, w.values, lw=1.2, color="#b3261e")
        ax.set_title(f"{latest} 年行情：{title}（{s} 至 {e}）", fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)

    w = _window(ts["fund"], start, end)
    axes[2].bar(w.index.astype(str), w.values, color="#2b579a", width=0.6)
    axes[2].set_title(f"{latest} 年行情：月度新发权益基金份额", fontsize=11)
    axes[2].set_ylabel("亿份")
    axes[2].tick_params(axis="x", rotation=45, labelsize=7)

    fig.tight_layout()
    data_io.save_figure(fig, f"p1_{latest}行情指标")
    plt.close(fig)


if __name__ == "__main__":
    run(demo="--demo" in sys.argv)
