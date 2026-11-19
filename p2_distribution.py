"""
项目二·上：沪深 300 收益率分布检验与 VaR 方法比较

对应作业：HW3 Q1。作业原来只做到「检验出非正态」为止，
这里往前推一步——非正态对风险计量意味着什么：
    同样一条收益率序列，正态法、历史模拟法、Cornish-Fisher 修正法
    算出的 99% VaR 差多少？差出来的部分就是尾部风险被低估的量。

运行：python p2_distribution.py [--demo]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
import data_io

plt.rcParams["font.sans-serif"] = [config.PLOT_FONT, "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

INDEX_CODE = "000300"


def monthly_returns(demo: bool) -> pd.Series:
    """沪深 300 月度收益率：取每月最后一个交易日的收盘价环比。"""
    idx = data_io.load("index", demo)
    idx = idx[idx["Indexcd"].astype(str).str.zfill(6) == INDEX_CODE].copy()
    idx["date"] = pd.to_datetime(idx["Idxtrd01"], format="%Y-%m-%d", errors="coerce")
    idx = idx.dropna(subset=["date", "Idxtrd05"]).drop_duplicates("date", keep="last")
    px = idx.set_index("date")["Idxtrd05"].sort_index()

    month_end = px.groupby(px.index.to_period("M")).last()
    return month_end.pct_change().dropna()


def normality_tests(r: pd.Series) -> pd.DataFrame:
    """三种正态性检验。Stata 的 sktest 对应这里的 D'Agostino K²（偏度-峰度联合检验）。"""
    jb_stat, jb_p = stats.jarque_bera(r)
    k2_stat, k2_p = stats.normaltest(r)
    sw_stat, sw_p = stats.shapiro(r)
    return pd.DataFrame(
        [
            ["观测数 N", f"{len(r)}", ""],
            ["偏度 Skewness", f"{stats.skew(r, bias=False):.4f}", "正态为 0"],
            ["峰度 Kurtosis", f"{stats.kurtosis(r, fisher=False, bias=False):.4f}", "正态为 3"],
            ["Jarque-Bera", f"{jb_stat:.3f}", f"p = {jb_p:.3e}"],
            ["D'Agostino K²（≈ sktest）", f"{k2_stat:.3f}", f"p = {k2_p:.3e}"],
            ["Shapiro-Wilk", f"{sw_stat:.4f}", f"p = {sw_p:.3e}"],
        ],
        columns=["检验", "统计量", "备注"],
    )


def cornish_fisher_var(mu: float, sigma: float, s: float, k: float, alpha: float) -> float:
    """
    Cornish-Fisher 修正 VaR：在正态分位数上用偏度、峰度做展开，
    把「尾部比正态更厚」这件事直接修进分位数里。
    s = 偏度, k = 超额峰度（正态为 0）
    """
    z = stats.norm.ppf(alpha)
    z_cf = (
        z
        + (z ** 2 - 1) * s / 6
        + (z ** 3 - 3 * z) * k / 24
        - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    )
    return -(mu + z_cf * sigma)


def var_table(r: pd.Series, levels=(0.95, 0.99)) -> pd.DataFrame:
    """三种方法、两个置信水平下的 VaR，以及历史模拟法的期望损失 ES。"""
    mu, sigma = r.mean(), r.std(ddof=1)
    s = stats.skew(r, bias=False)
    k = stats.kurtosis(r, fisher=True, bias=False)  # 超额峰度

    rows = []
    for lv in levels:
        alpha = 1 - lv
        var_hist = -np.quantile(r, alpha)
        var_norm = -(mu + stats.norm.ppf(alpha) * sigma)
        var_cf = cornish_fisher_var(mu, sigma, s, k, alpha)
        tail = r[r <= -var_hist]
        es = -tail.mean() if len(tail) else np.nan

        rows.append(
            {
                "置信水平": f"{lv:.0%}",
                "正态法 VaR": var_norm,
                "历史模拟法 VaR": var_hist,
                "Cornish-Fisher VaR": var_cf,
                "正态法低估(百分点)": (var_hist - var_norm) * 100,
                "正态法低估比例": var_hist / var_norm - 1 if var_norm else np.nan,
                "历史模拟法 ES": es,
            }
        )
    return pd.DataFrame(rows)


def plot_distribution(r: pd.Series, path_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.hist(r, bins=30, density=True, color="#c8d8ea", edgecolor="#5a7391", label="实际分布")

    x = np.linspace(r.min(), r.max(), 400)
    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std(ddof=1)), color="#b3261e", lw=1.8,
            label="正态拟合")
    ax.set_title(f"沪深 300 月度收益率分布（N={len(r)}，偏度 {stats.skew(r, bias=False):.2f}，"
                 f"峰度 {stats.kurtosis(r, fisher=False, bias=False):.2f}）", fontsize=11)
    ax.set_xlabel("月度收益率")
    ax.set_ylabel("密度")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    data_io.save_figure(fig, "p2_收益率分布")
    plt.close(fig)


def run(demo: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_io.set_demo(demo)
    print("=" * 68)
    print("项目二·上：沪深 300 收益分布检验与 VaR 方法比较" + ("（模拟数据）" if demo else ""))
    print("=" * 68)

    r = monthly_returns(demo)
    print(f"\n样本：{r.index.min():%Y-%m} 至 {r.index.max():%Y-%m}，月度收益 {len(r)} 个")

    tests = normality_tests(r)
    print("\n【正态性检验】")
    print(tests.to_string(index=False))
    data_io.save_table(tests, "p2_正态性检验")

    vt = var_table(r)
    print("\n【VaR 方法比较】(月度，持有期 1 个月)")
    print(vt.round(4).to_string(index=False))
    print("  注：99% 分位在月度样本里只由约 2 个观测决定，稳健性弱于 95%；")
    print("      实务上做 99% VaR 一般用日频或周频数据，样本量才不会太薄。")
    data_io.save_table(vt, "p2_VaR比较")

    plot_distribution(r)
    return tests, vt


if __name__ == "__main__":
    run(demo="--demo" in sys.argv)
