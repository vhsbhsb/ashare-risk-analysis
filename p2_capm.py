"""
项目二·下：A 股个股 Beta 估计与 CAPM 截面检验

对应作业：HW3 Q2。三步：
    1. 估计期：逐只个股跑 r_it = α_i + β_i · r_mt，取得 β_i 与其 t 值
    2. 组合期：按 β_i 排序，每 50 只等权构造一个组合（作业里是 70 组），
       组合层面重估 β_p——异质性噪声被分散后，β 的估计精度会明显改善
    3. 检验期：把各组合的平均超额收益对 β_p 做截面回归，看 SML 的斜率与截距

为什么要分组：单只股票的 R² 很低、β 的 t 值不稳，直接用个股 β 设风险限额会误伤；
组合层面稳定，说明 β 只能用在组合级。这条结论比「CAPM 成立」更有用。

运行：python p2_capm.py [--demo]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config
import data_io

plt.rcParams["font.sans-serif"] = [config.PLOT_FONT, "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def build_panel(demo: bool) -> tuple[pd.DataFrame, pd.Series]:
    """整理成 (Stkcd, Trdwnt, r_it, r_mt, r_ft) 的周度面板。"""
    stock = data_io.load("stock_weekly", demo)
    stock["Stkcd"] = stock["Stkcd"].astype(str).str.zfill(6)
    # 作业口径：只保留 6 开头（沪市）与 0 开头（深市主板/中小板）
    stock = stock[stock["Stkcd"].str[0].isin(["6", "0"])]
    stock = stock.rename(columns={"Wretnd": "r_it"})
    stock = stock.dropna(subset=["r_it"])
    # 剔除 |r| > 30% 的异常周（多为数据错误或停复牌衔接问题）
    stock = stock[stock["r_it"].abs() < config.RET_CLIP]
    stock = stock.drop_duplicates(["Stkcd", "Trdwnt"])

    # 等权市场收益率（用全样本个股收益的周均值代理）
    r_m = stock.groupby("Trdwnt")["r_it"].mean().rename("r_mt")

    rf = data_io.load("riskfree_weekly", demo).rename(
        columns={"tw": "Trdwnt", "riskfree_w": "r_ft"}
    )
    rf["Trdwnt"] = rf["Trdwnt"].astype(str)

    panel = stock.merge(rf[["Trdwnt", "r_ft"]], on="Trdwnt", how="inner")
    panel = panel.merge(r_m, on="Trdwnt", how="left")
    panel = panel.sort_values(["Trdwnt", "Stkcd"]).reset_index(drop=True)
    return panel, r_m


def beta_by_stock(panel: pd.DataFrame, est_weeks: set) -> pd.DataFrame:
    """
    逐只个股估计 Beta（闭式解，比循环跑 3000 次 OLS 快两个数量级）。
    beta = (n·Σrm·r − Σr·Σrm) / (n·Σrm² − (Σrm)²)
    """
    est = panel[panel["Trdwnt"].isin(est_weeks)].copy()
    est["r2"] = est["r_it"] ** 2
    est["m2"] = est["r_mt"] ** 2
    est["rm"] = est["r_it"] * est["r_mt"]

    g = est.groupby("Stkcd").agg(
        n=("r_it", "size"),
        sum_r=("r_it", "sum"),
        sum_m=("r_mt", "sum"),
        sum_rr=("r2", "sum"),
        sum_mm=("m2", "sum"),
        sum_rm=("rm", "sum"),
    )
    g = g[g["n"] >= config.MIN_WEEKS_FOR_BETA].copy()

    n = g["n"]
    mean_m = g["sum_m"] / n
    mean_r = g["sum_r"] / n
    sxx = g["sum_mm"] - n * mean_m ** 2
    sxy = g["sum_rm"] - n * mean_m * mean_r

    g["beta_i"] = sxy / sxx
    g["alpha_i"] = mean_r - g["beta_i"] * mean_m

    ssr = g["sum_rr"] - g["alpha_i"] * g["sum_r"] - g["beta_i"] * g["sum_rm"]
    sst = g["sum_rr"] - n * mean_r ** 2
    sigma2 = ssr / (n - 2)

    g["se_beta"] = np.sqrt(sigma2 / sxx)
    g["t_beta"] = g["beta_i"] / g["se_beta"]
    g["t_alpha"] = g["alpha_i"] / np.sqrt(sigma2 * (1 / n + mean_m ** 2 / sxx))
    g["R2_i"] = 1 - ssr / sst

    return g.reset_index()


def assign_groups(betas: pd.DataFrame) -> pd.DataFrame:
    """按 β 从小到大排序，每 50 只一个组合；不足 50 只的尾组丢掉（与作业一致）。"""
    b = betas.sort_values("beta_i").reset_index(drop=True)
    b["group_id"] = np.ceil((b.index + 1) / config.STOCKS_PER_PORTFOLIO).astype(int)
    counts = b.groupby("group_id")["Stkcd"].transform("size")
    return b[counts == config.STOCKS_PER_PORTFOLIO].copy()


def portfolio_series(panel: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """等权组合的周度超额收益。"""
    p = panel.merge(groups[["Stkcd", "group_id"]], on="Stkcd", how="inner")
    p["excess_it"] = p["r_it"] - p["r_ft"]
    p["mkt_excess"] = p["r_mt"] - p["r_ft"]

    port = (
        p.groupby(["group_id", "Trdwnt"])
        .agg(r_pt=("r_it", "mean"), r_pt_rf=("excess_it", "mean"), r_mt_rf=("mkt_excess", "first"))
        .reset_index()
    )
    return port


def portfolio_betas(port: pd.DataFrame, est_weeks: set) -> pd.DataFrame:
    """组合层面的时序回归：r_pt − rf = α_p + β_p · (r_mt − rf)。"""
    rows = []
    est = port[port["Trdwnt"].isin(est_weeks)]
    for gid, d in est.groupby("group_id"):
        X = sm.add_constant(d["r_mt_rf"].to_numpy())
        res = sm.OLS(d["r_pt_rf"].to_numpy(), X).fit()
        rows.append(
            {
                "group_id": gid,
                "alpha_p": res.params[0],
                "beta_p": res.params[1],
                "t_alpha_p": res.tvalues[0],
                "t_beta_p": res.tvalues[1],
                "R2_p": res.rsquared,
            }
        )
    return pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)


def cross_section(port: pd.DataFrame, pbetas: pd.DataFrame, test_weeks: set) -> dict:
    """检验期：组合平均超额收益对 β_p 回归，看 SML。"""
    test = port[port["Trdwnt"].isin(test_weeks)]
    avg = test.groupby("group_id")["r_pt_rf"].mean().rename("delta_R_pf").reset_index()
    d = avg.merge(pbetas[["group_id", "beta_p"]], on="group_id")

    X = sm.add_constant(d["beta_p"].to_numpy())
    res = sm.OLS(d["delta_R_pf"].to_numpy(), X).fit()

    mkt_excess_mean = float(port.loc[port["Trdwnt"].isin(test_weeks), "r_mt_rf"].mean())
    return {
        "data": d,
        "gamma0": float(res.params[0]),
        "gamma1": float(res.params[1]),
        "t_gamma0": float(res.tvalues[0]),
        "t_gamma1": float(res.tvalues[1]),
        "r2": float(res.rsquared),
        "mkt_excess_mean": mkt_excess_mean,
        "n_portfolios": int(len(d)),
    }


def plot_sml(d: pd.DataFrame, cs: dict):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.scatter(d["beta_p"], d["delta_R_pf"], s=42, color="#1f3b57", zorder=3, label="组合")
    x = np.linspace(d["beta_p"].min() * 0.97, d["beta_p"].max() * 1.02, 50)

    ax.plot(x, cs["gamma0"] + cs["gamma1"] * x, color="#2b579a", lw=2, label="拟合 SML")
    ax.plot(x, cs["mkt_excess_mean"] * x, color="#b3261e", lw=1.6, ls="--", label="理论 SML")

    for _, row in d.iterrows():
        ax.annotate(str(int(row["group_id"])), (row["beta_p"], row["delta_R_pf"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points", color="#555")

    ax.set_title(f"CAPM 截面检验：{cs['n_portfolios']} 个组合（γ1 t 值 {cs['t_gamma1']:.2f}，R² {cs['r2']:.2f}）",
                 fontsize=11)
    ax.set_xlabel("组合 Beta")
    ax.set_ylabel("平均周超额收益")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    data_io.save_figure(fig, "p2_CAPM截面检验")
    plt.close(fig)


def run(demo: bool = False) -> dict:
    data_io.set_demo(demo)
    print("=" * 68)
    print("项目二·下：个股 Beta 估计与 CAPM 截面检验" + ("（模拟数据）" if demo else ""))
    print("=" * 68)

    panel, r_m = build_panel(demo)
    weeks = np.sort(panel["Trdwnt"].unique())
    half = len(weeks) // 2
    est_weeks, test_weeks = set(weeks[:half]), set(weeks[half:])
    print(f"\n周数 {len(weeks)}：估计期 {weeks[0]} ~ {weeks[half-1]}，检验期 {weeks[half]} ~ {weeks[-1]}")

    betas = beta_by_stock(panel, est_weeks)
    print(f"个股 Beta 估计完成：{len(betas)} 只（估计期有效观测 ≥ {config.MIN_WEEKS_FOR_BETA} 周）")
    print(f"  个股 R² 中位数 {betas['R2_i'].median():.3f}；"
          f"|t(β)| > 1.96 的占比 {(betas['t_beta'].abs() > 1.96).mean():.1%}")

    groups = assign_groups(betas)
    port = portfolio_series(panel, groups)
    pbetas = portfolio_betas(port, est_weeks)
    print(f"组合构造完成：{len(pbetas)} 组 × {config.STOCKS_PER_PORTFOLIO} 只等权")
    print(f"  组合 Beta 全部显著的组数：{(pbetas['t_beta_p'].abs() > 1.96).sum()}/{len(pbetas)}；"
          f"组合 R² 中位数 {pbetas['R2_p'].median():.3f}")

    cs = cross_section(port, pbetas, test_weeks)
    print("\n【截面检验 SML】")
    print(f"  γ0 = {cs['gamma0']:.5f}（t = {cs['t_gamma0']:.2f}）")
    print(f"  γ1 = {cs['gamma1']:.5f}（t = {cs['t_gamma1']:.2f}）")
    print(f"  截面 R² = {cs['r2']:.3f}；市场平均周超额收益 = {cs['mkt_excess_mean']:.5f}")

    data_io.save_table(betas[["Stkcd", "alpha_i", "beta_i", "t_beta", "t_alpha", "R2_i"]], "p2_个股beta")
    data_io.save_table(pbetas, "p2_组合beta")
    data_io.save_table(cs["data"], "p2_组合超额收益")
    plot_sml(cs["data"], cs)
    return cs


if __name__ == "__main__":
    run(demo="--demo" in sys.argv)
