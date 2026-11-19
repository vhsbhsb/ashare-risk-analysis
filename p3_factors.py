"""
项目三：A 股规模与动量因子的组合检验

对应作业：HW4。两件事写在一个文件里，因为它们共用同一份月度面板：
    (a) 规模效应：按滞后一期流通市值十分组，看小市值组是否跑赢大市值组，
        再用市场模型 / FF3 把市场、规模、价值三层敞口剥开，看超额收益是否还在
    (b) 动量效应：按滞后一期收益率五分组，看赢家组是否继续跑赢输家组

关键设计（也是这类作业最容易出错的地方）：
    · 分组用的市值与收益都取滞后一期，避免用当月信息决定当月持仓（前视偏差）
    · 月度间隔必须连续，隔月断档的样本不能用 shift 直接补
    · 剔除创业板（300）与科创板（688），避免板块制度差异污染因子方向

运行：python p3_factors.py [--demo]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import config
import data_io

plt.rcParams["font.sans-serif"] = [config.PLOT_FONT, "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 面板
def monthly_panel(demo: bool) -> pd.DataFrame:
    """个股月度面板：代码、月份、滞后市值、滞后收益、当月收益。"""
    df = data_io.load("stock_monthly", demo)
    df["Stkcd"] = df["Stkcd"].astype(str).str.zfill(6)
    df = df[df["Stkcd"].str[:3].isin(config.MAIN_BOARD_PREFIX)].copy()

    df["ym"] = pd.PeriodIndex(df["Trdmnt"].astype(str).str[:7], freq="M")
    df = df[["Stkcd", "ym", "Msmvosd", "Mretnd"]].dropna()
    df = df.sort_values(["Stkcd", "ym"]).reset_index(drop=True)

    g = df.groupby("Stkcd")
    df["prev_ym"] = g["ym"].shift(1)
    df["lag_mv"] = g["Msmvosd"].shift(1)
    df["lag_ret"] = g["Mretnd"].shift(1)

    # 只保留月份连续的相邻两期，断档样本丢弃
    df = df[df["ym"] == df["prev_ym"] + 1]
    return df.drop(columns=["prev_ym"])


def factor_returns(demo: bool) -> pd.DataFrame:
    f = data_io.load("factors", demo).rename(
        columns={"RiskPremium1": "mktrf", "SMB1": "smb", "HML1": "hml"}
    )
    f["ym"] = pd.PeriodIndex(f["TradingMonth"].astype(str).str[:7], freq="M")
    return f[["ym", "mktrf", "smb", "hml"]].dropna().drop_duplicates("ym")


def assign_groups(df: pd.DataFrame, sort_col: str, n_groups: int) -> pd.DataFrame:
    """按 sort_col 在每个月内排名，等分成 n_groups 组（1 = 最小/最输家）。"""
    rank = df.groupby("ym")[sort_col].rank(method="first")
    size = df.groupby("ym")[sort_col].transform("size")
    df = df.copy()
    df["grp"] = np.ceil(n_groups * rank / size).clip(1, n_groups).astype(int)
    return df


def portfolio_returns(df: pd.DataFrame) -> pd.DataFrame:
    """等权组合的月度收益：(月 × 组) 的宽表。"""
    wide = df.pivot_table(index="ym", columns="grp", values="Mretnd", aggfunc="mean")
    wide.columns = [f"g{c}" for c in wide.columns]
    return wide.sort_index()


# ---------------------------------------------------------------- 统计
def group_summary(port: pd.DataFrame) -> pd.DataFrame:
    """每组平均月收益与 t 值。"""
    rows = []
    for c in [c for c in port.columns if c.startswith("g")]:
        s = port[c].dropna()
        t, p = stats.ttest_1samp(s, 0.0)
        rows.append({"组合": c, "平均月收益": s.mean(), "t值": t, "p值": p, "月数": len(s)})
    return pd.DataFrame(rows)


def alpha_table(port: pd.DataFrame, factors: pd.DataFrame, long_short: str, use_ff3: bool) -> pd.DataFrame:
    """各组合（含多空）对市场模型或 FF3 的 α 与因子暴露。"""
    d = port.join(factors.set_index("ym"), how="inner")
    xs = ["mktrf", "smb", "hml"] if use_ff3 else ["mktrf"]

    rows = []
    for c in [c for c in port.columns if c.startswith("g")] + [long_short]:
        s = d[[c] + xs].dropna()
        X = sm.add_constant(s[xs].to_numpy())
        res = sm.OLS(s[c].to_numpy(), X).fit()
        row = {"组合": c, "alpha": res.params[0], "t(alpha)": res.tvalues[0], "R2": res.rsquared}
        for i, x in enumerate(xs, start=1):
            row[f"β_{x}"] = res.params[i]
            row[f"t({x})"] = res.tvalues[i]
        rows.append(row)
    return pd.DataFrame(rows)


def ls_test(port: pd.DataFrame, long_short: str) -> tuple[float, float]:
    s = port[long_short].dropna()
    t, _ = stats.ttest_1samp(s, 0.0)
    return float(s.mean()), float(t)


# ---------------------------------------------------------------- 图
def plot_groups(port: pd.DataFrame, title: str, xlabel: str, fname: str):
    cols = [c for c in port.columns if c.startswith("g")]
    means = [port[c].mean() for c in cols]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(range(1, len(cols) + 1), np.array(means) * 100, color="#4a6fa5", width=0.65)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("平均月收益（%）")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    data_io.save_figure(fig, fname)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def run(demo: bool = False) -> dict:
    data_io.set_demo(demo)
    print("=" * 68)
    print("项目三：A 股规模与动量因子的组合检验" + ("（模拟数据）" if demo else ""))
    print("=" * 68)

    panel = monthly_panel(demo)
    factors = factor_returns(demo)
    span = f"{panel['ym'].min()} ~ {panel['ym'].max()}"
    print(f"\n样本：{span}，{panel['Stkcd'].nunique()} 只个股，{panel['ym'].nunique()} 个月，"
          f"{len(panel):,} 个股票-月观测")

    out = {}

    # ---------- (a) 规模效应 ----------
    print("\n【(a) 规模效应：十分组】")
    size_df = assign_groups(panel, "lag_mv", config.N_SIZE_GROUPS)
    size_port = portfolio_returns(size_df)
    size_port["LS"] = size_port[f"g{config.N_SIZE_GROUPS}"] - size_port["g1"]

    summ = group_summary(size_port)
    mean_ls, t_ls = ls_test(size_port, "LS")
    print(summ.round(4).to_string(index=False))
    print(f"  G{config.N_SIZE_GROUPS}−G1 = {mean_ls:.4%}/月（t = {t_ls:.2f}）")

    alpha_mm = alpha_table(size_port, factors, "LS", use_ff3=False)
    print("\n  市场模型 α（剥离市场 β 之后还剩多少）")
    print(alpha_mm[["组合", "alpha", "t(alpha)", "β_mktrf", "R2"]].round(4).to_string(index=False))

    data_io.save_table(summ, "p3_规模分组收益")
    data_io.save_table(alpha_mm, "p3_规模市场模型alpha")
    plot_groups(size_port, f"规模十分组平均月收益（G10−G1 = {mean_ls:.2%}，t = {t_ls:.2f}）",
                "规模分组（1 = 最小市值）", "p3_规模效应")

    # ---------- (b) 动量效应 ----------
    print("\n【(b) 动量效应：五分组】")
    mom_df = assign_groups(panel, "lag_ret", config.N_MOM_GROUPS)
    mom_port = portfolio_returns(mom_df)
    mom_port["LS"] = mom_port[f"g{config.N_MOM_GROUPS}"] - mom_port["g1"]

    summ_m = group_summary(mom_port)
    mean_ls_m, t_ls_m = ls_test(mom_port, "LS")
    print(summ_m.round(4).to_string(index=False))
    print(f"  G{config.N_MOM_GROUPS}−G1 = {mean_ls_m:.4%}/月（t = {t_ls_m:.2f}）")

    alpha_ff3 = alpha_table(mom_port, factors, "LS", use_ff3=True)
    print("\n  FF3 α（控制市场、规模、价值后的动量残差）")
    print(alpha_ff3[["组合", "alpha", "t(alpha)", "t(mktrf)", "t(smb)", "t(hml)"]].round(4).to_string(index=False))

    data_io.save_table(summ_m, "p3_动量分组收益")
    data_io.save_table(alpha_ff3, "p3_动量FF3alpha")
    plot_groups(mom_port, f"动量五分组平均月收益（G5−G1 = {mean_ls_m:.2%}，t = {t_ls_m:.2f}）",
                "动量分组（1 = 过去最输家）", "p3_动量效应")

    out = {
        "size_ls": mean_ls, "size_t": t_ls,
        "mom_ls": mean_ls_m, "mom_t": t_ls_m,
        "size_alpha": float(alpha_mm.loc[alpha_mm["组合"] == "LS", "alpha"].iloc[0]),
        "size_alpha_t": float(alpha_mm.loc[alpha_mm["组合"] == "LS", "t(alpha)"].iloc[0]),
        "mom_alpha": float(alpha_ff3.loc[alpha_ff3["组合"] == "LS", "alpha"].iloc[0]),
        "mom_alpha_t": float(alpha_ff3.loc[alpha_ff3["组合"] == "LS", "t(alpha)"].iloc[0]),
    }
    print("\n【小结】")
    print(f"  规模：多空 {out['size_ls']:.2%}/月，市场调整后 α {out['size_alpha']:.2%}"
          f"（t = {out['size_alpha_t']:.2f}）")
    print(f"  动量：多空 {out['mom_ls']:.2%}/月，FF3 后 α {out['mom_alpha']:.2%}"
          f"（t = {out['mom_alpha_t']:.2f}）")
    if out["mom_ls"] < 0 and out["mom_t"] < -1.96:
        print("  → 多空显著为负：本样本里赢家跑输输家，是反转而不是动量。")
    return out


if __name__ == "__main__":
    run(demo="--demo" in sys.argv)
