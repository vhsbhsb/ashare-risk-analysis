"""
数据读取层。

作用：把「CSMAR 导出的一堆 dta 文件」统一读成一个 DataFrame，
让你在分析代码里不用关心文件名、不用写 use/append。

两种模式：
    load("turnover")            -> 从 DATA_DIR 里按通配符拼接真实数据
    load("turnover", demo=True) -> 生成同结构的模拟数据，用于跑通流程

真实数据的文件名规则见 DATASETS，你可以直接改 pattern。
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

import config
import demo_data


# ---------------------------------------------------------------- 文件名规则
# key -> (glob 通配符, 需要保留的列)
# 通配符会匹配 DATA_DIR 下所有同类文件并按行拼接（对应 Stata 的 append using）
DATASETS = {
    # 全市场日成交额：Trddt=交易日期, Dnvaltrdtl=成交金额
    "turnover": ("*日交易额*.dta", ["Trddt", "Dnvaltrdtl"]),
    # 融资融券余额：Mtdate=日期, Mtbalance=融资融券余额
    "margin": ("*余额*.dta", ["Mtdate", "Mtbalance"]),
    # 新发基金份额：Date=日期, Scale=发行份额
    "fund_shares": ("*fund shares*.dta", ["Date", "Scale"]),
    # 沪深 300 估值：CSI300_PE=市盈率, CSI300_DividendYield=股息率
    "csi300_val": ("*CSI300*PE*.dta", ["Date", "CSI300_PE", "CSI300_DividendYield"]),
    # 国债收益率曲线：Yeartomatu=剩余期限, Yield=收益率(%)
    "bond": ("*国债*.dta", ["Trddt", "Yeartomatu", "Yield"]),
    # 指数日行情：Indexcd=指数代码, Idxtrd01=日期, Idxtrd05=收盘价
    "index": ("*指数*.dta", ["Indexcd", "Idxtrd01", "Idxtrd05"]),
    # 个股周收益：Stkcd=代码, Trdwnt=交易周, Wretnd=周收益率
    "stock_weekly": ("*周个股*.dta", ["Stkcd", "Trdwnt", "Wretnd"]),
    # 周度无风险利率：tw=交易周, riskfree_w=周无风险利率
    "riskfree_weekly": ("*riskfree*.dta", ["tw", "riskfree_w"]),
    # 个股月行情：Trdmnt=交易月, Msmvosd=流通市值, Mretnd=月收益率
    "stock_monthly": ("*月个股*.dta", ["Stkcd", "Trdmnt", "Msmvosd", "Mretnd"]),
    # 三因子：RiskPremium1=市场风险溢价, SMB1=规模因子, HML1=价值因子
    "factors": ("*因子*.dta", ["TradingMonth", "RiskPremium1", "SMB1", "HML1"]),
    # 个股日价格：stk_code=代码, Price=收盘价
    "price": ("*representative price*.dta", ["Date", "stk_code", "Price"]),
}


def _files_for(pattern: str) -> list[str]:
    files = sorted(glob.glob(str(Path(config.DATA_DIR) / pattern)))
    return files


def load(key: str, demo: bool = False) -> pd.DataFrame:
    """按 key 读取一份数据；demo=True 时返回模拟数据。"""
    if key not in DATASETS:
        raise KeyError(f"未知数据集 {key}，可选：{list(DATASETS)}")

    if demo:
        return demo_data.make(key)

    pattern, cols = DATASETS[key]
    files = _files_for(pattern)
    if not files:
        raise FileNotFoundError(
            f"在 {config.DATA_DIR} 里找不到匹配 {pattern} 的文件。\n"
            f"请确认：1) DATA_DIR 是否指向数据文件夹；"
            f"2) 文件名是否包含该通配符中的关键字。"
        )

    frames = []
    for f in files:
        df = pd.read_stata(f, convert_categoricals=False)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{Path(f).name} 缺少列 {missing}，实际列：{list(df.columns)[:15]}")
        frames.append(df[cols])

    return pd.concat(frames, ignore_index=True)


def save_table(df: pd.DataFrame, name: str) -> Path:
    """把结果表写成 csv 到 output/。"""
    path = _outdir() / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [表] {path.name}")
    return path


def save_figure(fig, name: str) -> Path:
    """把图存成 png 到 output/。"""
    path = _outdir() / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  [图] {path.name}")
    return path


# ---------------------------------------------------------------- 输出目录
# 模拟数据的输出单独放 output/demo/，避免和真实结果混在一起、
# 被人误当成实证结论。
_DEMO = False


def set_demo(flag: bool) -> None:
    global _DEMO
    _DEMO = bool(flag)


def _outdir() -> Path:
    path = Path(config.OUT_DIR) / ("demo" if _DEMO else "")
    path.mkdir(parents=True, exist_ok=True)
    return path
