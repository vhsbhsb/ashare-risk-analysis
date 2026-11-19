"""
一键运行全部三个项目。

    python run_all.py --demo      # 用模拟数据跑通全流程（不需要真实数据）
    python run_all.py             # 用 config.DATA_DIR 里的真实数据

每个项目也可以单独跑，见各文件顶部的说明。
"""
from __future__ import annotations

import sys
import time

import p1_market_heat
import p2_capm
import p2_distribution
import p3_factors


def main() -> None:
    demo = "--demo" in sys.argv
    if demo:
        print("\n>>> 模拟数据模式：结果仅用于验证流程，数值没有经济含义\n")

    t0 = time.time()
    p1_market_heat.run(demo=demo)
    p2_distribution.run(demo=demo)
    p2_capm.run(demo=demo)
    p3_factors.run(demo=demo)

    print("\n" + "=" * 68)
    print(f"全部完成，用时 {time.time() - t0:.1f} 秒；结果在 output/ 目录")
    print("=" * 68)


if __name__ == "__main__":
    main()
