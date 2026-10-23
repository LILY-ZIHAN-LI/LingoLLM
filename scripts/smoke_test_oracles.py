"""Week 1 的 go/no-go 检查：四个 oracle 能跑 + 吞吐量够不够。

跑通这个脚本 = proposal 里「oracle audit」milestone 完成。
用法: python scripts/smoke_test_oracles.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.oracles import OBJECTIVES, OracleBank  # noqa: E402

# 几个已知分子 + 一个故意写错的
PROBES = [
    "CC(=O)Oc1ccccc1C(=O)O",                      # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",               # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                 # ibuprofen
    "c1ccccc1" * 12,                              # 巨大但合法
    "C1=CC=CC=C",                                 # 非法
]


def main() -> int:
    print("加载 oracle（首次运行会下载 GSK3B/JNK3 的 RF 模型）...")
    t0 = time.perf_counter()
    bank = OracleBank()
    print(f"  加载耗时 {time.perf_counter() - t0:.1f}s\n")

    print(f"{'SMILES':<34} " + " ".join(f"{o:>7}" for o in OBJECTIVES))
    for smi in PROBES:
        v = bank.vector(smi)
        label = (smi[:31] + "...") if len(smi) > 34 else smi
        if v is None:
            print(f"{label:<34} {'INVALID':>7}")
        else:
            print(f"{label:<34} " + " ".join(f"{x:7.3f}" for x in v))

    # 吞吐量：决定实验规模能开多大
    print("\n吞吐量测试（200 个唯一分子）...")
    base = "CC(=O)Oc1ccccc1C(=O)O"
    mols = [base.replace("CC(=O)", "C" * (i % 5 + 1) + "C(=O)") for i in range(200)]
    t0 = time.perf_counter()
    for m in mols:
        bank.score(m)
    dt = time.perf_counter() - t0
    print(f"  {dt:.2f}s -> {len(mols) / dt:.1f} 分子/秒/核")
    print(f"  stats: {bank.stats()}")

    budget = 6 * 100 * 3 * 20 * 4        # 6 policy x 100 seed x 3 repeat x 20 step x 4 cand
    print(f"\n主实验上限 {budget:,} 次打分 -> "
          f"单核约 {budget / (len(mols) / dt) / 3600:.1f} 小时；"
          f"8 核并行约 {budget / (len(mols) / dt) / 3600 / 8:.1f} 小时")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
