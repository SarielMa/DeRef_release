"""CSV writers of the statistics step: each table as proportions and as a x100, two-decimal copy."""
import sys
import time
from pathlib import Path

from deref.stats.tables import times100

# columns scaled in the x100 copy (proportions); counts, tau and q are left unchanged
X100 = {
    "table1_gap_plain": ["share", "gap", "se", "lo", "hi", "contrib"],
    "argument_grid": ["auroc", "auroc_se", "acc_c_tau", "acc_c_05", "acc_w_tau", "acc_w_05", "gap", "gap_se", "gap_lo", "gap_hi"],
    "argument_diffs": ["d_auroc", "d_auroc_lo", "d_auroc_hi", "d_gap", "d_gap_lo", "d_gap_hi", "se_d_auroc", "se_d_gap"],
    "judge_averages": ["method_mean", "ref_mean", "mean_diff", "se", "lo", "hi"],
    "ablation": ["auroc_ref", "se_ref", "auroc_x", "se_x", "d", "lo", "hi", "se_d"],
}


def write_table(df, out_dir, name):
    """Write `<name>.csv` (proportions) and `<name>_x100.csv` (x100, two decimals)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    times100(df, X100[name]).to_csv(out_dir / f"{name}_x100.csv", index=False)
    log(f"wrote {out_dir / name}.csv ({len(df)} rows) and {name}_x100.csv")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)
