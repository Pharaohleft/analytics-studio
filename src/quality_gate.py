from __future__ import annotations
from typing import Dict, Optional, Tuple
import pandas as pd

def _dup_pair_ratio(df: pd.DataFrame, id_col: Optional[str], ts_col: Optional[str]) -> Optional[float]:
    if not id_col or not ts_col or id_col not in df.columns or ts_col not in df.columns:
        return None
    tmp = df[[id_col, ts_col]].copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce")
    return float(tmp.duplicated().mean())

def run_quality_checks(
    df: pd.DataFrame,
    *,
    id_col: Optional[str] = None,
    ts_col: Optional[str] = None,
    max_missing_pct: float = 0.20,
    max_dup_exact: float = 0.05,
    max_dup_pair: float = 0.02,
    max_ts_staleness_days: Optional[int] = None,
) -> Tuple[bool, str, Dict]:
    miss = df.isna().mean()
    high_miss = miss[miss > max_missing_pct].sort_values(ascending=False)
    dup_exact = float(df.duplicated().mean())
    dup_pair = _dup_pair_ratio(df, id_col, ts_col)
    fresh_ok = True
    staleness_days = None
    if ts_col and ts_col in df.columns and max_ts_staleness_days is not None:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        if ts.notna().any():
            staleness_days = int((pd.Timestamp.now(tz=None) - ts.max()).days)
            fresh_ok = staleness_days <= max_ts_staleness_days

    pass_all = (
        (dup_exact <= max_dup_exact) and
        (dup_pair is None or dup_pair <= max_dup_pair) and
        high_miss.empty and
        fresh_ok
    )

    lines = []
    lines.append(f"Exact duplicate rate: {dup_exact*100:.2f}% (≤ {max_dup_exact*100:.0f}% required)")
    if dup_pair is not None:
        lines.append(f"Duplicate rate by ({id_col},{ts_col}): {dup_pair*100:.2f}% (≤ {max_dup_pair*100:.0f}% required)")
    if not high_miss.empty:
        lines.append("High-missing columns (> {0:.0f}%): {1}".format(max_missing_pct*100, ', '.join(high_miss.index.tolist())))
    else:
        lines.append("Missingness OK (all columns under threshold).")
    if staleness_days is not None:
        lines.append(f"Freshness: last {ts_col} = {staleness_days} days ago (≤ {max_ts_staleness_days} required)")

    md = ("**QUALITY: PASS**" if pass_all else "**QUALITY: FAIL**") + "\n\n" + "\n".join(f"- {x}" for x in lines)
    details = {
        "missing_over_threshold": high_miss.to_dict(),
        "dup_exact_ratio": dup_exact,
        "dup_pair_ratio": dup_pair,
        "staleness_days": staleness_days,
        "thresholds": {
            "max_missing_pct": max_missing_pct,
            "max_dup_exact": max_dup_exact,
            "max_dup_pair": max_dup_pair,
            "max_ts_staleness_days": max_ts_staleness_days,
        }
    }
    return pass_all, md, details
