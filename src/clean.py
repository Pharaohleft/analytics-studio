from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from sqlalchemy import text
from .db import get_engine

def _winsorize_series(s: pd.Series, lower_q: float = 0.0, upper_q: float = 0.99) -> Tuple[pd.Series, int, float, float]:
    s_numeric = pd.to_numeric(s, errors="coerce")
    lo = s_numeric.quantile(lower_q)
    hi = s_numeric.quantile(upper_q)
    before_caps = ((s_numeric < lo) | (s_numeric > hi)).sum()
    s_capped = s_numeric.clip(lower=lo, upper=hi)
    return s_capped, int(before_caps), float(lo), float(hi)

def _mode_safe(s: pd.Series):
    vals = s.dropna()
    if vals.empty:
        return None
    return vals.mode().iloc[0] if not vals.mode().empty else None

def clean_dataframe(
    df: pd.DataFrame,
    *,
    id_col: Optional[str] = None,
    ts_col: Optional[str] = None,
    drop_exact_dups: bool = True,
    drop_dups_by_id_ts: bool = True,
    trim_strings: bool = True,
    winsorize_numeric: bool = True,
    winsor_upper_q: float = 0.99,
    impute_numeric: bool = True,
    impute_categorical: bool = True,
    categorical_fill_value: str = "UNKNOWN"
) -> Tuple[pd.DataFrame, str]:
    """
    Apply common cleaning operations and return (clean_df, report_text).
    """
    work = df.copy()
    report: List[str] = []
    n0 = len(work)

    # Trim string columns
    if trim_strings:
        str_cols = work.select_dtypes(include=["object"]).columns.tolist()
        for c in str_cols:
            work[c] = work[c].astype(str).str.strip()
        report.append(f"- Trimmed whitespace in {len(str_cols)} string columns")

    # Drop exact duplicate rows
    if drop_exact_dups:
        dups_all = int(work.duplicated().sum())
        if dups_all > 0:
            work = work.drop_duplicates().reset_index(drop=True)
        report.append(f"- Dropped exact duplicate rows: {dups_all}")

    # Drop duplicates by (id, ts)
    if drop_dups_by_id_ts and id_col and ts_col and id_col in work.columns and ts_col in work.columns:
        dups_pair = int(work.duplicated(subset=[id_col, ts_col]).sum())
        if dups_pair > 0:
            work = work.drop_duplicates(subset=[id_col, ts_col]).reset_index(drop=True)
        report.append(f"- Dropped duplicates by ({id_col}, {ts_col}): {dups_pair}")
    else:
        if drop_dups_by_id_ts:
            report.append("- Skipped pair-duplicate drop (id or ts column not provided)")

    # Winsorize numeric columns
    capped_counts = []
    if winsorize_numeric:
        num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            capped_s, n_capped, lo, hi = _winsorize_series(work[c], 0.0, winsor_upper_q)
            work[c] = capped_s
            if n_capped > 0:
                capped_counts.append((c, n_capped, lo, hi))
        if capped_counts:
            details = "; ".join([f"{c}: {n} caps at [{lo:.4g}, {hi:.4g}]" for c, n, lo, hi in capped_counts])
            report.append(f"- Winsorized numeric outliers (upper q={winsor_upper_q}): {details}")
        else:
            report.append(f"- Winsorized numeric outliers (upper q={winsor_upper_q}): none capped")

    # Impute missing values
    impute_notes = []
    if impute_numeric:
        for c in work.select_dtypes(include=[np.number]).columns:
            n_missing = int(work[c].isna().sum())
            if n_missing > 0:
                med = work[c].median()
                work[c] = work[c].fillna(med)
                impute_notes.append(f"{c}: {n_missing} -> median {med}")
    if impute_categorical:
        for c in work.select_dtypes(exclude=[np.number]).columns:
            n_missing = int(work[c].isna().sum())
            if n_missing > 0:
                mode_val = _mode_safe(work[c])
                fill_val = mode_val if mode_val is not None else categorical_fill_value
                work[c] = work[c].fillna(fill_val)
                impute_notes.append(f"{c}: {n_missing} -> {fill_val}")
    if impute_notes:
        report.append("- Imputations: " + "; ".join(impute_notes))
    else:
        report.append("- Imputations: none applied")

    # Summary
    n1 = len(work)
    report.append(f"- Row count before: {n0}, after: {n1}")
    return work, "\n".join(report)

def save_clean_to_postgres(
    df: pd.DataFrame,
    table: str,
    schema: str = "clean",
    if_exists: str = "replace"
) -> Tuple[str, int]:
    """
    Save cleaned DataFrame to Postgres under schema.table.
    Returns (qualified_table_name, rowcount).
    """
    eng = get_engine()
    sch = schema.strip()
    tbl = table.strip()
    if "." in tbl:
        # allow fully qualified override
        sch, tbl = tbl.split(".", 1)
    with eng.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{sch}"'))
        df.to_sql(tbl, con=conn.connection, schema=sch, if_exists=if_exists, index=False)
    return f"{sch}.{tbl}", len(df)
