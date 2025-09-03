import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List

ID_CANDIDATES = [
    "customer_id","player_id","patient_id","user_id","client_id","account_id",
    "id","entity_id","member_id","subscriber_id"
]
TS_CANDIDATES = [
    "order_date","game_date","visit_date","date","datetime","timestamp",
    "created_at","updated_at","event_time","event_ts","ts"
]
VAL_CANDIDATES = [
    "order_value","amount","price","total","revenue","sales","claim_amount",
    "points","score","value","qty","quantity","net","gross","charge"
]

def _rank_guess(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_l = [c.lower() for c in cols]
    # exact match first
    for cand in candidates:
        if cand in cols_l:
            return cols[cols_l.index(cand)]
    # substring match
    for cand in candidates:
        for i,c in enumerate(cols_l):
            if cand in c:
                return cols[i]
    return None

def _guess_roles(df: pd.DataFrame,
                 id_hint: Optional[str]=None,
                 ts_hint: Optional[str]=None,
                 val_hint: Optional[str]=None) -> Tuple[Optional[str],Optional[str],Optional[str],str]:
    cols = list(df.columns)
    msg = []
    id_col = id_hint if id_hint in cols else _rank_guess(cols, ID_CANDIDATES)
    ts_col = ts_hint if ts_hint in cols else _rank_guess(cols, TS_CANDIDATES)
    val_col = val_hint if val_hint in cols else _rank_guess(cols, VAL_CANDIDATES)

    msg.append(f"- Guessed **ID**: {id_col}" if id_col else "- Guessed **ID**: (not found)")
    msg.append(f"- Guessed **Timestamp**: {ts_col}" if ts_col else "- Guessed **Timestamp**: (not found)")
    msg.append(f"- Guessed **Value**: {val_col}" if val_col else "- Guessed **Value**: (not found)")

    return id_col, ts_col, val_col, "\n".join(msg)

def _parse_dates(df: pd.DataFrame, ts_col: Optional[str]) -> Tuple[pd.DataFrame,str]:
    notes = []
    if ts_col and ts_col in df.columns:
        try:
            before_dtype = str(df[ts_col].dtype)
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=False, infer_datetime_format=True)
            after_dtype = str(df[ts_col].dtype)
            coerced = int(df[ts_col].isna().sum())
            notes.append(f"- Parsed {ts_col} from {before_dtype} → {after_dtype} (coerced NA: {coerced})")
        except Exception as e:
            notes.append(f"- ❌ Failed to parse {ts_col} as datetime: {e}")
    else:
        notes.append("- (No timestamp column to parse)")
    return df, "\n".join(notes)

def _missingness(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    out = pd.DataFrame({"missing": miss, "missing_pct": pct}).sort_values("missing_pct", ascending=False)
    return out

def _dup_counts(df: pd.DataFrame, id_col: Optional[str], ts_col: Optional[str]) -> Dict[str,int]:
    out = {"row_dups_all_cols": int(df.duplicated().sum())}
    if id_col and ts_col and id_col in df.columns and ts_col in df.columns:
        out["row_dups_by_id_ts"] = int(df.duplicated(subset=[id_col, ts_col]).sum())
    return out

def _numeric_outliers(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    if num.empty:
        return pd.DataFrame(columns=["col","outlier_count","p99","max"])
    stats = []
    for c in num.columns:
        s = num[c].astype(float)
        m = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            out_n = 0
        else:
            z = (s - m) / sd
            out_n = int((np.abs(z) > 4).sum())
        p99 = float(np.nanpercentile(s, 99)) if s.notna().any() else np.nan
        mx = float(s.max()) if s.notna().any() else np.nan
        stats.append((c, out_n, p99, mx))
    return pd.DataFrame(stats, columns=["col","outlier_count","p99","max"]).sort_values("outlier_count", ascending=False)

def _dtypes_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        uniq = int(s.nunique(dropna=True))
        miss = int(s.isna().sum())
        miss_pct = round(miss/len(df)*100, 2) if len(df) else 0.0
        sample_vals = list(s.dropna().head(3).astype(str))
        rows.append([c, dtype, uniq, miss, miss_pct, sample_vals])
    return pd.DataFrame(rows, columns=["column","dtype","unique","missing","missing_pct","sample_values"])

def validate_csv(csv_path: str,
                 id_hint: Optional[str]=None,
                 ts_hint: Optional[str]=None,
                 val_hint: Optional[str]=None) -> str:
    p = Path(csv_path)
    if not p.exists():
        return f"❌ File not found: {csv_path}"

    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        return f"❌ Could not read CSV: {e}"

    n_rows, n_cols = df.shape
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024**2), 2)

    id_col, ts_col, val_col, guess_msg = _guess_roles(df, id_hint, ts_hint, val_hint)
    df, parse_msg = _parse_dates(df, ts_col)

    miss_tbl = _missingness(df)
    dtypes_tbl = _dtypes_table(df)
    dup_info = _dup_counts(df, id_col, ts_col)
    out_tbl = _numeric_outliers(df)

    report = []
    report.append(f"### File Summary")
    report.append(f"- Rows: **{n_rows}**, Cols: **{n_cols}**, Memory: **{mem_mb} MB**")
    report.append("")
    report.append("### Role Guesses")
    report.append(guess_msg)
    report.append("")
    report.append("### Timestamp Parsing")
    report.append(parse_msg)
    report.append("")
    report.append("### Missingness (top 10)")
    report.append(miss_tbl.head(10).to_markdown(index=True))
    report.append("")
    report.append("### Dtypes & Samples (top 15)")
    report.append(dtypes_tbl.head(15).to_markdown(index=False))
    report.append("")
    report.append("### Duplicates")
    report.append("- " + " | ".join([f"{k}: {v}" for k,v in dup_info.items()]))
    report.append("")
    report.append("### Numeric Outliers (z-score > 4, top 10)")
    report.append(out_tbl.head(10).to_markdown(index=False))
    report.append("")
    report.append("### Next suggested actions")
    actions = []
    if dup_info.get("row_dups_all_cols", 0) > 0:
        actions.append("- Drop exact duplicate rows.")
    if ts_col is None:
        actions.append("- Map a timestamp column (Upload & Map) before RFM/Churn.")
    else:
        actions.append("- Confirm timestamp timezone and granularity (day/hour).")
    if id_col is None:
        actions.append("- Map an ID column for entity-level metrics.")
    if not out_tbl.empty and (out_tbl["outlier_count"] > 0).any():
        actions.append("- Consider winsorizing extreme numeric outliers (e.g., cap at p99).")
    if miss_tbl["missing_pct"].max() > 20:
        actions.append("- Impute or drop columns with heavy missingness (>20%).")
    if not actions:
        actions.append("- Data looks healthy. Proceed to Clean → RFM.")
    report.extend(actions)

    return "\n".join(report)
