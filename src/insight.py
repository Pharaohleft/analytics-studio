# src/insight.py — decisive Insight Pack + DQ before/after + drivers + anomalies + ZIP bundle
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import json
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from pandas.api.types import is_object_dtype, is_string_dtype

from src.schema_map import suggest_mapping, apply_mapping
from src.clean import clean_dataframe, save_clean_to_postgres
from src.eda import load_df_from_source, infer_columns


# ------------------------ helpers ------------------------

def _ensure_dirs():
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    Path("outputs/clean").mkdir(parents=True, exist_ok=True)
    Path("outputs/bundles").mkdir(parents=True, exist_ok=True)
    Path("outputs/drivers").mkdir(parents=True, exist_ok=True)


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Series(df.columns, dtype="object")
    if not cols.duplicated().any():
        return df
    counts: Dict[str, int] = {}
    new_cols = []
    for col in cols:
        if col in counts:
            counts[col] += 1
            new_cols.append(f"{col}.{counts[col]}")
        else:
            counts[col] = 0
            new_cols.append(col)
    out = df.copy()
    out.columns = new_cols
    return out


def _rename_semantics(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize common secondary category to payment_method
    if "category.1" in df.columns and "payment_method" not in df.columns:
        df = df.rename(columns={"category.1": "payment_method"})
    return df


def _auto_map(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    sugg = suggest_mapping(df)
    def first(role: str) -> Optional[str]:
        cands = sugg.get(role) or []
        return cands[0][0] if cands else None
    mapping = {
        "entity_id": first("entity_id"),
        "event_ts": first("event_ts"),
        "value": first("value"),
        "quantity": first("quantity"),
        "category": first("category"),
    }
    mapped = apply_mapping(df, mapping)
    mapped = _rename_semantics(_dedupe_columns(mapped))
    return mapped, mapping


def _weekly_series(df: pd.DataFrame, ts_col: str, value_col: Optional[str] = None):
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if value_col and value_col in df.columns:
        val = pd.to_numeric(df[value_col], errors="coerce")
        g = (pd.DataFrame({"ts": ts, "val": val})
                .dropna(subset=["ts"])
                .groupby(pd.Grouper(key="ts", freq="W"))["val"].sum())
        metric = f"Sum({value_col})"
    else:
        g = (pd.DataFrame({"ts": ts})
                .dropna(subset=["ts"])
                .groupby(pd.Grouper(key="ts", freq="W")).size())
        metric = "Row count"
    return g, metric


def _last4_vs_prior4(weekly: pd.Series) -> Optional[Dict[str, float]]:
    if len(weekly) < 8:
        return None
    recent = float(weekly.tail(4).sum())
    prior = float(weekly.tail(8).head(4).sum())
    change = recent - prior
    pct = (change / prior * 100.0) if prior else None
    return {"recent_4w": recent, "prior_4w": prior, "abs_change": change, "pct_change": pct}


def _z_anomalies(weekly: pd.Series, z_thresh: float = 2.0) -> pd.DataFrame:
    if len(weekly) < 12:
        return pd.DataFrame(columns=["week", "value", "zscore"])
    x = weekly.astype(float)
    mu, sd = x.mean(), x.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return pd.DataFrame(columns=["week", "value", "zscore"])
    z = (x - mu) / sd
    anom = z[abs(z) >= z_thresh]
    out = pd.DataFrame({"week": anom.index, "value": x.loc[anom.index], "zscore": z.loc[anom.index]})
    return out.sort_values("week")


def _drivers_last4(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    ts_col = mapping.get("event_ts")
    val_col = mapping.get("value")
    if not ts_col or ts_col not in df.columns:
        return pd.DataFrame(columns=["dimension", "member", "value"])
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    end = ts.max()
    if pd.isna(end):
        return pd.DataFrame(columns=["dimension", "member", "value"])
    start = end - pd.Timedelta(weeks=4)

    dims = []
    for dim in ["category", "payment_method", "state", "city"]:
        if dim in df.columns:
            dims.append(dim)
    if not dims:
        return pd.DataFrame(columns=["dimension", "member", "value"])

    df4 = df.loc[(ts >= start) & (ts <= end)].copy()
    if val_col and val_col in df4.columns:
        df4["val"] = pd.to_numeric(df4[val_col], errors="coerce").fillna(0.0)
        agg_fn = ("val", "sum")
    else:
        df4["val"] = 1.0
        agg_fn = ("val", "sum")

    rows = []
    for dim in dims:
        top = (df4.groupby(dim)["val"].sum().sort_values(ascending=False).head(10))
        for member, v in top.items():
            rows.append({"dimension": dim, "member": str(member), "value": float(v)})
    return pd.DataFrame(rows)


def _kpis(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Dict[str, object]:
    out: Dict[str, object] = {"total_rows": int(len(df))}
    val_col = mapping.get("value")
    if val_col and val_col in df.columns:
        vals = pd.to_numeric(df[val_col], errors="coerce")
        out["total_value"] = float(np.nansum(vals))
        out["avg_value"] = float(np.nanmean(vals)) if len(vals) else None

    ts_col = mapping.get("event_ts")
    if ts_col and ts_col in df.columns:
        w_count, _ = _weekly_series(df, ts_col, value_col=None)
        w_value, _ = _weekly_series(df, ts_col, value_col=val_col if val_col in df.columns else None)
        out["count_last4_vs_prior4"] = _last4_vs_prior4(w_count)
        out["value_last4_vs_prior4"] = _last4_vs_prior4(w_value)
        out["count_anomalies"] = _z_anomalies(w_count).to_dict(orient="records")
        out["value_anomalies"] = _z_anomalies(w_value).to_dict(orient="records")
    return out


def _quality_before_after(raw: pd.DataFrame, cleaned: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Dict[str, object]:
    def miss(df): return df.isna().mean().round(4).to_dict()
    def dtypes(df): return {c: str(t) for c, t in df.dtypes.items()}
    def dup_exact(df): return float(df.duplicated().mean())
    def dup_pair(df, ent, ts):
        if ent and ent in df.columns and ts and ts in df.columns:
            tmp = df[[ent, ts]].copy()
            tmp[ts] = pd.to_datetime(tmp[ts], errors="coerce")
            return float(tmp.duplicated().mean())
        return None

    ent, ts = mapping.get("entity_id"), mapping.get("event_ts")
    return {
        "raw": {
            "rows": int(len(raw)),
            "missing_pct": miss(raw),
            "dtypes": dtypes(raw),
            "dup_exact_ratio": dup_exact(raw),
            "dup_pair_ratio": dup_pair(raw, ent, ts)
        },
        "cleaned": {
            "rows": int(len(cleaned)),
            "missing_pct": miss(cleaned),
            "dtypes": dtypes(cleaned),
            "dup_exact_ratio": dup_exact(cleaned),
            "dup_pair_ratio": dup_pair(cleaned, ent, ts)
        }
    }


def _render_html(takeaways: List[str],
                 mapping: Dict[str, Optional[str]],
                 kpis: Dict[str, object],
                 dq: Dict[str, object],
                 drivers4: pd.DataFrame,
                 figs: Dict[str, object]) -> str:
    parts: List[str] = []
    parts.append("<html><head><meta charset='utf-8'><title>Insight Pack</title></head><body>")
    parts.append("<h1>Insight Pack</h1>")

    parts.append("<h2>Top Takeaways</h2><ul>")
    for t in takeaways:
        parts.append(f"<li>{t}</li>")
    parts.append("</ul>")

    parts.append("<h2>KPI Movements</h2><ul>")
    def _fmt(v):
        return "n/a" if v is None else f"{v:.2f}"
    c = kpis.get("count_last4_vs_prior4")
    v = kpis.get("value_last4_vs_prior4")
    if c:
        pct = c.get("pct_change")
        parts.append(f"<li>Last 4w COUNT: {_fmt(c['recent_4w'])} vs prior {_fmt(c['prior_4w'])} ({_fmt(pct)}%)</li>")
    if v:
        pct = v.get("pct_change")
        parts.append(f"<li>Last 4w VALUE: {_fmt(v['recent_4w'])} vs prior {_fmt(v['prior_4w'])} ({_fmt(pct)}%)</li>")
    parts.append("</ul>")

    if (kpis.get("count_anomalies") or kpis.get("value_anomalies")):
        parts.append("<h2>Anomalies (z-score ≥ 2)</h2>")
        rows = []
        for kind, arr in [("Count", kpis.get("count_anomalies")), ("Value", kpis.get("value_anomalies"))]:
            for rec in (arr or []):
                rows.append({"metric": kind, "week": str(rec["week"])[:10], "value": rec["value"], "zscore": rec["zscore"]})
        if rows:
            parts.append(pd.DataFrame(rows).to_html(index=False))

    if drivers4 is not None and not drivers4.empty:
        parts.append("<h2>Top Drivers (last 4 weeks)</h2>")
        parts.append(drivers4.to_html(index=False))

    parts.append("<h2>Data Quality (before → after)</h2>")
    parts.append("<pre>" + json.dumps(dq, indent=2) + "</pre>")

    parts.append("<h2>Visuals</h2>")
    for name in ["missingness", "histogram", "top_categories", "timeseries"]:
        fig = figs.get(name)
        if fig is None:
            continue
        parts.append(f"<h3>{name.replace('_',' ').title()}</h3>")
        parts.append(pio.to_html(fig, include_plotlyjs="inline", full_html=False))

    parts.append("<hr><p>Roles: " + json.dumps({k: v for k, v in mapping.items() if v}, indent=2) + "</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _plots(df: pd.DataFrame, mapping: Dict[str, Optional[str]]):
    figs: Dict[str, object] = {}
    # missingness
    miss = df.isna().mean().sort_values(ascending=False)
    ser = miss[miss > 0].head(50) if (miss > 0).any() else miss.head(50)
    figs["missingness"] = px.bar(x=ser.index, y=(ser * 100)).update_layout(
        xaxis_title="Column", yaxis_title="% Missing", margin=dict(l=40, r=20, t=20, b=120), xaxis_tickangle=45
    )

    numeric_cols, cat_cols, dt_cols = infer_columns(df)

    # histogram: prefer value; else best numeric by variance (exclude id-like)
    def is_id_like(name: str, s: pd.Series) -> bool:
        name_l = name.lower()
        if any(k in name_l for k in ["id", "uuid", "guid", "ssn"]):
            return True
        uniq_ratio = s.nunique(dropna=True) / max(1, len(s))
        if uniq_ratio > 0.9:
            return True
        if pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing:
            return True
        return False

    hist_col = None
    val = mapping.get("value")
    if val and val in numeric_cols:
        hist_col = val
    else:
        cand = [c for c in numeric_cols if not is_id_like(c, pd.to_numeric(df[c], errors="coerce"))]
        if cand:
            hist_col = sorted(cand, key=lambda c: pd.to_numeric(df[c], errors="coerce").var(skipna=True), reverse=True)[0]
    if hist_col:
        s = pd.to_numeric(df[hist_col], errors="coerce").dropna()
        figs["histogram"] = px.histogram(s, x=hist_col, nbins=50).update_layout(
            xaxis_title=hist_col, yaxis_title="Count", margin=dict(l=40, r=20, t=20, b=40)
        )

    # top categories: prefer category/payment_method/state/city
    for pref in ["category", "payment_method", "state", "city"]:
        if pref in df.columns:
            cat_col = pref
            break
    else:
        cat_col = cat_cols[0] if cat_cols else None
    if cat_col:
        vc = df[cat_col].astype("category").value_counts(dropna=False).head(20)
        figs["top_categories"] = px.bar(x=vc.index.astype(str), y=vc.values).update_layout(
            xaxis_title=cat_col, yaxis_title="Count", margin=dict(l=40, r=20, t=20, b=120), xaxis_tickangle=45
        )

    # timeseries: prefer event_ts + value
    ts = mapping.get("event_ts")
    if ts and ts in df.columns:
        w_sum, _ = _weekly_series(df, ts, value_col=val if (val and val in df.columns) else None)
        figs["timeseries"] = px.line(x=w_sum.index, y=w_sum.values).update_layout(
            xaxis_title="Date", yaxis_title=("Sum(value)" if val else "Count"),
            margin=dict(l=40, r=20, t=20, b=40), xaxis_rangeslider_visible=True
        )
    return figs


def _top3_takeaways(kpis: Dict[str, object], drivers4: pd.DataFrame) -> List[str]:
    out: List[str] = []
    v = kpis.get("value_last4_vs_prior4")
    if v and v.get("pct_change") is not None:
        pct = v["pct_change"]
        direction = "up" if pct >= 0 else "down"
        out.append(f"Weekly value is {direction} {abs(pct):.1f}% over the last 4 weeks vs. prior 4.")
    c = kpis.get("count_last4_vs_prior4")
    if c and c.get("pct_change") is not None:
        pct = c["pct_change"]
        direction = "up" if pct >= 0 else "down"
        out.append(f"Weekly count is {direction} {abs(pct):.1f}% over the last 4 weeks vs. prior 4.")
    if drivers4 is not None and not drivers4.empty:
        top = drivers4.sort_values("value", ascending=False).iloc[0]
        out.append(f"Top recent driver: {top['dimension']} = {top['member']}.")
    if not out:
        out.append("The dataset loaded successfully and key summaries are available.")
    return out[:3]


def _schema_json(raw: pd.DataFrame, cleaned: pd.DataFrame, mapping: Dict[str, Optional[str]], dq: Dict[str, object]) -> Dict[str, object]:
    def sample_vals(s: pd.Series) -> List[str]:
        return [str(v) for v in s.dropna().unique()[:5]]
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "roles": {k: v for k, v in mapping.items() if v},
        "columns": [
            {"name": c, "dtype_after": str(cleaned[c].dtype), "examples": sample_vals(cleaned[c])}
            for c in cleaned.columns
        ],
        "dq_before_after": dq,
        "notes": "CSV written as flat text; Parquet includes typed datetimes if pyarrow is available."
    }


# ------------------------ main entry ------------------------

def analyze_and_export(file_path: Optional[str] = None,
                       table_name: Optional[str] = None,
                       save_clean_table: Optional[str] = None) -> Dict[str, str]:
    _ensure_dirs()

    if not file_path and not table_name:
        raise ValueError("Provide a CSV file path or a Postgres table name")

    # load
    if table_name:
        raw_df, _ = load_df_from_source(table_name=table_name)
        basename = table_name.replace(".", "_")
    else:
        raw_df, _ = load_df_from_source(file_path=file_path)
        basename = Path(file_path).stem

    # auto-map + semantics
    mapped_df, mapping = _auto_map(raw_df)

    # primary cleaner with fallback
    try:
        cleaned_df, clean_report = clean_dataframe(
            mapped_df,
            id_col=mapping.get("entity_id"),
            ts_col=mapping.get("event_ts"),
            drop_exact_dups=True,
            drop_dups_by_id_ts=True,
            trim_strings=True,
            winsorize_numeric=True,
            winsor_upper_q=0.99,
            impute_numeric=True,
            impute_categorical=True,
            categorical_fill_value="UNKNOWN"
        )
    except Exception as e:
        cleaned_df = mapped_df.copy()
        for i, c in enumerate(cleaned_df.columns):
            s = cleaned_df.iloc[:, i]
            if is_object_dtype(s) or is_string_dtype(s):
                cleaned_df.iloc[:, i] = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        ent = mapping.get("entity_id")
        ts  = mapping.get("event_ts")
        if ent and ent in cleaned_df.columns and ts and ts in cleaned_df.columns:
            cleaned_df[ts] = pd.to_datetime(cleaned_df[ts], errors="coerce")
            cleaned_df = cleaned_df.drop_duplicates(subset=[ent, ts])
        else:
            cleaned_df = cleaned_df.drop_duplicates()
        num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            med = cleaned_df[c].median()
            cleaned_df[c] = cleaned_df[c].fillna(med)
        cat_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
        for c in cat_cols:
            mode = cleaned_df[c].mode(dropna=True)
            cleaned_df[c] = cleaned_df[c].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")
        for c in num_cols:
            uq = cleaned_df[c].quantile(0.99)
            cleaned_df[c] = np.where(cleaned_df[c] > uq, uq, cleaned_df[c])
        clean_report = f"Fallback cleaning used because primary cleaner failed: {e}"

    # enforce friendly types for business fields
    if "order_id" in cleaned_df.columns:
        with pd.option_context("mode.use_inf_as_na", True):
            tmp = pd.to_numeric(cleaned_df["order_id"], errors="coerce").astype("Int64")
        cleaned_df["order_id"] = tmp
    if "review_score" in cleaned_df.columns:
        tmp = pd.to_numeric(cleaned_df["review_score"], errors="coerce").round().astype("Int64")
        cleaned_df["review_score"] = tmp

    # KPIs, drivers, anomalies
    k = _kpis(cleaned_df, mapping)
    drivers4 = _drivers_last4(cleaned_df, mapping)

    # DQ before/after
    dq = _quality_before_after(mapped_df, cleaned_df, mapping)

    # takeaways
    takeaways = _top3_takeaways(k, drivers4)

    # figures
    figs = _plots(cleaned_df, mapping)

    # render HTML
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"outputs/reports/{basename}_insight_{ts}.html"
    html = _render_html(takeaways, mapping, k, dq, drivers4, figs)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    # exports
    clean_path = f"outputs/clean/{basename}_clean_{ts}.csv"
    cleaned_df.to_csv(clean_path, index=False)

    parquet_path = None
    try:
        parquet_path = f"outputs/clean/{basename}_clean_{ts}.parquet"
        cleaned_df.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = None  # pyarrow not installed, skip silently

    # drivers export
    drivers_path = f"outputs/drivers/{basename}_drivers_last4w_{ts}.csv"
    if drivers4 is not None and not drivers4.empty:
        drivers4.to_csv(drivers_path, index=False)
    else:
        pd.DataFrame(columns=["dimension", "member", "value"]).to_csv(drivers_path, index=False)

    # schema json
    schema_obj = _schema_json(mapped_df, cleaned_df, mapping, dq)
    schema_path = f"outputs/clean/{basename}_schema_{ts}.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_obj, f, indent=2)

    # optional save to DB
    table_msg = ""
    if save_clean_table and save_clean_table.strip():
        try:
            full, n = save_clean_to_postgres(cleaned_df, save_clean_table.strip(), schema="clean", if_exists="replace")
            table_msg = f"Saved {n} rows to {full}."
        except Exception as e:
            table_msg = f"Save to Postgres failed: {e}"

    # bundle zip
    bundle_path = f"outputs/bundles/{basename}_bundle_{ts}.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(report_path, arcname=Path(report_path).name)
        z.write(clean_path, arcname=Path(clean_path).name)
        if parquet_path:
            z.write(parquet_path, arcname=Path(parquet_path).name)
        z.write(schema_path, arcname=Path(schema_path).name)
        z.write(drivers_path, arcname=Path(drivers_path).name)

    return {
        "top3_md": "\n".join([f"- {t}" for t in takeaways]),
        "quality_md": f"`\n{json.dumps(dq, indent=2)}\n`",
        "report_path": report_path,
        "clean_path": clean_path,
        "bundle_path": bundle_path,
        "table_msg": table_msg
    }
# --- SAFE CAST HELPERS (append near top, after imports) ---
import numpy as np
import pandas as pd

def _safe_business_casts(df: pd.DataFrame) -> pd.DataFrame:
    if "order_id" in df.columns:
        s = pd.to_numeric(df["order_id"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frac_ok = s.dropna().mod(1).eq(0).all()
        if frac_ok:
            df["order_id"] = s.astype("Int64")
        else:
            df["order_id"] = (
                df["order_id"].astype(str).str.strip()
                .str.replace(r"\D+", "", regex=True)
                .replace({"": pd.NA})
            )
    if "review_score" in df.columns:
        s = pd.to_numeric(df["review_score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df["review_score"] = s.round().astype("Int64")
    return df
