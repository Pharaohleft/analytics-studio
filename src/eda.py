import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import plotly.express as px

def load_df_from_source(file_path: Optional[str]=None,
                        table_name: Optional[str]=None,
                        engine=None,
                        sql_limit: int = 200000) -> Tuple[pd.DataFrame, str]:
    """
    Load a DataFrame either from a CSV file or a Postgres table.
    If both are provided, table_name takes precedence.
    """
    if table_name:
        if engine is None:
            raise ValueError("engine is required when loading from a table")
        q = f'SELECT * FROM {table_name} LIMIT {int(sql_limit)}'
        df = pd.read_sql(q, con=engine)
        return df, f"Loaded {len(df)} rows from table '{table_name}' (limit {sql_limit})."
    if file_path:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        df = pd.read_csv(p, low_memory=False)
        return df, f"Loaded {len(df)} rows from file '{p.name}'."
    raise ValueError("Provide either file_path or table_name")

def infer_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Return (numeric_cols, categorical_cols, datetime_cols) based on dtypes and simple parsing.
    """
    import warnings
    # Heuristics for datetime-like names
    dt_candidates = []
    for c in df.columns:
        if any(k in c.lower() for k in ["date","datetime","timestamp","ts","time","created","updated","order","game","visit"]):
            dt_candidates.append(c)

    df_copy = df.copy()
    parsed_dt = []
    for c in dt_candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                s = pd.to_datetime(df_copy[c], errors="coerce", utc=False)
            if s.notna().sum() > 0:
                df_copy[c] = s
                parsed_dt.append(c)
        except Exception:
            pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Treat low-cardinality non-numeric as categorical
    cat_cols = []
    for c in df.columns:
        if c in numeric_cols:
            continue
        nunique = df[c].nunique(dropna=True)
        if nunique <= max(50, int(len(df) * 0.05)) or df[c].dtype == "object":
            cat_cols.append(c)

    datetime_cols = parsed_dt
    return numeric_cols, cat_cols, datetime_cols


def profile_basic(df: pd.DataFrame, top_n: int = 15) -> str:
    n_rows, n_cols = df.shape
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024**2), 2)
    # Missingness
    miss = df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / max(1, len(df)) * 100).round(2)
    miss_tbl = pd.DataFrame({"missing": miss, "missing_pct": miss_pct}).head(top_n)
    # Dtypes table
    rows = []
    for c in df.columns[:top_n]:
        s = df[c]
        dtype = str(s.dtype)
        uniq = int(s.nunique(dropna=True))
        miss_c = int(s.isna().sum())
        miss_pct_c = round(miss_c / max(1, len(df)) * 100, 2)
        sample_vals = list(s.dropna().astype(str).head(3))
        rows.append([c, dtype, uniq, miss_c, miss_pct_c, sample_vals])
    dtypes_tbl = pd.DataFrame(rows, columns=["column","dtype","unique","missing","missing_pct","sample_values"])
    out = []
    out.append("### File Summary")
    out.append(f"- Rows: {n_rows}, Cols: {n_cols}, Memory: {mem_mb} MB")
    out.append("")
    out.append("### Missingness (top)")
    out.append(miss_tbl.to_markdown())
    out.append("")
    out.append("### Dtypes & Samples (top)")
    out.append(dtypes_tbl.to_markdown(index=False))
    return "\n".join(out)

def plot_missingness_html(df: pd.DataFrame, top_n: int = 50) -> str:
    miss_pct = (df.isna().sum() / max(1, len(df)) * 100).sort_values(ascending=False)
    miss_df = miss_pct.head(top_n).reset_index()
    miss_df.columns = ["column","missing_pct"]
    fig = px.bar(miss_df, x="column", y="missing_pct", title="Missingness by Column (%)")
    fig.update_layout(xaxis_title="column", yaxis_title="missing %", xaxis_tickangle=-45, margin=dict(l=40,r=10,t=40,b=120))
    return fig.to_html(full_html=False, include_plotlyjs="inline")

def plot_numeric_hist_html(df: pd.DataFrame, col: str, bins: int = 50) -> str:
    if col not in df.columns:
        return "<p>Column not found.</p>"
    s = pd.to_numeric(df[col], errors="coerce")
    fig = px.histogram(s, x=col, nbins=bins, title=f"Histogram: {col}")
    fig.update_layout(margin=dict(l=40,r=10,t=40,b=40))
    return fig.to_html(full_html=False, include_plotlyjs="inline")

def plot_top_categories_html(df: pd.DataFrame, col: str, top_n: int = 20) -> str:
    if col not in df.columns:
        return "<p>Column not found.</p>"
    vc = df[col].astype(str).value_counts(dropna=True).head(top_n).reset_index()
    vc.columns = [col, "count"]
    fig = px.bar(vc, x=col, y="count", title=f"Top {top_n} categories: {col}")
    fig.update_layout(xaxis_tickangle=-45, margin=dict(l=40,r=10,t=40,b=120))
    return fig.to_html(full_html=False, include_plotlyjs="inline")

def plot_timeseries_html(df: pd.DataFrame, ts_col: str, value_col: Optional[str]=None, freq: str = "W") -> str:
    if ts_col not in df.columns:
        return "<p>Timestamp column not found.</p>"
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    ok = ts.notna()
    if ok.sum() == 0:
        return "<p>No parsable timestamps in selected column.</p>"
    work = df.loc[ok].copy()
    work[ts_col] = pd.to_datetime(work[ts_col], errors="coerce")
    if value_col and value_col in work.columns:
        # Sum value per period
        g = work.set_index(ts_col)[value_col].astype(float).resample(freq).sum().reset_index()
        y = value_col
        title = f"{value_col} over time ({freq} resample)"
    else:
        # Count events per period
        g = work.set_index(ts_col).assign(cnt=1)["cnt"].resample(freq).sum().reset_index()
        y = "cnt"
        title = f"Event count over time ({freq} resample)"
    g.columns = [ts_col, y]
    fig = px.line(g, x=ts_col, y=y, markers=True, title=title)
    fig.update_layout(margin=dict(l=40,r=10,t=40,b=40))
    return fig.to_html(full_html=False, include_plotlyjs="inline")

import pandas as pd
import numpy as np
import plotly.express as px

def plot_missingness_fig(df: pd.DataFrame):
    miss = df.isna().mean().sort_values(ascending=False)
    ser = miss[miss > 0].head(50)
    if ser.empty:
        ser = miss.head(50)
    fig = px.bar(x=ser.index, y=(ser * 100))
    fig.update_layout(
        xaxis_title="Column", yaxis_title="% Missing",
        margin=dict(l=40, r=20, t=20, b=120), xaxis_tickangle=45
    )
    return fig

def plot_numeric_hist_fig(df: pd.DataFrame, num_col: str, bins: int = 50):
    s = pd.to_numeric(df[num_col], errors="coerce").dropna()
    fig = px.histogram(s, x=num_col, nbins=int(bins))
    fig.update_layout(
        xaxis_title=num_col, yaxis_title="Count",
        margin=dict(l=40, r=20, t=20, b=40)
    )
    return fig

def plot_top_categories_fig(df: pd.DataFrame, cat_col: str, top_n: int = 20):
    vc = df[cat_col].astype("category").value_counts(dropna=False).head(int(top_n))
    x = vc.index.astype(str)
    y = vc.values
    fig = px.bar(x=x, y=y)
    fig.update_layout(
        xaxis_title=cat_col, yaxis_title="Count",
        margin=dict(l=40, r=20, t=20, b=120), xaxis_tickangle=45
    )
    return fig

def plot_timeseries_fig(df: pd.DataFrame, ts_col: str, value_col: str | None = None, freq: str = "W"):
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if value_col:
        val = pd.to_numeric(df[value_col], errors="coerce")
        g = (pd.DataFrame({"ts": ts, "val": val})
               .dropna(subset=["ts"])
               .groupby(pd.Grouper(key="ts", freq=freq))["val"]
               .sum())
        ylab = f"Sum of {value_col}"
    else:
        g = (pd.DataFrame({"ts": ts})
               .dropna(subset=["ts"])
               .groupby(pd.Grouper(key="ts", freq=freq))
               .size())
        ylab = "Count"
    fig = px.line(x=g.index, y=g.values)
    fig.update_layout(
        xaxis_title="Date", yaxis_title=ylab,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_rangeslider_visible=True
    )
    return fig


# --- HOTFIX: robust plotly HTML (always includes JS) ---
import pandas as pd
import plotly.express as px
import plotly.io as pio

def _fig_to_html(fig):
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

def plot_missingness_html(df):
    miss = df.isna().mean().sort_values(ascending=False)
    ser = miss[miss > 0].head(50) if (miss > 0).any() else miss.head(50)
    fig = px.bar(x=ser.index, y=ser.values * 100)
    fig.update_layout(xaxis_title="Column", yaxis_title="% Missing",
                      margin=dict(l=40, r=20, t=20, b=120), xaxis_tickangle=45)
    return _fig_to_html(fig)

def plot_numeric_hist_html(df, num_col: str, bins: int = 50):
    s = pd.to_numeric(df[num_col], errors="coerce").dropna()
    fig = px.histogram(s, x=num_col, nbins=int(bins))
    fig.update_layout(xaxis_title=num_col, yaxis_title="Count",
                      margin=dict(l=40, r=20, t=20, b=40))
    return _fig_to_html(fig)

def plot_top_categories_html(df, cat_col: str, top_n: int = 20):
    vc = df[cat_col].astype("category").value_counts(dropna=False).head(int(top_n))
    fig = px.bar(x=vc.index.astype(str), y=vc.values)
    fig.update_layout(xaxis_title=cat_col, yaxis_title="Count",
                      margin=dict(l=40, r=20, t=20, b=120), xaxis_tickangle=45)
    return _fig_to_html(fig)

def plot_timeseries_html(df, ts_col: str, value_col: str = None, freq: str = "W"):
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if value_col:
        val = pd.to_numeric(df[value_col], errors="coerce")
        g = (pd.DataFrame({"ts": ts, "val": val})
               .dropna(subset=["ts"])
               .groupby(pd.Grouper(key="ts", freq=freq))["val"].sum())
        ylab = f"Sum({value_col})"
    else:
        g = (pd.DataFrame({"ts": ts})
               .dropna(subset=["ts"])
               .groupby(pd.Grouper(key="ts", freq=freq)).size())
        ylab = "Count"
    fig = px.line(x=g.index, y=g.values)
    fig.update_layout(xaxis_title="Date", yaxis_title=ylab,
                      margin=dict(l=40, r=20, t=20, b=40),
                      xaxis_rangeslider_visible=True)
    return _fig_to_html(fig)
# --- END HOTFIX ---
