import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from datetime import datetime
from pathlib import Path
from sqlalchemy import text

from src.db import get_engine
from src.ingest import ingest_csv
from src.validate import validate_csv
from src.eda import (
    load_df_from_source, infer_columns, profile_basic
)
from src.schema_map import suggest_mapping, apply_mapping, save_to_postgres
from src.clean import clean_dataframe, save_clean_to_postgres
from src.insight import analyze_and_export
from src.quality_gate import run_quality_checks
from src.pii import scan_pii

TAB_CSS = """
.gradio-container .tabs > .tab-nav { flex-wrap: wrap; overflow-x: auto; }
.gradio-container .tabs > .tab-nav button { padding: 0.35rem 0.6rem; font-size: 0.9rem; }
"""

# ---------------- Helpers: ID / defaults ----------------

def _is_id_like(name: str, s: pd.Series) -> bool:
    n = (name or "").lower()
    if any(k in n for k in ["id", "uuid", "guid", "ssn"]):
        return True
    try:
        if s.nunique(dropna=True) / max(1, len(s)) > 0.9:
            return True
    except Exception:
        pass
    try:
        if pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing:
            return True
    except Exception:
        pass
    return False

def _best_numeric(df: pd.DataFrame, numeric_cols):
    if "value" in numeric_cols:
        return "value"
    cand = []
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if not _is_id_like(c, s):
            cand.append((c, float(np.nanvar(s))))
    if not cand and numeric_cols:
        return numeric_cols[0]
    return sorted(cand, key=lambda x: x[1], reverse=True)[0][0] if cand else None

def _best_categorical(df: pd.DataFrame, cat_cols):
    for pref in ["category", "payment_method", "state", "city"]:
        if pref in df.columns and pref in cat_cols:
            return pref
    return cat_cols[0] if cat_cols else None

def _best_timestamp(dt_cols):
    for pref in ["event_ts", "order_date", "timestamp", "created_at", "date"]:
        if pref in dt_cols:
            return pref
    return dt_cols[0] if dt_cols else None

# ---------------- Plotly figures (not HTML) ----------------

def fig_missingness(df: pd.DataFrame):
    miss = df.isna().mean().sort_values(ascending=False)
    # Show all columns with >0 missingness, else show top 30 columns with 0%
    show = miss[miss > 0]
    if show.empty:
        show = miss.head(min(30, len(miss)))
    fig = px.bar(x=show.index, y=(show.values * 100.0), labels={"x": "Column", "y": "% Missing"})
    fig.update_layout(margin=dict(l=40, r=20, t=20, b=120), xaxis_tickangle=45)
    return fig

def fig_histogram(df: pd.DataFrame, col: str, bins: int = 50):
    s = pd.to_numeric(df[col], errors="coerce")
    fig = px.histogram(x=s, nbins=int(bins), labels={"x": col, "y": "Count"})
    fig.update_layout(margin=dict(l=40, r=20, t=20, b=40))
    return fig

def fig_top_categories(df: pd.DataFrame, col: str, topn: int = 20):
    vc = (df[col].astype(str).fillna("")).value_counts().head(int(topn))
    fig = px.bar(y=vc.index[::-1], x=vc.values[::-1],
                 orientation="h", labels={"x": "Count", "y": col})
    fig.update_layout(margin=dict(l=100, r=20, t=20, b=40))
    return fig

def fig_timeseries(df: pd.DataFrame, ts_col: str, val_col: str | None, qty_col: str | None, metric: str, freq: str):
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    base = pd.DataFrame({"ts": ts}).dropna(subset=["ts"])
    metric = metric or "Count"
    freq = freq or "W"
    if metric == "Count" or not val_col:
        series = base.groupby(pd.Grouper(key="ts", freq=freq)).size()
        ylab = "Count"
    elif metric == "Sum(value)":
        val = pd.to_numeric(df[val_col], errors="coerce")
        series = pd.DataFrame({"ts": ts, "val": val}).dropna(subset=["ts"]).groupby(pd.Grouper(key="ts", freq=freq))["val"].sum()
        ylab = f"Sum({val_col})"
    else:  # Avg(value)
        val = pd.to_numeric(df[val_col], errors="coerce")
        temp = pd.DataFrame({"ts": ts, "val": val}).dropna(subset=["ts"])
        if qty_col and qty_col in df.columns:
            qty = pd.to_numeric(df[qty_col], errors="coerce")
            temp["qty"] = qty
            agg = temp.groupby(pd.Grouper(key="ts", freq=freq)).agg({"val": "sum", "qty": "sum"})
            series = agg["val"] / agg["qty"].replace({0: np.nan})
            ylab = f"Avg({val_col}/qty)"
        else:
            series = temp.groupby(pd.Grouper(key="ts", freq=freq))["val"].mean()
            ylab = f"Avg({val_col})"
    fig = px.line(x=series.index, y=series.values)
    fig.update_layout(xaxis_title="Date", yaxis_title=ylab, margin=dict(l=40, r=20, t=20, b=40), xaxis_rangeslider_visible=True)
    return fig

# ---------------- File builders ----------------

def build_flow_report(df: pd.DataFrame,
                      profile_md: str,
                      figs: list,
                      insights_md: str,
                      quality_md: str,
                      gate_md: str,
                      pii_md: str,
                      table_msg: str | None,
                      out_dir: str = "outputs/reports") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{out_dir}/flow_insight_{ts}.html"

    # Prepare plotly blocks: include plotly.js once
    blocks = []
    for i, fg in enumerate(figs):
        blocks.append(pio.to_html(fg, include_plotlyjs=("cdn" if i == 0 else False), full_html=False))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Flow Insight Pack</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
h1,h2 {{ margin: 0.6rem 0; }}
section {{ margin-bottom: 2rem; }}
pre {{ background: #f6f8fa; padding: 12px; border-radius: 8px; overflow: auto; }}
hr {{ border: none; border-top: 1px solid #e2e2e2; margin: 1.5rem 0; }}
</style>
</head>
<body>
<h1>Flow Insight Pack</h1>
<section>
  <h2>Profile</h2>
  <pre>{profile_md}</pre>
</section>
<section>
  <h2>Charts</h2>
  <h3>Missingness</h3>
  {blocks[0] if len(blocks)>0 else "<p>No chart.</p>"}
  <h3>Histogram</h3>
  {blocks[1] if len(blocks)>1 else "<p>No chart.</p>"}
  <h3>Top Categories</h3>
  {blocks[2] if len(blocks)>2 else "<p>No chart.</p>"}
  <h3>Timeseries</h3>
  {blocks[3] if len(blocks)>3 else "<p>No chart.</p>"}
</section>
<section>
  <h2>Quality & PII</h2>
  <pre>{gate_md}</pre>
  <pre>{pii_md}</pre>
  <pre>{quality_md}</pre>
  <pre>{(table_msg or "").strip()}</pre>
</section>
<section>
  <h2>Insights</h2>
  <pre>{insights_md}</pre>
</section>
<hr />
<footer>Generated {datetime.now().isoformat(timespec="seconds")}</footer>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path

# ---------------- Minimal utilities we keep ----------------

def preview_split(file):
    if not file:
        return pd.DataFrame(), "Upload a CSV to preview."
    try:
        df = pd.read_csv(file.name)
        head5 = df.head(5)
        notes = f"Rows: {len(df)} | Cols: {len(df.columns)}\n\nColumns: {list(df.columns)}"
        return head5, notes
    except Exception as e:
        return pd.DataFrame(), f"ERROR Read error: {e}"

def load_to_db(file, table_name):
    if not file:
        return "ERROR No file."
    if not table_name or table_name.strip() == "":
        return "ERROR Enter a table name (e.g., staging.orders or customers)."
    try:
        tbl, n = ingest_csv(file.name, table_name.strip())
        return f"OK Loaded {n} rows into table {tbl}."
    except Exception as e:
        return f"ERROR Ingest error: {e}"

def system_check():
    report = []
    try:
        import numpy, sklearn, plotly, sqlalchemy, tabulate  # noqa
        report.append("Libraries: OK pandas / numpy / scikit-learn / plotly / SQLAlchemy / tabulate present")
    except Exception as e:
        report.append(f"Libraries: ERROR {e}")
    try:
        eng = get_engine()
        with eng.begin() as conn:
            ver = conn.execute(text("SELECT version()")).scalar()
            report.append(f"DB connect: OK {ver}")
            test_df = pd.DataFrame({"id": [1, 2], "note": ["hello", "world"]})
            conn.exec_driver_sql('DROP TABLE IF EXISTS public._ui_smoketest')
            test_df.to_sql("_ui_smoketest", con=conn, if_exists="replace", index=False, schema="public")
            cnt = conn.execute(text('SELECT COUNT(*) FROM public."_ui_smoketest"')).scalar()
            conn.exec_driver_sql('DROP TABLE IF EXISTS public._ui_smoketest')
            report.append("DB round-trip: OK wrote/read 2 rows" if cnt == 2 else f"DB round-trip: ERROR expected 2, got {cnt}")
    except Exception as e:
        report.append(f"Database: ERROR {e}")
    try:
        import plotly.express as px  # noqa
        report.append("Plotting: OK Plotly import")
    except Exception as e:
        report.append(f"Plotting: ERROR {e}")
    return "\n".join(f"- {line}" for line in report)

def run_validate(file, id_hint, ts_hint, val_hint):
    if not file:
        return "ERROR No file."
    return validate_csv(
        file.name,
        id_hint=(id_hint.strip() or None) if id_hint else None,
        ts_hint=(ts_hint.strip() or None) if ts_hint else None,
        val_hint=(val_hint.strip() or None) if val_hint else None,
    )

# ---------------- One-click FLOW ----------------

def run_flow(file, table_name, sql_limit, save_table, bins, topn, metric, freq):
    try:
        # Load
        if table_name and table_name.strip():
            eng = get_engine()
            df, _ = load_df_from_source(table_name=table_name.strip(), engine=eng, sql_limit=int(sql_limit or 200000))
        elif file:
            df, _ = load_df_from_source(file_path=file.name)
        else:
            return ("ERROR Provide a CSV or a Postgres table.",
                    None, None, None, None,
                    "", None, None, None, "")

        # Profile + inference
        prof = profile_basic(df)
        numeric_cols, cat_cols, dt_cols = infer_columns(df)
        num_def = _best_numeric(df, numeric_cols)
        cat_def = _best_categorical(df, cat_cols)
        dt_def = _best_timestamp(dt_cols)
        val_def = "value" if "value" in numeric_cols else num_def

        # Charts (figures)
        f_missing = fig_missingness(df)
        f_hist = fig_histogram(df, num_def, bins=int(bins or 50)) if num_def else px.scatter()
        f_cats = fig_top_categories(df, cat_def, topn=int(topn or 20)) if cat_def else px.scatter()
        qty_col = "quantity" if "quantity" in df.columns else None
        f_ts = fig_timeseries(df, dt_def, val_def, qty_col, metric or "Sum(value)", freq or "W") if dt_def else px.scatter()

        # Gate + PII
        id_guess = next((c for c in df.columns if c.lower() in ["customer_id","user_id","account_id","id","entity_id"]), None)
        pass_q, md_q, _ = run_quality_checks(df, id_col=id_guess, ts_col=dt_def, max_missing_pct=0.20, max_dup_exact=0.05, max_dup_pair=0.02, max_ts_staleness_days=None)
        pii_hits = scan_pii(df)
        pii_md = ("PII scan: " + (", ".join([h["column"] for h in pii_hits]) if pii_hits else "no obvious hits."))

        # Analyze & export
        res = analyze_and_export(
            file_path=(file.name if file else None),
            table_name=(table_name.strip() if table_name else None),
            save_clean_table=(save_table.strip() if save_table else None),
        )
        insights = res["top3_md"]
        quality = res["quality_md"]
        clean_path = res["clean_path"]
        bundle_path = res.get("bundle_path")
        table_msg = (res.get("table_msg") or "")

        # Build combined Flow report including charts
        flow_path = build_flow_report(
            df=df,
            profile_md=prof,
            figs=[f_missing, f_hist, f_cats, f_ts],
            insights_md=insights,
            quality_md=quality,
            gate_md=md_q,
            pii_md=pii_md,
            table_msg=table_msg
        )

        return (prof, f_missing, f_hist, f_cats, f_ts, insights, flow_path, clean_path, bundle_path, table_msg)

    except Exception as e:
        return (f"ERROR {e}", None, None, None, None, "", None, None, None, "")

# ---------------- Advanced (unchanged logic) ----------------

def mapper_suggest(file):
    empty = gr.update(choices=[], value=None)
    if not file:
        return "ERROR No file.", empty, empty, empty, empty, empty, empty, empty, empty, empty
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"ERROR Could not read CSV: {e}", empty, empty, empty, empty, empty, empty, empty, empty, empty
    sugg = suggest_mapping(df)
    cols = list(df.columns)
    def first_candidate(role: str):
        cands = sugg.get(role) or []
        return cands[0][0] if cands else None
    roles_list = ["entity_id","event_ts","value","quantity","order_id","product_id","category","city","state","country","zip","email","phone"]
    lines = ["### Suggested roles (top candidates)", ""]
    for role in roles_list:
        cands = sugg.get(role) or []
        show = ", ".join([f"{c} (score {s})" for c, s in cands]) if cands else "(no obvious match)"
        lines.append(f"- {role}: {show}")
    report = "\n".join(lines)
    def dd(role):
        cand = first_candidate(role)
        return gr.update(choices=cols, value=(cand if cand in cols else None))
    return (report,
            dd("entity_id"), dd("event_ts"), dd("value"), dd("quantity"),
            dd("order_id"), dd("product_id"), dd("category"), dd("city"), dd("state"))

def mapper_apply(file, table_name,
                 entity_id, event_ts, value, quantity, order_id, product_id, category, city, state):
    if not file:
        return pd.DataFrame(), "ERROR No file."
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return pd.DataFrame(), f"ERROR Could not read CSV: {e}"
    mapping = {
        "entity_id": entity_id or None, "event_ts": event_ts or None,
        "value": value or None, "quantity": quantity or None,
        "order_id": order_id or None, "product_id": product_id or None,
        "category": category or None, "city": city or None, "state": state or None,
    }
    df2 = apply_mapping(df, mapping)
    msg = [f"Applied mapping: { {k:v for k,v in mapping.items() if v} }",
           f"Result columns: {list(df2.columns)}"]
    if table_name and table_name.strip():
        try:
            full, n = save_to_postgres(df2, table_name.strip(), schema="staging", if_exists="replace")
            msg.append(f"Wrote {n} rows to {full}.")
        except Exception as e:
            msg.append(f"ERROR saving to Postgres: {e}")
    return df2.head(5), "\n".join(msg)

def clean_and_save(file, table_name, id_col, ts_col,
                   drop_exact, drop_pair, trim,
                   winsor, winsor_q, impute_num, impute_cat, cat_fill,
                   out_table):
    try:
        if table_name and table_name.strip():
            eng = get_engine()
            df, _ = load_df_from_source(table_name=table_name.strip(), engine=eng, sql_limit=2_000_000)
        elif file:
            df, _ = load_df_from_source(file_path=file.name)
        else:
            return pd.DataFrame(), "ERROR Provide a CSV or a Postgres table name."
        cleaned, report = clean_dataframe(
            df,
            id_col=(id_col.strip() or None) if id_col else None,
            ts_col=(ts_col.strip() or None) if ts_col else None,
            drop_exact_dups=bool(drop_exact),
            drop_dups_by_id_ts=bool(drop_pair),
            trim_strings=bool(trim),
            winsorize_numeric=bool(winsor),
            winsor_upper_q=float(winsor_q or 0.99),
            impute_numeric=bool(impute_num),
            impute_categorical=bool(impute_cat),
            categorical_fill_value=(cat_fill.strip() if cat_fill else "UNKNOWN"),
        )
        msg = ["### Cleaning report", "", report]
        if out_table and out_table.strip():
            try:
                full, n = save_clean_to_postgres(cleaned, out_table.strip(), schema="clean", if_exists="replace")
                msg.append(f"\nSaved {n} rows to {full}.")
            except Exception as e:
                msg.append(f"\nSave error: {e}")
        return cleaned.head(10), "\n".join(msg)
    except Exception as e:
        return pd.DataFrame(), f"ERROR Clean failed: {e}"

def analyze_ui(file, table_name, save_table):
    try:
        if (not file) and (not table_name or table_name.strip() == ""):
            return "ERROR Provide a CSV or a Postgres table.", "", None, None, None, ""
        res = analyze_and_export(
            file_path=(file.name if file else None),
            table_name=(table_name.strip() if table_name else None),
            save_clean_table=(save_table.strip() if save_table else None),
        )
        return res["top3_md"], res["quality_md"], res["report_path"], res["clean_path"], res.get("bundle_path"), res.get("table_msg","")
    except Exception as e:
        return f"ERROR {e}", "", None, None, None, ""

def make_pbids(table_name):
    if not table_name or not table_name.strip():
        return None, "Enter a table name (schema.table)."
    from src.powerbi import generate_pbids
    path = generate_pbids(table_name.strip())
    return path, f"Created PBIDS for {table_name.strip()}."

def train_ui(file, table):
    try:
        from src.ml import train_value_regressor
        res = train_value_regressor(file_path=(file.name if file else None),
                                    table_name=(table.strip() if table else None))
        return res["metrics_md"], res["model_path"], res["pred_path"]
    except Exception as e:
        return f"ERROR {e}", None, None

# ---------------- UI ----------------

with gr.Blocks(title="Analytics Studio", css=TAB_CSS) as demo:
    gr.Markdown("# Analytics Studio")

    with gr.Tabs():
        # ===== FLOW (all-in-one) =====
        with gr.Tab("Flow"):
            with gr.Row():
                file_src = gr.File(label="Upload CSV", file_types=[".csv"])
                table_src = gr.Textbox(label="Or Postgres table", placeholder="e.g., clean.events")
                sql_limit = gr.Number(value=200000, precision=0, label="SQL row limit")
            save_table = gr.Textbox(label="Also save cleaned to table (optional)", placeholder="clean.events")

            with gr.Row():
                bins_input = gr.Slider(minimum=5, maximum=200, value=50, step=1, label="Histogram bins")
                topn_input = gr.Radio(choices=[5, 10, 20], value=20, label="Top categories N")
                ts_metric = gr.Dropdown(choices=["Count", "Sum(value)", "Avg(value)"], value="Sum(value)", label="Timeseries metric")
                freq_input = gr.Dropdown(choices=["D", "W", "M"], value="W", label="Timeseries frequency")

            btn_run = gr.Button("Run Flow")

            with gr.Accordion("Quick Profile", open=True):
                prof_out = gr.Markdown()

            with gr.Accordion("Charts", open=True):
                with gr.Row():
                    plot_missing = gr.Plot(label="Missingness")
                    plot_hist = gr.Plot(label="Histogram")
                with gr.Row():
                    plot_cats = gr.Plot(label="Top Categories")
                    plot_ts = gr.Plot(label="Timeseries")

            with gr.Accordion("Insights", open=True):
                top3_out = gr.Markdown(label="Top 3 Takeaways")
                quality_out = gr.Markdown(label="Data Quality Report (before → after) + Gate + PII")

            with gr.Accordion("Downloads", open=True):
                report_file = gr.File(label="Flow Insight Pack (HTML)")
                clean_file = gr.File(label="Cleaned CSV")
                bundle_file = gr.File(label="Bundle (ZIP)")
                table_msg_out = gr.Markdown(label="Database Write")

            btn_run.click(
                run_flow,
                inputs=[file_src, table_src, sql_limit, save_table, bins_input, topn_input, ts_metric, freq_input],
                outputs=[prof_out, plot_missing, plot_hist, plot_cats, plot_ts, top3_out, report_file, clean_file, bundle_file, table_msg_out]
            )

        # ===== ADVANCED =====
        with gr.Tab("Advanced"):
            with gr.Accordion("Ingest / Preview", open=False):
                file_ing = gr.File(label="Upload CSV", file_types=[".csv"])
                table_ing = gr.Textbox(label="Target table name", placeholder="e.g., staging.orders or customers")
                with gr.Row():
                    btn_prev = gr.Button("Preview CSV")
                    btn_load = gr.Button("Load to Postgres")
                grid_prev = gr.DataFrame(label="Head (first 5 rows)", interactive=False)
                out_prev_notes = gr.Markdown()
                out_ingest = gr.Textbox(label="Load Output", lines=6)
                btn_prev.click(preview_split, inputs=file_ing, outputs=[grid_prev, out_prev_notes])
                btn_load.click(load_to_db, inputs=[file_ing, table_ing], outputs=out_ingest)

            with gr.Accordion("Map / Validate / Clean", open=False):
                # Map
                file_map = gr.File(label="Upload CSV", file_types=[".csv"])
                btn_suggest = gr.Button("Suggest mapping")
                suggest_md = gr.Markdown()
                with gr.Row():
                    entity_id = gr.Dropdown(label="Entity ID")
                    event_ts = gr.Dropdown(label="Timestamp")
                    value = gr.Dropdown(label="Value (amount)")
                    quantity = gr.Dropdown(label="Quantity")
                with gr.Row():
                    order_id = gr.Dropdown(label="Order/Txn ID")
                    product_id = gr.Dropdown(label="Product/Item ID")
                    category = gr.Dropdown(label="Category/Channel")
                    city = gr.Dropdown(label="City")
                    state = gr.Dropdown(label="State/Region")
                table_map = gr.Textbox(label="Save as table (optional)", placeholder="e.g., staging.events")
                btn_apply = gr.Button("Apply mapping (and save if table provided)")
                grid_mapped = gr.DataFrame(label="Preview of mapped data (head)", interactive=False)
                out_map_msg = gr.Markdown()
                btn_suggest.click(
                    mapper_suggest,
                    inputs=[file_map],
                    outputs=[suggest_md, entity_id, event_ts, value, quantity, order_id, product_id, category, city, state]
                )
                btn_apply.click(
                    mapper_apply,
                    inputs=[file_map, table_map, entity_id, event_ts, value, quantity, order_id, product_id, category, city, state],
                    outputs=[grid_mapped, out_map_msg]
                )

                # Validate
                file_val = gr.File(label="Upload CSV to Validate", file_types=[".csv"])
                with gr.Row():
                    id_hint = gr.Textbox(label="ID column (optional hint)", placeholder="e.g., customer_id")
                    ts_hint = gr.Textbox(label="Timestamp column (optional hint)", placeholder="e.g., order_date")
                    val_hint = gr.Textbox(label="Value column (optional hint)", placeholder="e.g., order_value")
                btn_val = gr.Button("Run Validation")
                out_val = gr.Markdown()
                btn_val.click(run_validate, inputs=[file_val, id_hint, ts_hint, val_hint], outputs=out_val)

                # Clean
                with gr.Row():
                    file_clean = gr.File(label="Upload CSV to clean", file_types=[".csv"])
                    table_clean = gr.Textbox(label="Or Postgres table name", placeholder="e.g., staging.events")
                with gr.Row():
                    id_col = gr.Textbox(label="ID column (optional for de-dup)", placeholder="e.g., customer_id")
                    ts_col = gr.Textbox(label="Timestamp column (optional for de-dup)", placeholder="e.g., event_ts or order_date")
                with gr.Row():
                    drop_exact = gr.Checkbox(value=True, label="Drop exact duplicate rows")
                    drop_pair = gr.Checkbox(value=True, label="Drop duplicates by (ID, Timestamp)")
                    trim = gr.Checkbox(value=True, label="Trim whitespace in string columns")
                with gr.Row():
                    winsor = gr.Checkbox(value=True, label="Winsorize numeric outliers")
                    winsor_q = gr.Slider(minimum=0.90, maximum=0.999, step=0.001, value=0.99, label="Winsor upper quantile")
                with gr.Row():
                    impute_num = gr.Checkbox(value=True, label="Impute missing numeric (median)")
                    impute_cat = gr.Checkbox(value=True, label="Impute missing categorical (mode/fallback)")
                    cat_fill = gr.Textbox(value="UNKNOWN", label="Categorical fallback if no mode")
                out_table = gr.Textbox(label="Write cleaned to table (optional)", placeholder="clean.events")
                btn_clean = gr.Button("Run Cleaning")
                grid_clean = gr.DataFrame(label="Cleaned preview (first 10 rows)", interactive=False)
                out_clean = gr.Markdown()
                btn_clean.click(
                    clean_and_save,
                    inputs=[file_clean, table_clean, id_col, ts_col,
                            drop_exact, drop_pair, trim,
                            winsor, winsor_q, impute_num, impute_cat, cat_fill,
                            out_table],
                    outputs=[grid_clean, out_clean]
                )

            with gr.Accordion("Analyze / Power BI / ML", open=False):
                # Analyze
                with gr.Row():
                    file_an = gr.File(label="Upload CSV", file_types=[".csv"])
                    table_an = gr.Textbox(label="Or Postgres table name", placeholder="e.g., staging.events")
                save_tbl = gr.Textbox(label="Also save cleaned to table (optional)", placeholder="clean.events")
                btn_an = gr.Button("Analyze my CSV")
                a_top3 = gr.Markdown(label="Top 3 Takeaways")
                a_quality = gr.Markdown(label="Data Quality Report")
                a_report = gr.File(label="Insight Pack (HTML)")
                a_clean = gr.File(label="Cleaned CSV")
                a_bundle = gr.File(label="Bundle (ZIP)")
                a_tablemsg = gr.Markdown(label="Database Write")
                btn_an.click(
                    analyze_ui,
                    inputs=[file_an, table_an, save_tbl],
                    outputs=[a_top3, a_quality, a_report, a_clean, a_bundle, a_tablemsg]
                )

                # Power BI
                table_pb = gr.Textbox(label="Table (schema.table)", placeholder="e.g., clean.events")
                btn_pb = gr.Button("Generate PBIDS")
                file_pb = gr.File(label="PBIDS file")
                pb_msg = gr.Markdown()
                btn_pb.click(make_pbids, inputs=[table_pb], outputs=[file_pb, pb_msg])

                # ML
                with gr.Row():
                    file_m = gr.File(label="CSV", file_types=[".csv"])
                    table_m = gr.Textbox(label="Or table name", placeholder="e.g., clean.events")
                btn_m = gr.Button("Train Model")
                m_metrics = gr.Markdown()
                m_model = gr.File(label="Model (joblib)")
                m_preds = gr.File(label="Predictions CSV")
                btn_m.click(train_ui, inputs=[file_m, table_m], outputs=[m_metrics, m_model, m_preds])

if __name__ == "__main__":
    demo.launch()
