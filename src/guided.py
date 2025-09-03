from __future__ import annotations
from typing import Optional, Dict
import pandas as pd

from src.eda import load_df_from_source
from src.schema_map import suggest_mapping, apply_mapping
from src.quality_gate import run_quality_checks
from src.pii import scan_pii
from src.insight import analyze_and_export

def run_end_to_end(file_path: Optional[str] = None,
                   table_name: Optional[str] = None,
                   save_clean_table: Optional[str] = None,
                   force: bool = False) -> Dict[str, str]:
    if not file_path and not table_name:
        raise ValueError("Provide file_path or table_name")
    if table_name:
        raw, _ = load_df_from_source(table_name=table_name)
    else:
        raw, _ = load_df_from_source(file_path=file_path)

    sugg = suggest_mapping(raw)
    def first(role): return (sugg.get(role) or [[None,0]])[0][0]
    mapping = {
        "entity_id": first("entity_id"),
        "event_ts": first("event_ts"),
        "value": first("value"),
        "quantity": first("quantity"),
        "category": first("category"),
    }
    mapped = apply_mapping(raw, mapping)

    pii_hits = scan_pii(mapped)
    pass_q, md_q, details_q = run_quality_checks(
        mapped,
        id_col=mapping.get("entity_id"),
        ts_col=mapping.get("event_ts"),
        max_missing_pct=0.20, max_dup_exact=0.05, max_dup_pair=0.02, max_ts_staleness_days=None
    )
    gate_md = md_q + ("\n\n**PII detected:** " + ", ".join(h["column"] for h in pii_hits) if pii_hits else "\n\nNo obvious PII detected.")

    if not pass_q and not force:
        return { "status_md": "**Stopped by Data Quality Gate** (fix data or run with force=True)\n\n" + gate_md }

    res = analyze_and_export(
        file_path=file_path,
        table_name=table_name,
        save_clean_table=save_clean_table
    )
    res["status_md"] = "**Completed** via Wizard\n\n" + gate_md
    return res
