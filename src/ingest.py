import re
import pandas as pd
from pathlib import Path
from sqlalchemy import text
from .db import get_engine

DATE_HINTS = {'date','dt','datetime','timestamp','order_date','signup_date','created_at','updated_at'}

def _clean_col(c: str) -> str:
    c = c.strip()
    c = re.sub(r'[^0-9a-zA-Z_]+', '_', c)
    c = re.sub(r'_+', '_', c)
    return c.strip('_').lower()

def _maybe_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if any(h in col for h in DATE_HINTS):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    return df

def safe_table_name(name: str) -> str:
    name = _clean_col(name)
    if not name:
        name = 'uploaded'
    return name

def ingest_csv(csv_path: str, table: str):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df.columns = [_clean_col(c) for c in df.columns]
    df = _maybe_parse_dates(df)

    eng = get_engine()
    tbl = safe_table_name(table)
    with eng.begin() as conn:
        # create schema if using dot notation like schema.table
        if '.' in tbl:
            schema, t = tbl.split('.', 1)
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        # write (replace for now; we can switch to append later)
        df.to_sql(tbl.split('.')[-1], con=conn, if_exists='replace', index=False, schema=(tbl.split('.')[0] if '.' in tbl else None))
    return tbl, len(df)
