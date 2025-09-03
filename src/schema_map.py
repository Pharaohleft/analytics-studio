import re
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from .db import get_engine
from sqlalchemy import text

# Canonical roles we care about
ROLES = [
    "entity_id",      # customer_id, user_id, patient_id, player_id, account_id...
    "event_ts",       # order_date, created_at, timestamp, game_date, visit_date...
    "value",          # order_value, amount, price, total, revenue, payment_value...
    "quantity",       # qty, quantity, units
    "order_id",       # order_id, transaction_id, invoice_id, ticket_id, claim_id
    "product_id",     # product_id, sku, item_id
    "category",       # category, segment, channel, payment_type, brand
    "city", "state", "country", "zip",
    "email", "phone",
]

# Synonyms/patterns for each role (broad coverage for common business datasets)
SYNONYMS: Dict[str, List[str]] = {
    "entity_id": [
        "customer", "client", "user", "account", "member", "subscriber", "patient",
        "player", "device", "lead", "contact", "visitor", "session", "person",
        "employee", "partner", "merchant", "seller", "buyer", "provider", "id"
    ],
    "event_ts": [
        "order_date", "purchase", "created", "updated", "date", "datetime", "timestamp",
        "event_time", "event_ts", "time", "game_date", "visit_date", "invoice_date",
        "billing_date", "occurred", "happened"
    ],
    "value": [
        "order_value", "amount", "price", "total", "revenue", "sales", "value",
        "payment_value", "gmv", "net", "gross", "charge", "cost", "premium", "fees"
    ],
    "quantity": ["qty", "quantity", "units", "items", "count"],
    "order_id": ["order_id", "transaction_id", "invoice_id", "sale_id", "ticket_id", "claim_id", "orderid", "txn_id"],
    "product_id": ["product_id", "sku", "item_id", "product", "productcode"],
    "category": ["category", "segment", "type", "class", "channel", "payment", "brand", "dept", "department"],
    "city": ["city", "town"],
    "state": ["state", "province", "region"],
    "country": ["country", "nation"],
    "zip": ["zip", "zipcode", "postal", "postcode"],
    "email": ["email", "e-mail"],
    "phone": ["phone", "msisdn", "mobile", "tel"],
}

NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")

def _clean_col(name: str) -> str:
    name = NON_ALNUM.sub("_", name.strip()).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    return name

def _tokenize(name: str) -> List[str]:
    return [t for t in NON_ALNUM.sub(" ", name.lower()).split() if t]

def _datetime_score(series: pd.Series) -> int:
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series, errors="coerce")
        ok = int(parsed.notna().sum())
        frac = ok / max(1, len(series))
        return 2 if frac >= 0.7 else (1 if frac >= 0.3 else 0)
    except Exception:
        return 0



def _id_cardinality_score(series: pd.Series) -> int:
    n = len(series)
    if n == 0:
        return 0
    uniq = series.nunique(dropna=True)
    ratio = uniq / n
    # Prefer moderately high cardinality
    if ratio >= 0.9:
        return 2
    if ratio >= 0.2:
        return 1
    return 0

def _numeric_score(series: pd.Series) -> int:
    return 1 if pd.api.types.is_numeric_dtype(series) else 0

def _match_score(colname: str, role: str) -> int:
    # lexical matching against synonyms
    tokens = set(_tokenize(colname))
    score = 0
    for syn in SYNONYMS.get(role, []):
        s_tokens = set(_tokenize(syn))
        if syn == colname.lower():
            score += 3
        elif syn in colname.lower():
            score += 2
        elif tokens & s_tokens:
            score += 1
    return score

def suggest_mapping(df: pd.DataFrame, top_k: int = 3) -> Dict[str, List[Tuple[str, int]]]:
    """
    Return for each role a list of (column_name, score) sorted by score desc.
    """
    out: Dict[str, List[Tuple[str, int]]] = {}
    for role in ROLES:
        scored: List[Tuple[str, int]] = []
        for c in df.columns:
            base = _match_score(c, role)
            boost = 0
            s = df[c]
            if role == "event_ts":
                boost += _datetime_score(s)
            elif role in ("value", "quantity"):
                boost += _numeric_score(s)
            elif role == "entity_id":
                boost += _id_cardinality_score(s)
            total = base + boost
            if total > 0:
                scored.append((c, total))
        scored.sort(key=lambda x: x[1], reverse=True)
        out[role] = scored[:top_k]
    return out

def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Rename selected columns to canonical role names if provided.
    Keeps all other columns as-is.
    """
    rename: Dict[str, str] = {}
    for role, col in mapping.items():
        if col and col in df.columns:
            rename[col] = role
    df2 = df.rename(columns=rename).copy()
    # Parse event_ts if present
    if "event_ts" in df2.columns:
        df2["event_ts"] = pd.to_datetime(df2["event_ts"], errors="coerce")
    # Ensure numeric for value/quantity if present
    for num_col in ("value", "quantity"):
        if num_col in df2.columns:
            df2[num_col] = pd.to_numeric(df2[num_col], errors="coerce")
    return df2

def save_to_postgres(df: pd.DataFrame, table: str, schema: str = "staging", if_exists: str = "replace") -> Tuple[str, int]:
    """
    Write DataFrame to Postgres as schema.table, creating schema if needed.
    """
    eng = get_engine()
    sch = schema
    tbl = table.strip()
    if "." in tbl:
        sch, tbl = tbl.split(".", 1)
    with eng.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{sch}"'))
        df.to_sql(tbl, con=conn, schema=sch, if_exists=if_exists, index=False)
    return f"{sch}.{tbl}", len(df)

