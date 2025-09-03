from __future__ import annotations
import re
from typing import Dict, List
import pandas as pd

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")
CARD  = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

def scan_pii(df: pd.DataFrame, sample_rows: int = 5000) -> List[Dict]:
    out: List[Dict] = []
    check = df.head(sample_rows)
    for c in df.columns:
        s = check[c].astype(str).fillna("")
        matches = 0
        for v in s:
            if EMAIL.search(v) or PHONE.search(v) or CARD.search(v):
                matches += 1
        if matches > 0:
            out.append({"column": c, "hits_in_sample": int(matches), "rows_scanned": int(len(s))})
    return out

def mask_column(df: pd.DataFrame, column: str, keep_last: int = 4) -> pd.DataFrame:
    if column not in df.columns:
        return df
    s = df[column].astype(str)
    df[column] = s.map(lambda x: x if len(x) <= keep_last else ("*"*(len(x)-keep_last) + x[-keep_last:]))
    return df
