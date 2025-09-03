from __future__ import annotations
from typing import Dict, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from src.eda import load_df_from_source
from src.schema_map import suggest_mapping, apply_mapping

TARGET_CANDIDATES = [
    "value","amount","revenue","sales","price","total","gmv","net_sales","amt","subtotal","grand_total"
]

def _is_id_like(name: str, s: pd.Series) -> bool:
    n = (name or "").lower()
    if any(k in n for k in ["id","uuid","guid","ssn","iban","account","pan"]):
        return True
    try:
        uniq_ratio = s.nunique(dropna=True) / max(1, len(s))
        if uniq_ratio > 0.9:
            return True
    except Exception:
        pass
    try:
        if pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing:
            return True
    except Exception:
        pass
    return False

def _auto_map(df: pd.DataFrame):
    sugg = suggest_mapping(df)
    def first(role): return (sugg.get(role) or [[None,0]])[0][0]
    mapping = {
        "entity_id": first("entity_id"),
        "event_ts": first("event_ts"),
        "value": first("value"),
        "quantity": first("quantity"),
        "category": first("category"),
    }
    mapped = apply_mapping(df, mapping)
    return mapped, mapping

def _guess_target(df: pd.DataFrame, mapped_value: Optional[str]) -> Optional[str]:
    if mapped_value and mapped_value in df.columns:
        s = pd.to_numeric(df[mapped_value], errors="coerce")
        if s.notna().any():
            return mapped_value
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in TARGET_CANDIDATES:
        if cand in cols_lower:
            s = pd.to_numeric(df[cols_lower[cand]], errors="coerce")
            if s.notna().any():
                return cols_lower[cand]
    numeric = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any() and not _is_id_like(c, s):
            numeric.append((c, float(np.nanvar(s))))
    if numeric:
        numeric.sort(key=lambda x: x[1], reverse=True)
        return numeric[0][0]
    return None

def _split_features(X: pd.DataFrame) -> (List[str], List[str]):
    num, cat = [], []
    for c in X.columns:
        s_num = pd.to_numeric(X[c], errors="coerce")
        if s_num.notna().mean() > 0.5:
            num.append(c)
        else:
            cat.append(c)
    return num, cat

def train_value_regressor(file_path: Optional[str] = None, table_name: Optional[str] = None) -> Dict[str, str]:
    if not file_path and not table_name:
        raise ValueError("Provide file_path or table_name")
    if table_name:
        df, _ = load_df_from_source(table_name=table_name)
        base = table_name.replace(".", "_")
    else:
        df, _ = load_df_from_source(file_path=file_path)
        base = Path(file_path).stem

    df, mapping = _auto_map(df)
    y_col = _guess_target(df, mapping.get("value"))
    if not y_col:
        raise ValueError("Could not infer a numeric target (e.g., value/amount/revenue).")

    y = pd.to_numeric(df[y_col], errors="coerce")
    X = df.drop(columns=[y_col])

    num, cat = _split_features(X)

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), cat),
        ],
        remainder="drop"
    )
    model = GradientBoostingRegressor(random_state=42)
    pipe = Pipeline([("prep", pre), ("model", model)])

    ok = y.notna()
    if ok.sum() < 10:
        raise ValueError(f"Not enough non-null targets in '{y_col}' to train (found {int(ok.sum())}).")

    X_train, X_test, y_train, y_test = train_test_split(X[ok], y[ok], test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    r2  = float(r2_score(y_test, pred))
    mape = float(np.mean(np.abs((y_test - pred) / np.maximum(1e-8, np.abs(y_test)))) * 100.0)

    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    model_path = f"outputs/models/{base}_gbr.joblib"
    joblib.dump(pipe, model_path)

    pred_path = f"outputs/models/{base}_predictions.csv"
    out = X_test.copy()
    out["actual_value"] = y_test.values
    out["predicted_value"] = pred
    out.to_csv(pred_path, index=False)

    return {
        "metrics_md": f"**Target:** {y_col}  \n**MAE:** {mae:.4f}  \n**MAPE:** {mape:.2f}%  \n**R²:** {r2:.4f}",
        "model_path": model_path,
        "pred_path": pred_path,
    }
