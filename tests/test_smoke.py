import pandas as pd
from src.eda import infer_columns, profile_basic

def test_infer_columns_basic():
    df = pd.DataFrame({
        "id":[1,2,3],
        "event_ts":["2025-01-01","2025-01-02","2025-01-03"],
        "value":[10.5, 9.0, 11.2],
        "category":["A","B","A"]
    })
    num, cat, dt = infer_columns(df)
    assert "value" in num
    assert "category" in cat
    assert "event_ts" in dt

def test_profile_basic_runs():
    df = pd.DataFrame({"a":[1,2,3]})
    md = profile_basic(df)
    assert "rows" in md.lower()
