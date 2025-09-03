import pandas as pd
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)
n = 5000

# dates across ~20 months in clean ISO format (no warnings)
start = pd.Timestamp("2024-01-01")
end   = pd.Timestamp("2025-08-01")
days = (end - start).days
event_ts = [start + pd.Timedelta(int(rng.integers(0, days)), unit="D") for _ in range(n)]
event_ts = pd.to_datetime(event_ts).strftime("%Y-%m-%d")  # ISO date strings

order_id = np.arange(1, n+1)
customer_id = [f"CUST{rng.integers(10000,99999)}" for _ in range(n)]
category = rng.choice(["A","B","C","D"], size=n, p=[0.35,0.30,0.25,0.10])
payment_type = rng.choice(["credit_card","pix","boleto","voucher","paypal"], size=n, p=[0.5,0.2,0.15,0.1,0.05])
city = rng.choice(["Sao Paulo","Rio de Janeiro","Curitiba","Porto Alegre","Belo Horizonte","Campinas"], size=n)
state = rng.choice(["SP","RJ","PR","RS","MG"], size=n)
quantity = rng.integers(1,6,size=n)

# heavy-tailed "amount" with a few outliers
amount = rng.lognormal(mean=3.0, sigma=0.7, size=n).round(2)
out_idx = rng.choice(n, size=int(0.01*n), replace=False)
amount[out_idx] = amount[out_idx] * rng.integers(5, 12, size=len(out_idx))
amount = amount.round(2)

# review_score with some missing
review_score = rng.integers(1,6,size=n).astype("float")
miss_idx = rng.choice(n, size=int(0.10*n), replace=False)
review_score[miss_idx] = np.nan

# columns with deliberate missingness to showcase "Missingness" chart
optional_code = np.where(rng.random(n) < 0.85, None, rng.integers(100,999,size=n).astype(str))
notes = np.where(rng.random(n) < 0.30, None, "ok")

df = pd.DataFrame({
    "order_id": order_id,
    "customer_id": customer_id,
    "event_ts": event_ts,          # timestamp column (ISO)
    "category": category,          # categorical
    "payment_type": payment_type,  # categorical
    "city": city,                  # categorical
    "state": state,                # categorical
    "quantity": quantity,          # numeric
    "amount": amount,              # numeric
    "review_score": review_score,  # numeric with missing
    "optional_code": optional_code,# mostly missing
    "notes": notes                 # some missing
})

Path("data_raw").mkdir(parents=True, exist_ok=True)
out_path = Path("data_raw") / "sample_business.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} with {len(df)} rows and {df.shape[1]} columns")
