import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB   = os.getenv("PG_DB", "analytics")
PG_USER = os.getenv("PG_USER", "analytics")
PG_PASS = os.getenv("PG_PASSWORD", "analytics")

def get_engine():
    url = f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(url, pool_pre_ping=True)

if __name__ == "__main__":
    # quick smoke test when run directly: prints server version
    eng = get_engine()
    with eng.connect() as conn:
        ver = conn.execute(text("SELECT version();")).scalar()
        print("✅ Connected to:", ver)
