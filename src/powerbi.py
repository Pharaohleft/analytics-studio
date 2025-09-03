from __future__ import annotations
import json, os
from pathlib import Path

def generate_pbids(table: str, out_dir: str = "outputs/powerbi") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    host = os.getenv("PG_HOST", "localhost")
    port = int(os.getenv("PG_PORT", "5432"))
    db   = os.getenv("PG_DB", "analytics")
    user = os.getenv("PG_USER", "analytics")
    name = table.replace(".", "_")
    content = {
      "version": "0.1",
      "connections": [
        {
          "details": {
            "protocol": "postgresql",
            "database": db,
            "server": f"{host}",
            "port": port
          },
          "name": f"PostgreSQL - {db}",
          "mode": "DirectQuery",
          "nativeQuery": f"select * from {table}",
          "authentication": {
            "type": "UsernamePassword",
            "username": user
          }
        }
      ]
    }
    path = f"{out_dir}/{name}.pbids"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
    return path
