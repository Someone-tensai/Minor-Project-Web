"""
Persists past inference runs so the UI can show a history of results
across models -- this is the point of a *comparative* study.

Storage is intentionally simple for a showcase project:
  - metadata -> SQLite (history.db, one file, no server to run)
  - images   -> local disk (backend/history_images/)

If this ever gets deployed somewhere with an ephemeral filesystem
(serverless, or a free-tier host that wipes disk on restart), swap
`_save_image_bytes()` below for an upload to Cloudinary/S/3/etc and
have it return a URL instead of a local path -- nothing else in this
file or in main.py needs to change.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).parent / "history.db"
IMAGES_DIR = Path(__file__).parent / "history_images"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    IMAGES_DIR.mkdir(exist_ok=True)
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                model_name TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                predicted_index INTEGER NOT NULL,
                probabilities TEXT NOT NULL,
                original_image_path TEXT NOT NULL,
                overlay_image_path TEXT NOT NULL
            )
        """)


def _save_image_bytes(png_bytes: bytes, filename: str) -> str:
    """Local-disk implementation. Returns a path relative to IMAGES_DIR."""
    (IMAGES_DIR / filename).write_bytes(png_bytes)
    return filename


def save_run(model_name: str, predicted_class: str, predicted_index: int,
             probabilities: dict, original_png: bytes, overlay_png: bytes) -> dict:
    run_id = uuid.uuid4().hex[:12]
    original_path = _save_image_bytes(original_png, f"{run_id}_original.png")
    overlay_path = _save_image_bytes(overlay_png, f"{run_id}_overlay.png")
    created_at = time.time()

    with _connect() as conn:
        conn.execute(
            "INSERT INTO runs (id, created_at, model_name, predicted_class, "
            "predicted_index, probabilities, original_image_path, overlay_image_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, created_at, model_name, predicted_class, predicted_index,
             json.dumps(probabilities), original_path, overlay_path),
        )

    return {"id": run_id, "created_at": created_at}


def list_runs(limit: int = 200) -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at, model_name, predicted_class, predicted_index, "
            "original_image_path, overlay_image_path FROM runs "
            "ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["probabilities"] = json.loads(record["probabilities"])
    return record


def delete_run(run_id: str) -> bool:
    record = get_run(run_id)
    if record is None:
        return False
    for key in ("original_image_path", "overlay_image_path"):
        path = IMAGES_DIR / record[key]
        if path.exists():
            path.unlink()
    with _connect() as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    return True
