"""
database.py  -  Day 2 module
Dummy owner database + challan records (SQLite).
Creates parking.db and a challan_images/ folder next to this file.
"""

import os
import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

DB_PATH = "parking.db"
IMG_DIR = "challan_images"

# Clearly fake sample data. Add your own entries from the Owners tab.
DUMMY_OWNERS = [
    ("UP80AB1234", "Demo Owner 1", "+910000000001"),
    ("DL1CAB0001", "Demo Owner 2", "+910000000002"),
    ("HR26DK0001", "Demo Owner 3", "+910000000003"),
    ("UP65AB1001", "Demo Owner 4", "+910000000004"),
    ("UP32CD2002", "Demo Owner 5", "+910000000005"),
    ("UP14EF3003", "Demo Owner 6", "+910000000006"),
    ("DL3CAF4004", "Demo Owner 7", "+910000000007"),
    ("DL7CQ5005", "Demo Owner 8", "+910000000008"),
    ("DL10CB6006", "Demo Owner 9", "+910000000009"),
    ("HR55ET7097", "Demo Owner 10", "+910000000010"),
    ("HR51BK8008", "Demo Owner 11", "+910000000011"),
    ("RJ14CD9009", "Demo Owner 12", "+910000000012"),
    ("RJ27GH1010", "Demo Owner 13", "+910000000013"),
    ("MH12AB2020", "Demo Owner 14", "+910000000014"),
    ("MH02DE3030", "Demo Owner 15", "+910000000015"),
    ("KA01MN4040", "Demo Owner 16", "+910000000016"),
    ("TN09PQ5050", "Demo Owner 17", "+910000000017"),
    ("GJ01RS6060", "Demo Owner 18", "+910000000018"),
    ("PB10TU7070", "Demo Owner 19", "+910000000019"),
    ("UK07VW8080", "Demo Owner 20", "+910000000020"),
]


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def normalize(plate):
    """Upper-case and keep only letters/digits."""
    return re.sub(r"[^A-Z0-9]", "", (plate or "").upper())


def init_db():
    os.makedirs(IMG_DIR, exist_ok=True)
    with _db() as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS owners "
            "(plate TEXT PRIMARY KEY, name TEXT, phone TEXT, email TEXT DEFAULT '')"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS challans ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, plate TEXT, owner_name TEXT, "
            "phone TEXT, zone TEXT, fine INTEGER, timestamp TEXT, "
            "image_path TEXT, status TEXT, email TEXT DEFAULT '')"
        )
        # Upgrade an older parking.db that has no email column yet
        for table in ("owners", "challans"):
            cols = [r[1] for r in c.execute(f"PRAGMA table_info({table})")]
            if "email" not in cols:
                c.execute(f"ALTER TABLE {table} ADD COLUMN email TEXT DEFAULT ''")
        # Adds any missing sample owners; never overwrites existing ones
        c.executemany(
            "INSERT OR IGNORE INTO owners (plate, name, phone) VALUES (?, ?, ?)",
            DUMMY_OWNERS,
        )


def get_owner(plate):
    """Return {'plate','name','phone'} or None if not found."""
    with _db() as c:
        row = c.execute(
            "SELECT * FROM owners WHERE plate = ?", (normalize(plate),)
        ).fetchone()
    return dict(row) if row else None


def add_owner(plate, name, phone="", email=""):
    with _db() as c:
        c.execute(
            "INSERT OR REPLACE INTO owners (plate, name, phone, email) VALUES (?, ?, ?, ?)",
            (normalize(plate), name.strip(), (phone or "").strip(), (email or "").strip()),
        )


def list_owners():
    with _db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM owners ORDER BY plate")]


def create_challan(plate, owner, zone, fine, image_path, status="Pending"):
    """Save a challan (and a copy of the image). Returns the challan as a dict."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plate = normalize(plate)

    # keep a copy of the evidence image
    ext = os.path.splitext(image_path)[1] or ".jpg"
    saved_img = os.path.join(
        IMG_DIR, f"{plate}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    )
    shutil.copy(image_path, saved_img)

    with _db() as c:
        cur = c.execute(
            "INSERT INTO challans "
            "(plate, owner_name, phone, email, zone, fine, timestamp, image_path, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (plate, owner["name"], owner.get("phone", ""), owner.get("email", ""),
             zone, fine, ts, saved_img, status),
        )
        challan_id = cur.lastrowid

    return {
        "id": challan_id, "plate": plate, "owner_name": owner["name"],
        "phone": owner.get("phone", ""), "email": owner.get("email", ""),
        "zone": zone, "fine": fine,
        "timestamp": ts, "image_path": saved_img, "status": status,
    }


def update_challan_status(challan_id, status):
    with _db() as c:
        c.execute("UPDATE challans SET status = ? WHERE id = ?", (status, challan_id))


def find_recent_challan(plate, zone, minutes=60):
    """Latest challan for this plate + zone in the last N minutes, or None."""
    cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
    with _db() as c:
        row = c.execute(
            "SELECT * FROM challans WHERE plate = ? AND zone = ? AND timestamp >= ? "
            "ORDER BY id DESC LIMIT 1",
            (normalize(plate), zone, cutoff),
        ).fetchone()
    return dict(row) if row else None


def get_challans():
    with _db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM challans ORDER BY id DESC")]
