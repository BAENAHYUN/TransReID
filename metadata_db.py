"""
metadata_db.py — SQLite-backed metadata store for ForensicPhotoSearch.

Replaces the large JSON metadata file approach so that 10 M+ records
do not have to be loaded into RAM at startup.  The public interface is
intentionally identical to a plain Python list:

    db = MetadataDB(path)
    record = db[vector_id]   # O(1) primary-key lookup
    n      = len(db)          # COUNT(*) — fast with SQLite

Migration (run once after existing JSON files are built):

    python metadata_db.py migrate \
        --json  data/embeddings/qwen_image_metadata.json \
        --db    data/embeddings/qwen_image_metadata.db

    python metadata_db.py migrate \
        --json  data/semantic_video/metadata.json \
        --db    data/semantic_video/metadata.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict


# =========================================================
# MetadataDB
# =========================================================

class MetadataDB:
    """
    Read-only view over a SQLite metadata store.

    The underlying table has exactly two columns:

        vector_id  INTEGER  PRIMARY KEY   (== FAISS vector index)
        data       TEXT     NOT NULL      (JSON-serialised record dict)

    SQLite handles the file paging; only the rows actually requested
    are read from disk, so a 10 M-row database uses virtually no RAM
    at startup.

    Parameters
    ----------
    db_path : str | Path
        Path to the .db file produced by :func:`build_from_json`.
    """

    def __init__(self, db_path: str | Path) -> None:

        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Metadata database not found:\n{self.db_path}\n\n"
                "Run the migration first:\n"
                "  python metadata_db.py migrate --json <json> --db <db>"
            )

        # check_same_thread=False is safe for read-only, single-writer use.
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row

        # Larger page cache (64 MB) — speeds up sequential scans during
        # validation and random reads during reranking.
        self._conn.execute("PRAGMA cache_size = -65536")

        # Read-ahead hint (not all SQLite versions honour it, harmless if not).
        self._conn.execute("PRAGMA mmap_size = 536870912")  # 512 MB mmap

    # ------------------------------------------------------------------
    # List-compatible interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:

        row = self._conn.execute(
            "SELECT COUNT(*) FROM metadata"
        ).fetchone()

        return int(row[0]) if row else 0

    def __getitem__(self, vector_id: int) -> Dict[str, Any]:

        row = self._conn.execute(
            "SELECT data FROM metadata WHERE vector_id = ?",
            (int(vector_id),),
        ).fetchone()

        if row is None:
            raise KeyError(
                f"vector_id {vector_id} not found in {self.db_path}"
            )

        return json.loads(row["data"])

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =========================================================
# Migration helper
# =========================================================

def build_from_json(
    json_path: str | Path,
    db_path: str | Path,
    batch_size: int = 10_000,
) -> None:
    """
    One-time migration: JSON list → SQLite.

    Parameters
    ----------
    json_path : path to the existing metadata JSON file.
        Must be a JSON array indexed by vector_id (0-based).
    db_path   : output .db file path.  Overwritten if it already exists.
    batch_size: records per INSERT transaction.  Tune up for faster SSD.
    """

    json_path = Path(json_path)
    db_path   = Path(db_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Reading  {json_path} …")
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)
    print(f"[+] {total:,} records → {db_path}")

    # Overwrite any previous .db
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))

    try:

        # Fast bulk-insert settings (safe to use during initial build).
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous  = OFF")
        conn.execute("PRAGMA cache_size   = -65536")

        conn.execute("""
            CREATE TABLE metadata (
                vector_id  INTEGER  PRIMARY KEY,
                data       TEXT     NOT NULL
            )
        """)

        inserted = 0
        batch: list[tuple[int, str]] = []

        for i, record in enumerate(records):
            batch.append((i, json.dumps(record, ensure_ascii=False)))
            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT INTO metadata VALUES (?, ?)",
                    batch,
                )
                conn.commit()
                inserted += len(batch)
                batch = []
                print(
                    f"  {inserted:>{len(str(total))},} / {total:,}",
                    end="\r",
                    flush=True,
                )

        if batch:
            conn.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                batch,
            )
            conn.commit()
            inserted += len(batch)

        # Restore WAL for concurrent read access after build.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.commit()

    finally:
        conn.close()

    size_mb = db_path.stat().st_size / 1_048_576
    print(f"\n[+] Done. {inserted:,} records · {size_mb:.1f} MB → {db_path}")


# =========================================================
# CLI
# =========================================================

def _cmd_migrate(args: argparse.Namespace) -> None:
    build_from_json(args.json, args.db, batch_size=args.batch)


def main() -> None:

    ap = argparse.ArgumentParser(
        description="ForensicPhotoSearch metadata migration tool"
    )
    sub = ap.add_subparsers(dest="command")

    migrate = sub.add_parser(
        "migrate",
        help="Convert a JSON metadata array to a SQLite database."
    )
    migrate.add_argument(
        "--json",
        required=True,
        metavar="PATH",
        help="Input JSON file (list indexed by vector_id).",
    )
    migrate.add_argument(
        "--db",
        required=True,
        metavar="PATH",
        help="Output SQLite .db file.",
    )
    migrate.add_argument(
        "--batch",
        type=int,
        default=10_000,
        metavar="N",
        help="INSERT batch size (default: 10000).",
    )

    args = ap.parse_args()

    if args.command == "migrate":
        _cmd_migrate(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
