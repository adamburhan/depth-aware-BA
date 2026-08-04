"""One-off: write done_*.json markers for sequences processed before the
marker system existed. Verifies each stage's artifacts before marking;
markers are labeled backfilled (they carry no resolved config).

    python scripts/backfill_markers.py \
        --data_root $SCRATCH/datasets/scannetpp/data/amb3r \
        --output_root $SCRATCH/experiments/depth-aware-ba/scannetpp_amb3r \
        09c1414f1b 0d2ee665be ...
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def mark(path: Path, stage: str) -> None:
    (path / f"done_{stage}.json").write_text(json.dumps({
        "stage": stage,
        "backfilled": True,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    print(f"  {stage}: marked")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--output_root", type=Path, required=True)
    ap.add_argument("sequences", nargs="+")
    args = ap.parse_args()

    for seq in args.sequences:
        print(f"{seq}:")
        canon = args.data_root / seq / "dslr"
        out = args.output_root / seq / "dslr"

        bundles = list((canon / "depth_bundles").glob("*.npz")) if canon.exists() else []
        if (canon / "intrinsics.json").exists() and bundles:
            mark(canon, "preprocess")
        else:
            print(f"  preprocess: MISSING ({canon})")

        db = out / "database.db"
        if db.exists():
            mark(out, "db")
            conn = sqlite3.connect(db)
            try:
                sensors = [r[0] for r in conn.execute(
                    "SELECT sensor FROM depthba_depth_meta ORDER BY sensor")]
            except sqlite3.OperationalError:
                sensors = []
            finally:
                conn.close()
            if sensors:
                print(f"  sensors in db: {sensors}")
                mark(out, "attach")
            else:
                print("  attach: no sensor rows, not marked")
        else:
            print(f"  db: MISSING ({db})")


if __name__ == "__main__":
    main()
