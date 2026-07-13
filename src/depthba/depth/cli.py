import argparse
from pathlib import Path

from depthba.config import AttachConfig
from depthba.depth.attach_depths import run


def main_attach() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--dump_dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="re-ingest: delete this sensor's existing rows first")

    args = parser.parse_args()
    config = AttachConfig.load(args.config)
    run(config, args.db, args.dump_dir, force=args.force)
