import argparse
from pathlib import Path

from depthba.config import DBConfig
from depthba.frontends.colmap_runner import run_db


def main_db() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sequence", default=None,
                        help="substituted into the config's image_path "
                             "'{sequence}' placeholder (dataset-level configs)")
    parser.add_argument("--vocab_tree_path", default=None,
                        help="local vocab tree .bin (required for "
                             "matching.loop_detection; compute nodes can't download it)")

    args = parser.parse_args()
    config = DBConfig.load(args.config)

    if "{sequence}" in config.image_path:
        if args.sequence is None:
            parser.error(f"config image_path {config.image_path!r} requires --sequence")
        config.image_path = config.image_path.format(sequence=args.sequence)
    elif args.sequence is not None:
        parser.error("--sequence given but config image_path has no {sequence} placeholder")

    run_db(config, args.data_root, args.output_dir, args.vocab_tree_path)
