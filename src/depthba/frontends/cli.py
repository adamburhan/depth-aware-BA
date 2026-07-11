import argparse, os
from pathlib import Path
from depthba.frontends.colmap_runner import run_db
from depthba.config import DBConfig

def main_db() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    
    args = parser.parse_args()
    config = DBConfig.load(args.config)
    run_db(config, args.data_root, args.output_dir)