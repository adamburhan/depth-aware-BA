"""Hydra entrypoint for the pipeline stages (db -> attach).

Local run:
    python run.py experiment=scannetpp_amb3r_bimodal sequence=09c1414f1b

Cluster, one SLURM job per sequence (submitit array):
    python run.py -m hydra/launcher=cpu experiment=scannetpp_amb3r_bimodal \
        'sequence=09c1414f1b,0d2ee665be'

A stage is done iff its marker (done_<stage>.json in output_dir) exists;
the marker is written only after the stage succeeds, so a killed job is
re-run, not skipped. force=true re-runs the selected stages; running a
stage deletes the markers of all downstream stages (its outputs feed
theirs). Invalidate manually by deleting a marker.
"""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from depthba.config import AttachConfig, DBConfig
from depthba.depth.attach_depths import run as run_attach
from depthba.frontends.colmap_runner import run_db

log = logging.getLogger(__name__)


def stage_db(cfg: DictConfig) -> None:
    config = DBConfig.from_dict(
        OmegaConf.to_container(cfg.db, resolve=True), source="cfg.db"
    )
    if "{sequence}" in config.image_path:
        if cfg.sequence is None:
            raise ValueError(f"config image_path {config.image_path!r} requires sequence=")
        config.image_path = config.image_path.format(sequence=cfg.sequence)
    elif cfg.sequence is not None:
        raise ValueError("sequence= given but config image_path has no {sequence} placeholder")
    config.camera.resolve(Path(cfg.data_root), cfg.sequence)
    run_db(config, Path(cfg.data_root), Path(cfg.output_dir), cfg.vocab_tree_path)


def stage_attach(cfg: DictConfig) -> None:
    config = AttachConfig.from_dict(
        OmegaConf.to_container(cfg.depth_extract, resolve=True), source="cfg.depth_extract"
    )
    db_path = Path(cfg.output_dir) / "database.db"
    run_attach(config, db_path, Path(cfg.dump_dir), force=cfg.force)


STAGES = {"db": stage_db, "attach": stage_attach}  # dict order = pipeline order


def marker_path(output_dir: Path, stage: str) -> Path:
    return output_dir / f"done_{stage}.json"


def mark_done(output_dir: Path, stage: str, cfg: DictConfig) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    marker_path(output_dir, stage).write_text(json.dumps({
        "stage": stage,
        "commit": commit,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, indent=2))


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    unknown = set(cfg.stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)} (available: {list(STAGES)})")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order = list(STAGES)
    for stage in cfg.stages:
        marker = marker_path(output_dir, stage)
        if marker.exists() and not cfg.force:
            log.info(f"[{stage}] {marker} exists, skipping")
            continue
        log.info(f"[{stage}] running")
        # Invalidate downstream markers up front: the stage may destroy its
        # old outputs immediately (run_db unlinks database.db), so a crash
        # mid-stage must not leave downstream stages looking done.
        for downstream in order[order.index(stage) + 1:]:
            marker_path(output_dir, downstream).unlink(missing_ok=True)
        STAGES[stage](cfg)
        mark_done(output_dir, stage, cfg)
        log.info(f"[{stage}] done")


if __name__ == "__main__":
    main()
