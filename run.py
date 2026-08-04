"""Hydra entrypoint for the pipeline stages:

    preprocess -> depthgen -> db -> attach -> ba -> gs -> eval

(preprocess/depthgen/ba/gs/eval are stubs until built.)

Local run:
    python run.py experiment=scannetpp_amb3r_bimodal sequence=09c1414f1b

Cluster, one SLURM job per sequence (submitit array):
    python run.py -m hydra/launcher=cpu experiment=scannetpp_amb3r_bimodal \
        'sequence=09c1414f1b,0d2ee665be'

A stage is done iff its marker (done_<stage>.json in its stage_dir) exists;
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

from depthba.config import AttachConfig, DBConfig, PreprocessConfig
from depthba.depth.attach_depths import run as run_attach
from depthba.frontends.colmap_runner import run_db
from depthba.preprocess import unpack_amb3r

log = logging.getLogger(__name__)


def stage_preprocess(cfg: DictConfig) -> None:
    config = PreprocessConfig.from_dict(
        OmegaConf.to_container(cfg.preprocess, resolve=True), source="cfg.preprocess"
    )
    if cfg.sequence is None:
        raise ValueError("preprocess requires sequence=")
    raw_images = Path(cfg.raw_root) / config.image_subdir.format(sequence=cfg.sequence)
    if not raw_images.exists():
        raise FileNotFoundError(f"raw image dir {raw_images} does not exist")
    out_dir = stage_dir("preprocess", cfg)

    # amb3r inference (GPU, hours) in the amb3r repo's own venv; sub-skip on
    # the npz so an unpack failure doesn't redo it.
    npz = out_dir / f"scene_{cfg.sequence}_results.npz"
    if npz.exists():
        log.info(f"[preprocess] {npz} exists, skipping amb3r inference")
    else:
        amb3r_python = Path(config.amb3r_repo) / ".venv/bin/python"
        if not amb3r_python.exists():
            raise FileNotFoundError(f"amb3r venv python not found: {amb3r_python}")
        subprocess.run(
            [str(amb3r_python), "sfm/run.py",
             "--data_path", str(raw_images),
             "--demo_name", str(cfg.sequence),
             "--results_path", str(out_dir)],
            cwd=config.amb3r_repo, check=True,
        )

    unpack_amb3r(npz, out_dir, raw_images)


def stage_depthgen(cfg: DictConfig) -> None:
    # DepthBundles for non-amb3r sources (depthpro, gt_mesh), run on the
    # canonical images produced by preprocess
    raise NotImplementedError("depthgen stage not built yet")


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


def stage_ba(cfg: DictConfig) -> None:
    # depth-aware incremental mapping: backends/custom_incremental_pipeline.py
    # with a configs/depthba group entry
    raise NotImplementedError("ba stage not built yet")


def stage_gs(cfg: DictConfig) -> None:
    # 3DGS training on the BA output (absorbs scripts/build_3dgs_data_snpp.sh
    # + run_amb3r_3dgs.sh)
    raise NotImplementedError("gs stage not built yet")


def stage_eval(cfg: DictConfig) -> None:
    # NVS evaluation (absorbs src/depthba/eval/nvs/scannetpp)
    raise NotImplementedError("eval stage not built yet")


STAGES = {
    "preprocess": stage_preprocess,
    "depthgen": stage_depthgen,
    "db": stage_db,
    "attach": stage_attach,
    "ba": stage_ba,
    "gs": stage_gs,
    "eval": stage_eval,
}  # dict order = pipeline order


def stage_dir(stage: str, cfg: DictConfig) -> Path:
    """Where a stage's outputs and done marker live. preprocess writes into
    the canonical dataset tree; everything else goes under output_dir."""
    if stage == "preprocess":
        return Path(cfg.data_root) / cfg.preprocess.out_subdir.format(sequence=cfg.sequence)
    return Path(cfg.output_dir)


def marker_path(stage: str, cfg: DictConfig) -> Path:
    return stage_dir(stage, cfg) / f"done_{stage}.json"


def mark_done(stage: str, cfg: DictConfig) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    marker_path(stage, cfg).write_text(json.dumps({
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

    order = list(STAGES)
    for stage in cfg.stages:
        marker = marker_path(stage, cfg)
        if marker.exists() and not cfg.force:
            log.info(f"[{stage}] {marker} exists, skipping")
            continue
        log.info(f"[{stage}] running")
        # Invalidate downstream markers up front: the stage may destroy its
        # old outputs immediately (run_db unlinks database.db), so a crash
        # mid-stage must not leave downstream stages looking done.
        for downstream in order[order.index(stage) + 1:]:
            marker_path(downstream, cfg).unlink(missing_ok=True)
        stage_dir(stage, cfg).mkdir(parents=True, exist_ok=True)
        STAGES[stage](cfg)
        mark_done(stage, cfg)
        log.info(f"[{stage}] done")


if __name__ == "__main__":
    main()
