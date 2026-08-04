"""Hydra entrypoint for the pipeline stages:

    preprocess -> depthgen -> db -> attach -> ba -> gs -> eval

(depthgen/gs/eval are stubs until built.)

Local run:
    python run.py experiment=scannetpp_amb3r_bimodal sequence=09c1414f1b

Cluster, one SLURM job per sequence (submitit array):
    python run.py -m hydra/launcher=cpu experiment=scannetpp_amb3r_bimodal \
        'sequence=09c1414f1b,0d2ee665be'

ba sensitivity sweep — one job and one ba/<variant>/ output dir per cell:
    python run.py -m hydra/launcher=cpu 'stages=[ba]' depthba=amb3r_gmm_local \
        'depthba.sigma=0.02,0.05,0.1' 'depthba.huber_scale=null,2.0' \
        experiment=scannetpp_amb3r_bimodal 'sequence=09c1414f1b,0d2ee665be'

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
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from depthba.config import (
    AttachConfig,
    DBConfig,
    DepthBAConfig,
    GSConfig,
    PreprocessConfig,
)
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


def ba_variant(cfg: DictConfig) -> str:
    """Directory name for the ba condition: the depthba option, plus any
    depthba.* CLI overrides (sweep values) so sweep cells don't collide:
    depthba=amb3r_gmm_local depthba.sigma=0.05 -> amb3r_gmm_local_sigma0.05.
    exp_id separates repeats of one identical condition."""
    hc = HydraConfig.get()
    name = hc.runtime.choices["depthba"]
    mods = sorted(
        o.removeprefix("depthba.").replace("=", "")
        for o in hc.overrides.task if o.startswith("depthba.")
    )
    if cfg.exp_id is not None:
        mods.append(str(cfg.exp_id))
    return "_".join([name, *mods])


def stage_ba(cfg: DictConfig) -> None:
    # pyceres import chain is Linux-only (macOS glog collision), so keep the
    # pipeline import out of module scope.
    from depthba.backends.custom_incremental_pipeline import main as run_mapping

    config = DepthBAConfig.from_dict(
        OmegaConf.to_container(cfg.depthba, resolve=True), source="cfg.depthba"
    )
    run_mapping(
        database_path=Path(cfg.output_dir) / "database.db",
        image_path=stage_dir("preprocess", cfg) / "images",
        output_path=stage_dir("ba", cfg),
        # sensor=null baseline goes through the validated no-depth-ctx path
        depthba_config=None if config.sensor is None else config,
    )


def _symlink(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def stage_gs(cfg: DictConfig) -> None:
    config = GSConfig.from_dict(
        OmegaConf.to_container(cfg.gs, resolve=True), source="cfg.gs"
    )
    ba_dir = stage_dir("ba", cfg)
    # A fragmented reconstruction trains only sub-model 0, which silently makes
    # this arm incomparable to the others — fail instead.
    submodels = [p for p in ba_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if len(submodels) != 1:
        raise ValueError(
            f"{ba_dir}: {len(submodels)} sub-models, expected 1 (arm not comparable)"
        )

    out_dir = stage_dir("gs", cfg)
    source = out_dir / "source"
    (source / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    _symlink(stage_dir("preprocess", cfg) / "images", source / "images")
    for name in ("cameras", "images", "points3D"):
        _symlink(submodels[0] / f"{name}.bin", source / "sparse" / "0" / f"{name}.bin")

    iterations = [str(i) for i in config.iterations]
    run = lambda args: subprocess.run(  # noqa: E731
        [config.python_bin, *args], cwd=config.repo, check=True
    )
    run(["train.py", "-s", str(source), "-m", str(out_dir), "--eval",
         "-r", str(config.resolution), "--quiet", "--disable_viewer",
         "--save_iterations", *iterations, "--test_iterations", *iterations])
    # render.py --iteration takes ONE value (unlike train.py's --save_iterations),
    # so loop; metrics.py then scores every test/ours_<iter> dir it finds.
    for it in iterations:
        run(["render.py", "-m", str(out_dir), "--iteration", it, "--skip_train"])
    run(["metrics.py", "-m", str(out_dir)])


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
    the canonical dataset tree; ba gets a per-variant subdir (sweep cells must
    not collide) and gs a per-repeat one under it; everything else goes under
    output_dir."""
    if stage == "preprocess":
        return Path(cfg.data_root) / cfg.preprocess.out_subdir.format(sequence=cfg.sequence)
    if stage == "ba":
        return Path(cfg.output_dir) / "ba" / ba_variant(cfg)
    if stage == "gs":
        return stage_dir("ba", cfg) / f"gs_repeat{cfg.gs_repeat}"
    return Path(cfg.output_dir)


def invalidate_downstream(stage: str, cfg: DictConfig) -> None:
    """Delete downstream done markers within the running stage's blast
    radius: db/attach (and preprocess/depthgen) feed every ba variant, so
    they sweep all of output_dir; ba only stales its own variant subtree."""
    order = list(STAGES)
    scope = stage_dir(stage, cfg) if stage in ("ba", "gs", "eval") else Path(cfg.output_dir)
    if not scope.exists():
        return
    for downstream in order[order.index(stage) + 1:]:
        for marker in scope.rglob(f"done_{downstream}.json"):
            marker.unlink()


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

    for stage in cfg.stages:
        marker = marker_path(stage, cfg)
        if marker.exists() and not cfg.force:
            log.info(f"[{stage}] {marker} exists, skipping")
            continue
        log.info(f"[{stage}] running")
        # Invalidate downstream markers up front: the stage may destroy its
        # old outputs immediately (run_db unlinks database.db), so a crash
        # mid-stage must not leave downstream stages looking done.
        invalidate_downstream(stage, cfg)
        stage_dir(stage, cfg).mkdir(parents=True, exist_ok=True)
        STAGES[stage](cfg)
        mark_done(stage, cfg)
        log.info(f"[{stage}] done")


if __name__ == "__main__":
    main()
