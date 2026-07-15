import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import enlighten

import pycolmap
from pycolmap import logging


def incremental_mapping_with_pbar(
    database_path: Path, image_path: Path, sfm_path: Path
) -> dict[int, pycolmap.Reconstruction]:
    with pycolmap.Database.open(database_path) as database:
        num_images = database.num_images()
    with enlighten.Manager() as manager:
        with manager.counter(
            total=num_images, desc="Images registered:"
        ) as pbar:
            pbar.update(0, force=True)
            reconstructions = pycolmap.incremental_mapping(
                database_path,
                image_path,
                sfm_path,
                initial_image_pair_callback=lambda: pbar.update(2),
                next_image_callback=lambda: pbar.update(1),
            )
    return reconstructions


def run() -> None:
    output_path = Path("example/")
    image_path = Path("/Users/adam/Documents/MILA/projects/depth-uncertainty-slam/data/kicker/images/dslr_images_undistorted/")
    database_path = output_path / "database.db"
    sfm_path = output_path / "sfm"

    output_path.mkdir(exist_ok=True)
    # The log filename is postfixed with the execution timestamp.
    logging.set_log_destination(logging.INFO, output_path / "INFO.log.")

    # data_url = "https://cvg-data.inf.ethz.ch/local-feature-evaluation-schoenberger2017/Strecha-Fountain.zip"
    # if not image_path.exists():
    #     logging.info("Downloading the data.")
    #     zip_path = output_path / "data.zip"
    #     urllib.request.urlretrieve(data_url, zip_path)
    #     with zipfile.ZipFile(zip_path, "r") as fid:
    #         fid.extractall(output_path)
    #     logging.info(f"Data extracted to {output_path}.")
    
    # data_url = "https://cvg-data.inf.ethz.ch/local-feature-evaluation-schoenberger2017/Strecha-Fountain.zip"
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image path {image_path} does not exist. Please download the dataset and extract it to this path."
        )


    if database_path.exists():
        database_path.unlink()
    pycolmap.set_random_seed(0)
    pycolmap.extract_features(database_path, image_path)
    pycolmap.match_exhaustive(database_path)

    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir(exist_ok=True)

    # recs = incremental_mapping_with_pbar(database_path, image_path, sfm_path)
    # alternatively, use:
    from depthba.backends import custom_incremental_pipeline
    recs = custom_incremental_pipeline.main(
        database_path, image_path, sfm_path
    )
    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")


def run_db(config, data_root, output_dir):
    output_path = Path(output_dir)
    image_path = Path(data_root) / config.image_path
    database_path = output_path / "database.db"

    output_path.mkdir(exist_ok=True, parents=True)
    # The log filename is postfixed with the execution timestamp.
    logging.set_log_destination(logging.INFO, output_path / "INFO.log.")

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image path {image_path} does not exist. Please download the dataset and extract it to this path."
        )

    if database_path.exists():
        database_path.unlink()

    pycolmap.set_random_seed(config.seed)

    reader_options = pycolmap.ImageReaderOptions()
    if config.camera.model is not None:
        reader_options.camera_model = config.camera.model
    if config.camera.params is not None:
        # Initial values only; whether BA refines them is a mapper-stage
        # choice (ba_refine_* on IncrementalPipelineOptions).
        reader_options.camera_params = ",".join(map(str, config.camera.params))

    # stride subsamples the (sorted) image list; stride=1 takes everything.
    image_names = []
    if config.stride > 1:
        image_names = sorted(
            p.name for p in image_path.iterdir() if p.is_file()
        )[:: config.stride]
        logging.info(f"stride={config.stride}: {len(image_names)} images selected")

    # COLMAP sizes thread pools to the NODE's cores, not the SLURM cgroup —
    # ~100 SIFT threads in an 8-CPU/32GB allocation is an OOM kill.
    num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
    if num_threads > 0:
        logging.info(f"Limiting COLMAP to {num_threads} threads (SLURM allocation)")

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = num_threads
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.num_threads = num_threads

    camera_mode = (
        pycolmap.CameraMode.SINGLE if config.camera.single_camera
        else pycolmap.CameraMode.AUTO
    )
    pycolmap.extract_features(
        database_path,
        image_path,
        image_names=image_names,
        camera_mode=camera_mode,
        reader_options=reader_options,
        extraction_options=extraction_options,
    )

    if config.matching.method == "exhaustive":
        pycolmap.match_exhaustive(database_path, matching_options=matching_options)
    elif config.matching.method == "sequential":
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = config.matching.overlap
        pairing.loop_detection = config.matching.loop_detection
        pycolmap.match_sequential(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing,
        )
    else:
        raise ValueError(f"unknown matching method: {config.matching.method!r}")


if __name__ == "__main__":
    run()