"""
Python reimplementation of the bundle adjustment for the incremental mapper of
C++ with equivalent logic, built as an explicit pyceres problem so that depth
factors can be added on top of the standard reprojection objective.

build_problem mirrors COLMAP's DefaultBundleAdjuster constructor
(src/colmap/estimators/bundle_adjustment.cc) phase by phase. All parameter
blocks are local numpy copies (never views into pycolmap memory — pointer
lifetime is unknowable from Python); results are written back to the
reconstruction after the solve.

Monocular assumption: every image is its rig's reference sensor
(sensor_from_rig identity); build_problem raises on multi-sensor rigs.
"""

import collections
import copy

import numpy as np

import pyceres

import pycolmap
from pycolmap import logging

from depthba.backends.depth_context import DepthContext, median_depth_ratio

# ---------------------------------------------------------------------------
# Problem construction (mirrors DefaultBundleAdjuster, phase by phase)
# ---------------------------------------------------------------------------


def _make_loss(ceres_options):
    kind = ceres_options.loss_function_type
    scale = ceres_options.loss_function_scale
    if kind == pycolmap.LossFunctionType.TRIVIAL:
        return None
    if kind == pycolmap.LossFunctionType.SOFT_L1:
        return pyceres.SoftLOneLoss(scale)
    if kind == pycolmap.LossFunctionType.CAUCHY:
        return pyceres.CauchyLoss(scale)
    if kind == pycolmap.LossFunctionType.HUBER:
        return pyceres.HuberLoss(scale)
    raise ValueError(f"unsupported loss function type: {kind}")


def _pose7(frame) -> np.ndarray:
    """pose7 = [quat(xyzw) | t] of frame.rig_from_world (Rigid3d layout;
    convention pinned by gate A' — factors and ReprojErrorCost both take it)."""
    rig_from_world = frame.rig_from_world
    return np.r_[
        np.asarray(rig_from_world.rotation.quat, dtype=np.float64),
        np.asarray(rig_from_world.translation, dtype=np.float64),
    ]


def _rigid_from_pose7(pose7: np.ndarray) -> pycolmap.Rigid3d:
    return pycolmap.Rigid3d(pycolmap.Rotation3d(pose7[:4]), pose7[4:])


def _add_image_to_problem(problem, blocks, loss, ba_config, reconstruction, image_id):
    """Phase 1: one reprojection residual per triangulated observation of a
    config image; fills the per-point config-observation counts."""
    image = reconstruction.images[image_id]
    frame_id = image.frame_id
    camera_id = image.camera_id
    if frame_id not in blocks["poses"]:
        blocks["poses"][frame_id] = _pose7(image.frame)
        if ba_config.has_constant_rig_from_world_pose(frame_id):
            blocks["const_poses"].add(frame_id)
    if camera_id not in blocks["cams"]:
        blocks["cams"][camera_id] = (
            reconstruction.cameras[camera_id].params.astype(np.float64).copy()
        )
    blocks["config_cams"].add(camera_id)

    camera = reconstruction.cameras[camera_id]
    for p2d in image.points2D:
        if not p2d.has_point3D():
            continue
        point3D_id = p2d.point3D_id
        if point3D_id not in blocks["points"]:
            blocks["points"][point3D_id] = (
                reconstruction.points3D[point3D_id].xyz.astype(np.float64).copy()
            )
        cost = pycolmap.cost_functions.ReprojErrorCost(camera.model, p2d.xy)
        problem.add_residual_block(
            cost, loss,
            [blocks["points"][point3D_id], blocks["poses"][frame_id],
             blocks["cams"][camera_id]],
        )
        blocks["point_num_obs"][point3D_id] += 1


def _add_point_to_problem(problem, blocks, loss, ba_config, reconstruction, point3D_id):
    """Phase 2: for explicitly-listed points, add their observations from
    images OUTSIDE the config, with those poses (and cameras) constant —
    this is what anchors a variable point to the rest of the map."""
    point3D = reconstruction.points3D[point3D_id]
    if point3D_id not in blocks["points"]:
        blocks["points"][point3D_id] = point3D.xyz.astype(np.float64).copy()
    for element in point3D.track.elements:
        if element.image_id in ba_config.images:
            continue  # residual already added in phase 1
        image = reconstruction.images[element.image_id]
        frame_id = image.frame_id
        camera_id = image.camera_id
        if frame_id not in blocks["poses"]:
            blocks["poses"][frame_id] = _pose7(image.frame)
        blocks["const_poses"].add(frame_id)  # out-of-config poses only anchor
        if camera_id not in blocks["cams"]:
            blocks["cams"][camera_id] = (
                reconstruction.cameras[camera_id].params.astype(np.float64).copy()
            )
        if camera_id not in blocks["config_cams"]:
            blocks["const_cams"].add(camera_id)
        camera = reconstruction.cameras[camera_id]
        p2d = image.points2D[element.point2D_idx]
        cost = pycolmap.cost_functions.ReprojErrorCost(camera.model, p2d.xy)
        problem.add_residual_block(
            cost, loss,
            [blocks["points"][point3D_id], blocks["poses"][frame_id],
             blocks["cams"][camera_id]],
        )


def _parameterize_cameras(problem, blocks, ba_options, ba_config, reconstruction):
    refine_any = (
        ba_options.refine_focal_length
        or ba_options.refine_principal_point
        or ba_options.refine_extra_params
    )
    for camera_id, params in blocks["cams"].items():
        if (
            not refine_any
            or camera_id in blocks["const_cams"]
            or ba_config.has_constant_cam_intrinsics(camera_id)
        ):
            problem.set_parameter_block_constant(params)
            blocks["const_cams"].add(camera_id)
            continue
        camera = reconstruction.cameras[camera_id]
        const_idxs = []
        if not ba_options.refine_focal_length:
            const_idxs += list(camera.focal_length_idxs())
        if not ba_options.refine_principal_point:
            const_idxs += list(camera.principal_point_idxs())
        if not ba_options.refine_extra_params:
            const_idxs += list(camera.extra_params_idxs())
        if const_idxs:
            problem.set_manifold(
                params, pyceres.SubsetManifold(len(params), sorted(const_idxs))
            )


def _parameterize_frames(problem, blocks, ba_options):
    for frame_id, pose7 in blocks["poses"].items():
        if not ba_options.refine_rig_from_world or frame_id in blocks["const_poses"]:
            problem.set_parameter_block_constant(pose7)
            blocks["const_poses"].add(frame_id)
        else:
            problem.set_manifold(
                pose7,
                pyceres.ProductManifold(
                    pyceres.EigenQuaternionManifold(), pyceres.EuclideanManifold(3)
                ),
            )


def _parameterize_points(problem, blocks, ba_config, reconstruction):
    """A point whose track extends outside the config images is co-owned by
    poses not in this problem -> constant, unless explicitly variable."""
    for point3D_id, xyz in blocks["points"].items():
        num_obs = blocks["point_num_obs"].get(point3D_id, 0)
        track_length = reconstruction.points3D[point3D_id].track.length()
        if ba_config.has_constant_point(point3D_id) or (
            num_obs < track_length and not ba_config.has_variable_point(point3D_id)
        ):
            problem.set_parameter_block_constant(xyz)
            blocks["const_points"].add(point3D_id)


def _fix_gauge_three_points(problem, blocks):
    """Freeze three linearly independent points (rank check mirrors C++)."""
    fixed = []

    def maybe_add(xyz):
        candidate = fixed + [xyz]
        if np.linalg.matrix_rank(np.column_stack(candidate)) == len(candidate):
            fixed.append(xyz.copy())
            return True
        return False

    for point3D_id in sorted(blocks["const_points"]):
        if maybe_add(blocks["points"][point3D_id]) and len(fixed) >= 3:
            return
    for point3D_id in sorted(blocks["points"]):
        if point3D_id in blocks["const_points"]:
            continue
        if maybe_add(blocks["points"][point3D_id]):
            problem.set_parameter_block_constant(blocks["points"][point3D_id])
            blocks["const_points"].add(point3D_id)
            if len(fixed) >= 3:
                return
    logging.warning(
        f"Failed to fix gauge: only {len(fixed)} independent fixed points"
    )


def _fix_gauge_two_cams(problem, blocks, ba_options, ba_config, reconstruction):
    """6+1 gauge fix: one frame pose fully constant, plus one translation
    coordinate (the largest-baseline one) of a second frame."""
    if not ba_options.refine_rig_from_world:
        return
    # Config frames in deterministic order.
    frames, seen = [], set()
    for image_id in sorted(ba_config.images):
        frame_id = reconstruction.images[image_id].frame_id
        if frame_id not in seen:
            seen.add(frame_id)
            frames.append(frame_id)

    already_const = [f for f in frames if f in blocks["const_poses"]]
    if len(already_const) >= 2:
        return
    frame1 = already_const[0] if already_const else None

    frame2, fixed_dim = None, None
    for frame_id in frames:
        if frame1 is None:
            frame1 = frame_id
            continue
        if frame_id == frame1:
            continue
        baseline = (
            _rigid_from_pose7(blocks["poses"][frame1])
            * _rigid_from_pose7(blocks["poses"][frame_id]).inverse()
        ).translation
        dim = int(np.argmax(np.abs(baseline)))
        if abs(baseline[dim]) > 1e-9:
            frame2, fixed_dim = frame_id, dim
            break

    if frame1 is None or frame2 is None:
        logging.warning(
            "Failed to fix gauge with two cameras; falling back to three points"
        )
        _fix_gauge_three_points(problem, blocks)
        return

    if frame1 not in blocks["const_poses"]:
        problem.set_parameter_block_constant(blocks["poses"][frame1])
        blocks["const_poses"].add(frame1)
    if frame2 not in blocks["const_poses"]:
        if ba_options.constant_rig_from_world_rotation:
            manifold = pyceres.SubsetManifold(7, [0, 1, 2, 3, 4 + fixed_dim])
        else:
            manifold = pyceres.ProductManifold(
                pyceres.EigenQuaternionManifold(),
                pyceres.SubsetManifold(3, [fixed_dim]),
            )
        problem.set_manifold(blocks["poses"][frame2], manifold)


def _add_depth_factors(problem, blocks, ba_config, reconstruction, depth_ctx):
    """Phase 5 (ours): depth factors on the SAME pose/point blocks as the
    reprojection residuals, plus per-image alpha/beta with optional priors."""
    cfg = depth_ctx.config
    num_added = 0
    for image_id in sorted(ba_config.images):
        rows = depth_ctx.rows.get(image_id)
        if not rows:
            continue
        image = reconstruction.images[image_id]
        frame_id = image.frame_id
        pose_const = frame_id in blocks["const_poses"]
        if image_id not in depth_ctx.alphas and cfg.alpha_init == "median":
            depth_ctx.affine(image_id, median_depth_ratio(image, reconstruction, rows))
        alpha, beta = depth_ctx.affine(image_id)

        num_before = num_added
        for idx, p2d in enumerate(image.points2D):
            row = rows.get(idx)
            if row is None or row.is_sky or not p2d.has_point3D():
                continue
            point3D_id = p2d.point3D_id
            if point3D_id not in blocks["points"]:
                continue
            if pose_const and point3D_id in blocks["const_points"]:
                continue  # dead factor: nothing variable to constrain
            problem.add_residual_block(
                depth_ctx.make_cost(row), None,
                [blocks["poses"][frame_id], blocks["points"][point3D_id],
                 alpha, beta],
            )
            num_added += 1
        if num_added == num_before:
            continue  # alpha/beta never entered this problem

        if not cfg.per_image_scale:
            problem.set_parameter_block_constant(alpha)
        elif cfg.prior_sigma_alpha is not None:
            problem.add_residual_block(
                pyceres.factors.NormalPrior([1.0], [[cfg.prior_sigma_alpha**2]]),
                None, [alpha],
            )
        if not cfg.per_image_shift:
            problem.set_parameter_block_constant(beta)
        elif cfg.prior_sigma_beta is not None:
            problem.add_residual_block(
                pyceres.factors.NormalPrior([0.0], [[cfg.prior_sigma_beta**2]]),
                None, [beta],
            )
    return num_added


def build_problem(
    ba_options: pycolmap.BundleAdjustmentOptions,
    ba_config: pycolmap.BundleAdjustmentConfig,
    reconstruction: pycolmap.Reconstruction,
    depth_ctx: DepthContext | None = None,
    in_global: bool = True,
):
    """Compile ba_config/ba_options into a pyceres problem (+ depth factors).

    Returns (problem, blocks). The caller must keep `blocks` alive until
    after the solve (ceres holds raw pointers into its arrays) and use it
    for the write-back.
    """
    for rig in reconstruction.rigs.values():
        if len(rig.non_ref_sensors) > 0:
            raise NotImplementedError("build_problem assumes monocular identity rigs")

    problem = pyceres.Problem()
    loss = _make_loss(ba_options.ceres)
    blocks = {
        "poses": {},                    # frame_id -> pose7 [quat xyzw | t]
        "points": {},                   # point3D_id -> xyz
        "cams": {},                     # camera_id -> params
        "const_poses": set(),
        "const_points": set(),
        "const_cams": set(),
        "config_cams": set(),           # cameras serving at least one config image
        "point_num_obs": collections.Counter(),  # config-image obs per point
    }

    # Phases 1/2: residuals (order matters — later phases read the counts).
    for image_id in sorted(ba_config.images):
        _add_image_to_problem(problem, blocks, loss, ba_config, reconstruction, image_id)
    for point3D_id in ba_config.variable_points:
        _add_point_to_problem(problem, blocks, loss, ba_config, reconstruction, point3D_id)
    for point3D_id in ba_config.constant_points:
        _add_point_to_problem(problem, blocks, loss, ba_config, reconstruction, point3D_id)

    # Phase 3: constants and manifolds.
    _parameterize_cameras(problem, blocks, ba_options, ba_config, reconstruction)
    _parameterize_frames(problem, blocks, ba_options)
    _parameterize_points(problem, blocks, ba_config, reconstruction)

    # Phase 4: gauge.
    gauge = ba_config.fixed_gauge
    if gauge == pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD:
        _fix_gauge_two_cams(problem, blocks, ba_options, ba_config, reconstruction)
    elif gauge == pycolmap.BundleAdjustmentGauge.THREE_POINTS:
        _fix_gauge_three_points(problem, blocks)
    # UNSPECIFIED: the config's constants are expected to pin the gauge.

    # Phase 5: depth factors (ours).
    if depth_ctx is not None and depth_ctx.active(in_global):
        num_depth = _add_depth_factors(
            problem, blocks, ba_config, reconstruction, depth_ctx
        )
        logging.verbose(1, f"=> Added {num_depth} depth factors")

    return problem, blocks


# ---------------------------------------------------------------------------
# Solve, write-back, summary
# ---------------------------------------------------------------------------


class _SummaryShim:
    """The three members callers in this file consume from a
    pycolmap.BundleAdjustmentSummary, backed by a ceres SolverSummary."""

    def __init__(self, summary):
        self._summary = summary
        self.num_residuals = summary.num_residuals

    def brief_report(self) -> str:
        return self._summary.BriefReport()

    def is_solution_usable(self) -> bool:
        return self._summary.termination_type in (
            pyceres.TerminationType.CONVERGENCE,
            pyceres.TerminationType.NO_CONVERGENCE,
            pyceres.TerminationType.USER_SUCCESS,
        )


def _manual_solver_options(ceres_options):
    """Fallback if pycolmap's create_solver_options rejects a pyceres
    problem: copy the load-bearing fields, hardcode the BA-standard solver."""
    src = ceres_options.solver_options
    options = pyceres.SolverOptions()
    options.max_num_iterations = src.max_num_iterations
    options.max_linear_solver_iterations = src.max_linear_solver_iterations
    options.function_tolerance = src.function_tolerance
    options.gradient_tolerance = src.gradient_tolerance
    options.parameter_tolerance = src.parameter_tolerance
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    options.minimizer_progress_to_stdout = False
    return options


def _write_back(reconstruction, blocks) -> None:
    for frame_id, pose7 in blocks["poses"].items():
        if frame_id in blocks["const_poses"]:
            continue
        reconstruction.frame(frame_id).rig_from_world = _rigid_from_pose7(pose7)
    for point3D_id, xyz in blocks["points"].items():
        if point3D_id in blocks["const_points"]:
            continue
        reconstruction.points3D[point3D_id].xyz = xyz
    for camera_id, params in blocks["cams"].items():
        if camera_id in blocks["const_cams"]:
            continue
        reconstruction.cameras[camera_id].params = params


def solve_bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    ba_options: pycolmap.BundleAdjustmentOptions,
    ba_config: pycolmap.BundleAdjustmentConfig,
    depth_ctx: DepthContext | None = None,
    in_global: bool = True,
):
    problem, blocks = build_problem(
        ba_options, ba_config, reconstruction, depth_ctx, in_global
    )
    try:
        solver_options = ba_options.ceres.create_solver_options(ba_config, problem)
    except Exception as exc:
        logging.warning(f"create_solver_options failed ({exc}); using manual options")
        solver_options = _manual_solver_options(ba_options.ceres)
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, problem, summary)
    _write_back(reconstruction, blocks)
    return _SummaryShim(summary)


def adjust_global_bundle(
    mapper: pycolmap.IncrementalMapper,
    mapper_options: pycolmap.IncrementalMapperOptions,
    ba_options: pycolmap.BundleAdjustmentOptions,
    depth_ctx: DepthContext | None = None,
) -> bool:
    """Equivalent to mapper.adjust_global_bundle(...)"""
    reconstruction = mapper.reconstruction
    assert reconstruction is not None
    reg_frame_ids = reconstruction.reg_frame_ids()
    if len(reg_frame_ids) < 2:
        logging.fatal("At least two images must be registered for global BA")
    custom_ba_options = copy.deepcopy(ba_options)

    # Use stricter convergence criteria for first registered images
    if len(reg_frame_ids) < 10:  # kMinNumRegImagesForFastBA = 10
        custom_ba_options.ceres.solver_options.function_tolerance /= 10
        custom_ba_options.ceres.solver_options.gradient_tolerance /= 10
        custom_ba_options.ceres.solver_options.parameter_tolerance /= 10
        custom_ba_options.ceres.solver_options.max_num_iterations *= 2
        custom_ba_options.ceres.solver_options.max_linear_solver_iterations = (
            200
        )

    # Avoid degeneracies in bundle adjustment
    mapper.observation_manager.filter_observations_with_negative_depth()

    # Configure bundle adjustment
    ba_config = pycolmap.BundleAdjustmentConfig()
    for frame_id in reg_frame_ids:
        frame = reconstruction.frame(frame_id)
        for data_id in frame.data_ids:
            if data_id.sensor_id.type != pycolmap.SensorType.CAMERA:
                continue
            ba_config.add_image(data_id.id)

    # Fix the existing images, if option specified
    if mapper_options.fix_existing_frames:
        for frame_id in reg_frame_ids:
            if frame_id in mapper.existing_frame_ids:
                ba_config.set_constant_rig_from_world_pose(frame_id)

    for rig_id in mapper_options.constant_rigs:
        for sensor_id in reconstruction.rig(rig_id).non_ref_sensors:
            ba_config.set_constant_sensor_from_rig_pose(sensor_id)

    for camera_id in mapper_options.constant_cameras:
        ba_config.set_constant_cam_intrinsics(camera_id)

    # TODO: Add python support for prior positions
    # Fixing the gauge with two cameras leads to a more stable optimization
    # with fewer steps as compared to fixing three points.
    ba_config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)

    # Run bundle adjustment
    summary = solve_bundle_adjustment(
        reconstruction, custom_ba_options, ba_config, depth_ctx, in_global=True
    )
    logging.info("Global Bundle Adjustment")
    logging.info(summary.brief_report())
    return summary.is_solution_usable()


def iterative_global_refinement(
    mapper: pycolmap.IncrementalMapper,
    max_num_refinements: int,
    max_refinement_change: float,
    mapper_options: pycolmap.IncrementalMapperOptions,
    ba_options: pycolmap.BundleAdjustmentOptions,
    tri_options: pycolmap.IncrementalTriangulatorOptions,
    normalize_reconstruction: bool = True,
    depth_ctx: DepthContext | None = None,
) -> bool:
    """Equivalent to mapper.iterative_global_refinement(...)"""
    reconstruction = mapper.reconstruction
    mapper.complete_and_merge_tracks(tri_options)
    num_retriangulated_observations = mapper.retriangulate(tri_options)
    logging.verbose(
        1, f"=> Retriangulated observations: {num_retriangulated_observations}"
    )
    for _ in range(max_num_refinements):
        num_observations = reconstruction.compute_num_observations()
        # mapper.adjust_global_bundle(mapper_options, ba_options)
        if not adjust_global_bundle(mapper, mapper_options, ba_options, depth_ctx):
            return False
        if normalize_reconstruction:
            reconstruction.normalize()
        num_changed_observations = mapper.complete_and_merge_tracks(tri_options)
        num_changed_observations += mapper.filter_points(mapper_options)
        changed = (
            num_changed_observations / num_observations
            if num_observations > 0
            else 0
        )
        logging.verbose(1, f"=> Changed observations: {changed:.6f}")
        if changed < max_refinement_change:
            break
    return True


def adjust_local_bundle(
    mapper: pycolmap.IncrementalMapper,
    mapper_options: pycolmap.IncrementalMapperOptions,
    ba_options: pycolmap.BundleAdjustmentOptions,
    tri_options: pycolmap.IncrementalTriangulatorOptions,
    image_id: int,
    point3D_ids: set[int],
    depth_ctx: DepthContext | None = None,
) -> pycolmap.LocalBundleAdjustmentReport:
    """Equivalent to mapper.adjust_local_bundle(...)"""
    reconstruction = mapper.reconstruction
    assert reconstruction is not None
    report = pycolmap.LocalBundleAdjustmentReport()

    # Find images that have most 3D points with given image in common
    local_bundle = mapper.find_local_bundle(mapper_options, image_id)
    image_ids = set()

    # Do the bundle adjustment only if there is any connected images
    if local_bundle:
        ba_config = pycolmap.BundleAdjustmentConfig()
        ba_config.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)

        # Insert the images of all local frames.
        image = reconstruction.image(image_id)
        frame_ids = {image.frame_id}
        assert image.frame is not None
        for data_id in image.frame.image_ids:
            ba_config.add_image(data_id.id)
        for local_image_id in local_bundle:
            local_image = reconstruction.image(local_image_id)
            frame_ids.add(local_image.frame_id)
            assert local_image.frame is not None
            for data_id in local_image.frame.image_ids:
                ba_config.add_image(data_id.id)

        # Fix the existing images, if options specified
        if mapper_options.fix_existing_frames:
            for frame_id in frame_ids:
                if frame_id in mapper.existing_frame_ids:
                    ba_config.set_constant_rig_from_world_pose(frame_id)

        # Fix rig poses, if not all frames within the local bundle.
        num_frames_per_rig: dict[int, int] = collections.defaultdict(int)
        for frame_id in frame_ids:
            frame = reconstruction.frame(frame_id)
            num_frames_per_rig[frame.rig_id] += 1
        for rig_id, num_frames_local in num_frames_per_rig.items():
            if (
                rig_id in mapper_options.constant_rigs
                or num_frames_local < mapper.num_reg_frames_per_rig[rig_id]
            ):
                for sensor_id in reconstruction.rig(rig_id).non_ref_sensors:
                    ba_config.set_constant_sensor_from_rig_pose(sensor_id)

        # Fix camera intrinsics, if not all images within local bundle.
        # (Loop variable renamed from upstream: the pycolmap example reuses
        # `image_id` here, clobbering the function argument that
        # complete_image needs further down.)
        num_images_per_camera: dict[int, int] = collections.defaultdict(int)
        for config_image_id in ba_config.images:
            config_image = reconstruction.images[config_image_id]
            num_images_per_camera[config_image.camera_id] += 1
        for camera_id, num_images_local in num_images_per_camera.items():
            if (
                camera_id in mapper_options.constant_cameras
                or num_images_local
                < mapper.num_reg_images_per_camera[camera_id]
            ):
                ba_config.set_constant_cam_intrinsics(camera_id)

        # Make sure, we refine all new and short-track 3D points, no matter if
        # they are fully contained in the local image set or not. Do not include
        # long track 3D points as they are usually already very stable and
        # adding to them to bundle adjustment and track merging/completion would
        # slow down the local bundle adjustment significantly.
        variable_point3D_ids = set()
        for point3D_id in list(point3D_ids):
            point3D = reconstruction.point3D(point3D_id)
            kMaxTrackLength = 15
            if (
                point3D.error == -1.0
            ) or point3D.track.length() <= kMaxTrackLength:
                ba_config.add_variable_point(point3D_id)
                variable_point3D_ids.add(point3D_id)

        # Adjust the local bundle
        summary = solve_bundle_adjustment(
            mapper.reconstruction, ba_options, ba_config, depth_ctx,
            in_global=False,
        )
        logging.info("Local Bundle Adjustment")
        logging.info(summary.brief_report())

        image_ids = ba_config.images
        report.num_adjusted_observations = int(summary.num_residuals / 2)
        # Merge refined tracks with other existing points
        report.num_merged_observations = mapper.triangulator.merge_tracks(
            tri_options, variable_point3D_ids
        )
        # Complete tracks that may have failed to triangulate before refinement
        # of camera pose and calibration in bundle adjustment. This may avoid
        # that some points are filtered and helps for subsequent image
        # registrations.
        report.num_completed_observations = mapper.triangulator.complete_tracks(
            tri_options, variable_point3D_ids
        )
        report.num_completed_observations += mapper.triangulator.complete_image(
            tri_options, image_id
        )

    report.num_filtered_observations = (
        mapper.observation_manager.filter_points3D_in_images(
            mapper_options.filter_max_reproj_error,
            mapper_options.filter_min_tri_angle,
            image_ids,
        )
    )
    report.num_filtered_observations += (
        mapper.observation_manager.filter_points3D(
            mapper_options.filter_max_reproj_error,
            mapper_options.filter_min_tri_angle,
            point3D_ids,
        )
    )
    return report


def iterative_local_refinement(
    mapper: pycolmap.IncrementalMapper,
    max_num_refinements: int,
    max_refinement_change: float,
    mapper_options: pycolmap.IncrementalMapperOptions,
    ba_options: pycolmap.BundleAdjustmentOptions,
    tri_options: pycolmap.IncrementalTriangulatorOptions,
    image_id: int,
    depth_ctx: DepthContext | None = None,
) -> None:
    """Equivalent to mapper.iterative_local_refinement(...)"""
    custom_ba_options = copy.deepcopy(ba_options)
    for _ in range(max_num_refinements):
        # report = mapper.adjust_local_bundle(
        #     mapper_options,
        #     custom_ba_options,
        #     tri_options,
        #     image_id,
        #     mapper.get_modified_points3D(),
        # )
        report = adjust_local_bundle(
            mapper,
            mapper_options,
            custom_ba_options,
            tri_options,
            image_id,
            mapper.get_modified_points3D(),
            depth_ctx,
        )
        logging.verbose(
            1, f"=> Merged observations: {report.num_merged_observations}"
        )
        logging.verbose(
            1, f"=> Completed observations: {report.num_completed_observations}"
        )
        logging.verbose(
            1, f"=> Filtered observations: {report.num_filtered_observations}"
        )
        changed = 0.0
        if report.num_adjusted_observations > 0:
            changed = (
                report.num_merged_observations
                + report.num_completed_observations
                + report.num_filtered_observations
            ) / report.num_adjusted_observations
        logging.verbose(1, f"=> Changed observations: {changed:.6f}")
        if changed < max_refinement_change:
            break

        # Only use robust cost function for first iteration
        custom_ba_options.ceres.loss_function_type = (
            pycolmap.LossFunctionType.TRIVIAL
        )
    mapper.clear_modified_points3D()