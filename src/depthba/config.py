import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import yaml


@dataclass
class CameraConfig:
    single_camera: bool = True
    model: str | None = None
    params: list[float] | None = None
    params_path: str | None = None   # relative to data_root, may contain "{sequence}"

    def __post_init__(self) -> None:
        if self.params is not None and self.params_path is not None:
            raise ValueError("camera: specify either params or params_path, not both")

    def resolve(self, data_root: Path, sequence: str | None) -> None:
        """Fill in `params` from `params_path`. No-op if params_path unset."""
        if self.params_path is None:
            return
        path_str = self.params_path
        if "{sequence}" in path_str:
            if sequence is None:
                raise ValueError(f"camera params_path {path_str!r} requires a sequence")
            path_str = path_str.format(sequence=sequence)
        full_path = data_root / path_str
        if not full_path.exists():
            raise ValueError(f"Camera params_path {full_path} does not exist")
        import json
        with open(full_path, "r") as f:
            intrinsics = json.load(f)
        self.params = [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]]
        if self.params:
            print(f"Intrinsics loaded from {full_path}: {self.params}")
        self.params_path = None  # mark resolved


@dataclass
class MatchingConfig:
    method: Literal["exhaustive", "sequential"] = "exhaustive"
    # sequential-only, for T&T video later:
    overlap: int = 10
    loop_detection: bool = False


@dataclass
class SiftConfig:
    num_features: int = 8192
    use_gpu: bool = False          # flip on for cluster


@dataclass
class PreprocessConfig:
    """amb3r inference + npz unpack into the canonical tree. raw_root and
    data_root (the trees these subdirs live under) stay in run.py's
    top-level config."""

    amb3r_repo: str                 # amb3r checkout with its own .venv
    image_subdir: str               # under raw_root, "{sequence}" placeholder
    out_subdir: str                 # under data_root, "{sequence}" placeholder

    @classmethod
    def from_dict(cls, raw: dict, source: str = "<dict>") -> "PreprocessConfig":
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys {unknown} in {source} — typo?")
        return cls(**raw)


@dataclass
class GSConfig:
    """3DGS training in the gaussian-splatting repo's own venv."""

    repo: str                       # gaussian-splatting checkout
    python_bin: str                 # interpreter of that repo's venv
    iterations: list[int] = field(default_factory=lambda: [7000, 30000])
    resolution: int = 1

    @classmethod
    def from_dict(cls, raw: dict, source: str = "<dict>") -> "GSConfig":
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys {unknown} in {source} — typo?")
        return cls(**raw)


@dataclass
class AttachConfig:
    """Sensor identity for one attach_depths ingest. Machine-specific inputs
    (database path, dump dir, force) stay in run.py's top-level config."""

    sensor: str                     # row key in depthba_depth_meta, e.g. "mda_native_k4"
    method: str                     # key into extractors.EXTRACTORS
    sigma_space: str | None = None  # "log"/"linear"/"inverse"; None = sensor emits no sigmas
    method_params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict, source: str = "<dict>") -> "AttachConfig":
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys {unknown} in {source} — typo?")
        return cls(**raw)


@dataclass
class DBConfig:
    image_path: str                # relative to data_root, resolved at run time
    stride: int
    camera: CameraConfig
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    sift: SiftConfig = field(default_factory=SiftConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if Path(self.image_path).is_absolute():
            raise ValueError(
                f"image_path must be relative to data_root for portability: {self.image_path}"
            )

    @classmethod
    def from_dict(cls, raw: dict, source: str = "<dict>") -> "DBConfig":
        raw = dict(raw)
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys {unknown} in {source} — typo?")

        camera_fields = raw.pop("camera", {})
        known_camera = {f.name for f in dataclasses.fields(CameraConfig)}
        unknown_camera = set(camera_fields) - known_camera
        if unknown_camera:
            raise ValueError(f"Unknown camera config keys {unknown_camera} in {source} — typo?")

        return cls(
            camera=CameraConfig(**camera_fields),
            matching=MatchingConfig(**raw.pop("matching", {})),
            sift=SiftConfig(**raw.pop("sift", {})),
            **raw,
        )
        
        
@dataclass
class DepthBAConfig:
    """Depth-factor layer for BA. Picks a sensor by name; everything about
    that sensor (extractor method, K, sigma_space) is read from its
    depthba_depth_meta row — the db is the single source of truth. Factor
    arity follows meta.num_modes (1 -> plain, >1 -> max-mixture).

    The per-image affine is SCALE ONLY: beta stays constant at 0 (the slot
    survives because the fork's factors take it as a parameter block). A
    free shift adds no measurable accuracy on our sensors, and mp-sfm pins
    it to zero for the same reason.

    Deliberate omissions: no wmin/gating knobs — later experimental
    conditions, added when the experiment exists. Sky exclusion is
    unconditional by design.
    """

    sensor: str | None = None            # row key in depthba_depth_meta; None = depth off
    depth_space: Literal["log", "linear", "inverse"] = "log"
    depth_in_global: bool = True
    depth_in_local: bool = False         
    sigma: float = 0.15                
    sigma_scale: float = 1.0           
    depth_loss: Literal["huber", "cauchy"] = "huber"
    huber_scale: float | None = 2.0      
    huber_adaptive: bool = True         
    shared_scale: bool = False           
    per_image_scale: bool = True         # alpha block variable (else constant at 1.0)
    prior_sigma_alpha: float | None = None   # None = no prior (weak default per design)
    alpha_init: Literal["median", "unit"] = "median"

    def __post_init__(self) -> None:
        # Literal is not enforced at runtime; a typo'd yaml value would
        # otherwise sail through and misroute factor construction.
        if self.depth_space not in ("log", "linear", "inverse"):
            raise ValueError(f"depth_space must be log/linear/inverse, got {self.depth_space!r}")
        if self.alpha_init not in ("median", "unit"):
            raise ValueError(f"alpha_init must be median/unit, got {self.alpha_init!r}")
        if self.depth_loss not in ("huber", "cauchy"):
            raise ValueError(f"depth_loss must be huber/cauchy, got {self.depth_loss!r}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.sigma_scale <= 0:
            raise ValueError(f"sigma_scale must be > 0, got {self.sigma_scale}")
        if self.huber_scale is not None and self.huber_scale <= 0:
            raise ValueError(f"huber_scale must be > 0 or null, got {self.huber_scale}")
        # huber_adaptive is vacuous without a scale to multiply (both the fit
        # and the loss are skipped), and it is now on by default — so
        # huber_scale: null alone must keep meaning "plain quadratic".
        if self.huber_adaptive and self.huber_scale is not None and self.depth_space != "log":
            raise ValueError(
                "huber_adaptive is log-space only: the linear/inverse residual "
                "formulas have no parity coverage against the fork's factors"
            )
        if self.shared_scale and self.per_image_scale:
            raise ValueError(
                "shared_scale is exclusive with per_image_scale: "
                "one global alpha replaces the per-image affine blocks"
            )
        if self.prior_sigma_alpha is not None and self.prior_sigma_alpha <= 0:
            raise ValueError(
                f"prior_sigma_alpha must be > 0 or null, got {self.prior_sigma_alpha}"
            )

    @classmethod
    def from_dict(cls, raw: dict, source: str = "<dict>") -> "DepthBAConfig":
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys {unknown} in {source} — typo?")
        return cls(**raw)

    @classmethod
    def load(cls, path: Path) -> "DepthBAConfig":
        raw = yaml.safe_load(path.read_text())
        return cls.from_dict(raw, source=str(path))
