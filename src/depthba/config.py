import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import yaml


@dataclass
class CameraConfig:
    single_camera: bool = True
    # manual-only fields:
    model: str | None = None                    # "PINHOLE", ...
    params: list[float] | None = None           # fx, fy, cx, cy
    width: int | None = None
    height: int | None = None
    intrinsics_are_approximate: bool = False


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
    def load(cls, path: Path) -> "DBConfig":
        raw = yaml.safe_load(path.read_text())
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys {unknown} in {path} — typo?")
        return cls(
            camera=CameraConfig(**raw.pop("camera")),
            matching=MatchingConfig(**raw.pop("matching", {})),
            sift=SiftConfig(**raw.pop("sift", {})),
            **raw,
        )