"""Configuration for the Weather Generator.

Both entry points (CLI via a YAML file, API via request parameters) end up
producing the same immutable `Config` object:

    cfg = load_config("config.yaml")                 # CLI
    cfg = build_config_from_api(params.model_dump()) # API

Pass `cfg` into functions explicitly rather than importing it as a global.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Allowed values
# --------------------------------------------------------------------------- #

ENSEMBLE_MEMBERS = (
    "BASELINE", "M01", "M04", "M05", "M06", "M07", "M08", "M09", "M10",
    "M11", "M12", "M13", "M15", "M23", "M25", "M27", "M29",
)
RESOLUTIONS = ("DAILY", "HOURLY")
VARIABLES = (
    "RAINFALL",
    "TEMPERATURE",
    "WIND SPEED",
    "VAPOUR PRESSURE",
    "SUNSHINE DURATION",
    "POTENTIAL EVAPOTRANSPIRATION",
)

# --------------------------------------------------------------------------- #
# Defaults (used by the API; keep in step with the schema below)
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "parameters_directory": "data/parameters/",
        "input_weather_series": "data/parameters/IWS",
        "output_directory": "outputs/",
        "write_outputs": False,
    },
    "model": {"ensemble_member": "M01"},
    "temporal": {"start_year": 2061, "period": 20, "resolution": "DAILY"},
    "spatial": {"grid_size_km": 5, "coordinates": []},
    "processing": {
        "realisation_count": 10,
        "variables": ["RAINFALL"],
        "workers": 20,
        "seed": 1,
    },
    "temperature": {
        "R2": "R2.parquet",
        "standard_error": "STANDARD_ERROR.parquet",
        "transformation_fixed": "TRANSFORMATION_FIXED.parquet",
        "transformed_statistics": "TRANSFORMED_STATISTICS.parquet",
        "offset": "OFFSET.parquet",
        "wgen_params": "WGENPARAMS.parquet",
    },
    "plotting": {
        "create_plots": True,
        "mean_rainfall_path": "data/plotting/MEAN_HUK_RF.parquet",
        "rprf_path": "data/plotting/RL24H_HUK.parquet",
        "IWS_raw_statistics": "data/plotting/InputWeatherSeries_RawStatistics.parquet",
        "return_periods": [2, 3, 5, 10, 15, 20, 25, 30],
    },
}

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=False)
class PathsConfig:
    parameters_directory: Path
    input_weather_series: Path
    output_directory: Path
    write_outputs: bool


@dataclass(frozen=True)
class ModelConfig:
    ensemble_member: str

    def __post_init__(self):
        if self.ensemble_member not in ENSEMBLE_MEMBERS:
            raise ValueError(
                f"ensemble_member must be one of {ENSEMBLE_MEMBERS}, "
                f"got {self.ensemble_member!r}"
            )


@dataclass(frozen=True)
class TemporalConfig:
    start_year: int
    period: int
    resolution: str

    def __post_init__(self):
        if self.resolution not in RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {RESOLUTIONS}, got {self.resolution!r}"
            )
        if self.period < 1:
            raise ValueError(f"period must be at least 1 year, got {self.period}")


@dataclass(frozen=True)
class SpatialConfig:
    grid_size_km: int
    # British National Grid (EPSG:27700): (easting, northing)
    coordinates: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if self.grid_size_km <= 0:
            raise ValueError(f"grid_size_km must be positive, got {self.grid_size_km}")
        for c in self.coordinates:
            if len(c) != 2:
                raise ValueError(f"each coordinate must be [easting, northing], got {c}")


@dataclass(frozen=False)
class ProcessingConfig:
    realisation_count: int
    variables: tuple[str, ...]
    workers: int
    seed: int  # always a concrete value after loading (-1 is resolved)

    def __post_init__(self):
        if self.realisation_count < 1:
            raise ValueError("realisation_count must be at least 1")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if not self.variables:
            raise ValueError("at least one variable must be selected")
        bad = [v for v in self.variables if v not in VARIABLES]
        if bad:
            raise ValueError(f"unknown variable(s) {bad}; options are {VARIABLES}")


@dataclass(frozen=True)
class TemperatureConfig:
    R2: str
    standard_error: str
    transformation_fixed: str
    transformed_statistics: str  # spelling matches the YAML key
    offset: str
    wgen_params: str


@dataclass(frozen=True)
class PlottingConfig:
    create_plots: bool
    mean_rainfall_path: Path
    rprf_path: Path
    IWS_raw_statistics: Path
    return_periods: tuple[int, ...]


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    model: ModelConfig
    temporal: TemporalConfig
    spatial: SpatialConfig
    processing: ProcessingConfig
    temperature: TemperatureConfig
    plotting: PlottingConfig

    @property
    def end_year(self) -> int:
        return self.temporal.start_year + self.temporal.period - 1

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict (Paths left as Path objects), including derived values."""
        data = asdict(self)
        data["temporal"]["end_year"] = self.end_year
        return data

    def write_json(self, path: str | Path) -> None:
        """Dump the config to a JSON file (Paths written as strings)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _path(value: str | Path, base_dir: Path | None) -> Path:
    p = Path(value)
    return p if base_dir is None or p.is_absolute() else base_dir / p


def config_from_dict(raw: dict[str, Any], base_dir: str | Path | None = None) -> Config:
    """Build a validated Config from a nested dict.

    base_dir: if given, relative paths are resolved against it (e.g. the repo
    root) so results don't depend on the directory the script was launched from.
    """
    base = Path(base_dir) if base_dir is not None else None

    p = raw["paths"]
    t = raw["temporal"]
    s = raw["spatial"]
    pr = raw["processing"]
    pl = raw["plotting"]

    seed = int(pr["seed"])
    if seed == -1:
        seed = random.randint(0, 2**31 - 1)

    return Config(
        paths=PathsConfig(
            parameters_directory=_path(p["parameters_directory"], base),
            input_weather_series=_path(p["input_weather_series"], base),
            output_directory=_path(p["output_directory"], base),
            write_outputs=bool(p["write_outputs"]),
        ),
        model=ModelConfig(ensemble_member=str(raw["model"]["ensemble_member"]).upper()),
        temporal=TemporalConfig(
            start_year=int(t["start_year"]),
            period=int(t["period"]),
            resolution=str(t["resolution"]).upper(),
        ),
        spatial=SpatialConfig(
            grid_size_km=int(s["grid_size_km"]),
            coordinates=tuple((c[0], c[1]) if len(c) == 2 else tuple(c) for c in s["coordinates"]),
        ),
        processing=ProcessingConfig(
            realisation_count=int(pr["realisation_count"]),
            variables=tuple(str(v).upper() for v in pr["variables"]),
            workers=int(pr["workers"]),
            seed=seed,
        ),
        temperature=TemperatureConfig(**raw["temperature"]),
        plotting=PlottingConfig(
            create_plots=bool(pl["create_plots"]),
            mean_rainfall_path=_path(pl["mean_rainfall_path"], base),
            rprf_path=_path(pl["rprf_path"], base),
            IWS_raw_statistics=_path(pl["IWS_raw_statistics"], base),
            return_periods=tuple(int(x) for x in pl["return_periods"]),
        ),
    )


def load_config(path: str | Path, base_dir: str | Path | None = None) -> Config:
    """CLI entry point: read a YAML file and return a Config."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return config_from_dict(raw, base_dir=base_dir)


def build_config_from_api(p: dict[str, Any], base_dir: str | Path | None = None) -> Config:
    """API entry point: apply request parameters over DEFAULT_CONFIG.

    Raises ValueError if the combination of values is invalid.
    """
    raw = copy.deepcopy(DEFAULT_CONFIG)

    raw["model"]["ensemble_member"] = p["ensemble_member"]
    raw["temporal"].update(
        start_year=p["start_year"],
        period=p["period"],
        resolution=p["resolution"].upper(),
    )
    raw["spatial"].update(
        grid_size_km=p["grid_size_km"],
        coordinates=[[c["easting"], c["northing"]] for c in p["coordinates"]],
    )
    raw["processing"].update(
        realisation_count=p["realisation_count"],
        variables=[v.upper() for v in p["variables"]],
        workers=p["workers"],
        seed=p["seed"],
    )
    return config_from_dict(raw, base_dir=base_dir)