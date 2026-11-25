import time
import sys
import argparse
from pathlib import Path
from datetime import datetime
import random
import json
from config import Config, load_config
from dataclasses import asdict
from processors import (
    process_rainfall, process_temperature, process_wind_speed,
    process_vapour_pressure, process_sunshine_duration, process_pet,
)

DEPENDENTS = {
    "RAINFALL": ["RAINFALL", "TEMPERATURE", "WIND SPEED", "VAPOUR PRESSURE", "SUNSHINE DURATION", "POTENTIAL EVAPOTRANSPIRATION"],
    "TEMPERATURE": ["TEMPERATURE", "WIND SPEED", "VAPOUR PRESSURE", "SUNSHINE DURATION", "POTENTIAL EVAPOTRANSPIRATION"],
    "WIND SPEED": ["WIND SPEED", "POTENTIAL EVAPOTRANSPIRATION"],
    "VAPOUR PRESSURE": ["VAPOUR PRESSURE", "POTENTIAL EVAPOTRANSPIRATION"],
    "SUNSHINE DURATION": ["SUNSHINE DURATION", "POTENTIAL EVAPOTRANSPIRATION"],
    "POTENTIAL EVAPOTRANSPIRATION": ["POTENTIAL EVAPOTRANSPIRATION"],
}

def create_output_dir(cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_directory = Path(cfg.paths.output_directory / timestamp)
    run_directory.mkdir(parents=True, exist_ok=True)
    cfg.paths.output_directory = run_directory
    return cfg

def run(cfg: Config):
    start_time = time.time()

    try:
        coordinates = [tuple(coord) for coord in cfg.spatial.coordinates]
    except (KeyError, TypeError) as e:
        raise ValueError(
            "Failed to load coordinates: check formatting in config file."
        ) from e

    if cfg.processing.seed == -1:
        new_seed = random.randint(1, 1000000)
        cfg.processing.seed = new_seed
        print(f"seed set randomly to {new_seed}")

    cfg = create_output_dir(cfg)

    if cfg.paths.write_outputs:
        with open(cfg.paths.output_directory / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)

    if (
            (cfg.model.ensemble_member == "BASELINE" and cfg.temporal.start_year != 2001)
            or (cfg.model.ensemble_member == "BASELINE" and cfg.temporal.resolution == "DAILY")
            or (cfg.model.ensemble_member == "BASELINE" and len(cfg.processing.variables) > 1)
    ):
        raise Exception("For BASELINE model: temporal start_year must be 2001, temporal resolution must be HOURLY, and processing variables must only include RAINFALL")

    if not any(v in cfg.processing.variables for v in DEPENDENTS):
        raise Exception("No valid weather variable names in config")

    if cfg.temporal.resolution == "HOURLY" and len(cfg.processing.variables) > 1:
        raise Exception("HOURLY temporal resolution only supported for weather variable RAINFALL")

    end_year = cfg.temporal.start_year + cfg.temporal.period - 1
    base_name = f"{cfg.spatial.grid_size_km}KM_{cfg.temporal.start_year}_{end_year}_{cfg.model.ensemble_member}_{cfg.temporal.resolution}.parquet"

    states = {}
    results = {}

    processing_start_time = time.time()

    for coods in coordinates:
        easting = coods[0]
        northing = coods[1]

        try:
            if any(item in DEPENDENTS["RAINFALL"] for item in cfg.processing.variables):
                results["RAINFALL"] = process_rainfall(
                    cfg, easting, northing, base_name)

            if any(item in DEPENDENTS["TEMPERATURE"] for item in cfg.processing.variables):
                results["TEMPERATURE"] = process_temperature(
                    cfg, results.copy(), easting, northing, base_name)

            if any(item in DEPENDENTS["WIND SPEED"] for item in cfg.processing.variables):
                results["WIND SPEED"] = process_wind_speed(
                    cfg, results.copy(), easting, northing, base_name)

            if any(item in DEPENDENTS["VAPOUR PRESSURE"] for item in cfg.processing.variables):
                results["VAPOUR PRESSURE"] = process_vapour_pressure(
                    cfg, results.copy(), easting, northing, base_name)

            if any(item in DEPENDENTS["SUNSHINE DURATION"] for item in cfg.processing.variables):
                results["SUNSHINE DURATION"] = process_sunshine_duration(
                    cfg, results.copy(), easting, northing, base_name)

            if any(item in DEPENDENTS["POTENTIAL EVAPOTRANSPIRATION"] for item in cfg.processing.variables):
                results["POTENTIAL EVAPOTRANSPIRATION"] = process_pet(
                    cfg, results.copy(), easting, northing, base_name)

            processing_time = round(time.time() - processing_start_time)
            states[(easting, northing)] = f"SUCCESS: completed in {processing_time} seconds"

        except Exception as e:
            print(f"Failed to process [{easting}, {northing}]: {e}]")
            states[(easting, northing)] = f"FAILED: {e}"

    end_time = time.time()
    print(f"——— Done - finished in {round(end_time - start_time)} seconds ———")

    for k, v in states.items():
        print(k, v)

    if not cfg.paths.write_outputs:
        return results
    else:
        return 0

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.yaml",
        required=False,
        help='Path to config .yaml file',
    )
    args = parser.parse_args(argv)
    results = run(load_config(args.config))
    return results

if __name__ == "__main__":
    main(sys.exit(main()))