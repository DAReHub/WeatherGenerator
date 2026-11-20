import time
import sys
import argparse
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime
import random
import json
from WeatherVariables import rainfall
from WeatherVariables import temperature
from Plotting import plot_rainfall
from Plotting import plot_temperature

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        print("config loaded: ", config)
        return config

def create_output_dir(dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_directory = dir / timestamp
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory

def run(config: dict):
    print(config)
    start_time = time.time()

    parameters_directory = Path(config["paths"]["parameters_directory"])
    IWS_path = Path(config["paths"]["input_weather_series"])
    output_directory = Path(config["paths"]["output_directory"])
    write_outputs = bool(config["paths"]["write_outputs"])
    model = str(config["model"]["ensemble_member"])
    start_year = int(config["temporal"]["start_year"])
    period = int(config["temporal"]["period"])
    end_year = start_year + period - 1
    temp_resolution = str(config["temporal"]["resolution"])
    grid_size = int(config["spatial"]["grid_size_km"])
    try:
        coordinates = [tuple(coord) for coord in config["spatial"]["coordinates"]]
    except (KeyError, TypeError) as e:
        raise ValueError(
            "Failed to load coordinates: check formatting in config file."
        ) from e
    realisation_count = int(config["processing"]["realisation_count"])
    variables = list(config["processing"]["variables"])
    workers = int(config["processing"]["workers"])
    R2_path = parameters_directory / config["temperature"]["R2"]
    standard_error_path = parameters_directory / config["temperature"]["standard_error"]
    transformation_fixed_path = parameters_directory / config["temperature"]["transformation_fixed"]
    transformed_statistics_path = parameters_directory / config["temperature"]["transformed_statitics"]
    offset_path = parameters_directory / config["temperature"]["offset"]
    wgen_params_path = parameters_directory / config["temperature"]["wgen_params"]
    seed = config["processing"]["seed"]
    create_plots = bool(config["plotting"]["create_plots"])
    mean_rf_path = Path(config["plotting"]["mean_rainfall_path"])
    rprf_path = Path(config["plotting"]["rprf_path"]) #todo: What is rprf?
    IWS_raw_statistics = Path(config["plotting"]["IWS_raw_statistics"])
    return_periods = list(config["plotting"]["return_periods"])

    if "RAINFALL" not in variables:
        raise Exception("RAINFALL is a requred weather variable")

    if (
            (model == "BASELINE" and start_year != 2001)
            or (model == "BASELINE" and temp_resolution == "DAILY")
            or (model == "BASELINE" and len(variables) > 1)
    ):
        raise Exception("For BASELINE model: temporal start_year must be 2001, temporal resolution must be HOURLY, and processing variables must only include RAINFALL")

    if temp_resolution == "HOURLY" and len(variables) > 1:
        raise Exception("HOURLY temporal resolution only supported for weather variable RAINFALL")

    if seed == -1:
        seed = random.randint(1, 1000000)
        print(f"seed set randomly to {seed}")

    params_base_name = f"{grid_size}KM_{start_year}_{end_year}_{model}_{temp_resolution}.parquet"
    rainfall_parameters = pd.read_parquet(IWS_path / f"RAINFALL_{params_base_name}")

    run_directory = create_output_dir(output_directory)

    if write_outputs:
        with open(f"{run_directory}/config.json", "w") as f:
            json.dump(config, f, indent=2)

    states = {}
    results = {}

    for coods in coordinates:
        easting = coods[0]
        northing = coods[1]

        try:
            print(f"Easting {easting}, Northing {northing} [Rainfall]: Processing")
            processing_start_time = time.time()
            rainfall_reals = rainfall.GetRealizations(
                parameters=rainfall_parameters,
                n_realizations=realisation_count,
                RealizationStartYear=start_year,
                temp_resolution=temp_resolution,
                EASTING=easting,
                NORTHING=northing,
                n_years=period,
                base_seed=seed,
            )
            results[f"E{easting}_N{northing}_RAINFALL"] = rainfall_reals
            if write_outputs:
                print(f"Easting {easting}, Northing {northing} [RAINFALL]: Writing")
                rainfall_reals.to_csv(
                    run_directory / f"E{easting}_N{northing}_RAINFALL.csv",
                    index=False
                )
            if write_outputs and create_plots: # todo: what to do about baseline plotting?
                print(f"Easting {easting}, Northing {northing} [RAINFALL]: Plotting")
                # TODO: for this weather variable and others: write plots to memory and then either write to disk here or iobuffer to return to user through UI
                plot_rainfall.main(
                    mean_rf_path=mean_rf_path,
                    rprf_path=rprf_path,
                    realisations=rainfall_reals,
                    EASTING=easting,
                    NORTHING=northing,
                    n_realizations=realisation_count,
                    return_periods=return_periods,
                    output_dir=run_directory,
                )

            if "TEMPERATURE" in variables:
                print(f"Easting {easting}, Northing {northing} [TEMPERATURE]: Processing")
                IWS = pd.read_parquet(IWS_path / f"TEMPERATURE_{params_base_name}")
                r2 = pd.read_parquet(R2_path)
                SER = pd.read_parquet(standard_error_path)
                WGPAR = pd.read_parquet(wgen_params_path)
                transfix = pd.read_parquet(transformation_fixed_path)
                transtat = pd.read_parquet(transformed_statistics_path)
                offset = pd.read_parquet(offset_path)

                temp_reals_dict = temperature.process_temperature(
                    all_realizations_df=rainfall_reals,
                    realisation_count=realisation_count,
                    IWS=IWS,
                    r2=r2,
                    SER=SER,
                    WGPAR=WGPAR,
                    transfix=transfix,
                    transtat=transtat,
                    offset=offset,
                    easting=easting,
                    northing=northing,
                    predictors={ # todo: REMOVE HARDCODING
                        ('temp_avg', 'DDD'): ['temp_avg_lag1'],
                        ('temp_avg', 'DD'): ['temp_avg_lag1'],
                        ('temp_avg', 'DW'): ['temp_avg_lag1', 'prcp'],
                        ('temp_avg', 'WD'): ['temp_avg_lag1', 'prcp_lag1'],
                        ('temp_avg', 'WW'): ['temp_avg_lag1'],
                        ('dtr', 'DDD'): ['dtr_lag1'],
                        ('dtr', 'DD'): ['dtr_lag1'],
                        ('dtr', 'DW'): ['dtr_lag1', 'prcp'],
                        ('dtr', 'WD'): ['dtr_lag1', 'prcp_lag1'],
                        ('dtr', 'WW'): ['dtr_lag1'],
                        ('vap_press', 'DDD'): ['vap_press_lag1', 'temp_avg', 'dtr'],
                        ('vap_press', 'DD'): ['vap_press_lag1', 'temp_avg', 'dtr'],
                        ('vap_press', 'DW'): ['vap_press_lag1', 'prcp', 'temp_avg', 'dtr'],
                        ('vap_press', 'WD'): ['vap_press_lag1', 'temp_avg', 'dtr'],
                        ('vap_press', 'WW'): ['vap_press_lag1', 'prcp', 'temp_avg', 'dtr'],
                        ('wind_speed', 'DDD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],
                        ('wind_speed', 'DD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],
                        ('wind_speed', 'DW'): ['wind_speed_lag1', 'prcp', 'temp_avg', 'dtr'],
                        ('wind_speed', 'WD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],
                        ('wind_speed', 'WW'): ['wind_speed_lag1', 'prcp', 'temp_avg', 'dtr'],
                        ('sun_dur', 'DDD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],
                        ('sun_dur', 'DD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],
                        ('sun_dur', 'DW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr'],
                        ('sun_dur', 'WD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],
                        ('sun_dur', 'WW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr']
                    },
                    workers=workers,
                    seed=seed
                )
                for key, value in temp_reals_dict.items():
                    results[f"E{easting}_N{northing}_TEMPERATURE_{key}"] = value
                if write_outputs:
                    print(f"Easting {easting}, Northing {northing} [TEMPERATURE]: Writing")
                    for key, value in temp_reals_dict.items():
                        value.to_csv(
                            run_directory / f"E{easting}_N{northing}_TEMPERATURE_{key}.csv",
                            index=False
                        )
                if write_outputs and create_plots:
                    print(f"Easting {easting}, Northing {northing} [TEMPERATURE]: Plotting")
                    plot_temperature.main(
                        obs_path=IWS_raw_statistics,
                        realisations=temp_reals_dict["average_filled"],
                        EASTING=easting,
                        NORTHING=northing,
                        n_realizations=realisation_count,
                        output_dir=run_directory,
                    )

            if "WIND" in variables:
                pass

            if "VAPOUR PRESSURE" in variables:
                pass

            if "SUNSHINE DURATION" in variables:
                pass

            if "POTENTIAL EVAPOTRANSPIRATION" in variables:
                pass

            processing_time = round(time.time() - processing_start_time)
            states[(easting, northing)] = f"SUCCESS: completed in {processing_time} seconds"

        except Exception as e:
            print(f"Failed to process [{easting}, {northing}]: {e}]")
            states[(easting, northing)] = f"FAILED: {e}"

    end_time = time.time()
    print(f"——— Done - finished in {round(end_time - start_time)} seconds ———")

    for k, v in states.items():
        print(k, v)

    if not write_outputs:
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