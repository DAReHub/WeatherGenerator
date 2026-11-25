import pandas as pd
from WeatherVariables import rainfall
from WeatherVariables import temperature
from Plotting import plot_rainfall
from Plotting import plot_temperature

def process_rainfall(cfg, easting, northing, base_name):
    name = f"E{easting}_N{northing}_RAINFALL"
    print_name = f"Easting {easting}, Northing {northing} [Rainfall]"
    print(f"{print_name}: Processing")

    IWS = pd.read_parquet(cfg.paths.input_weather_series / f"RAINFALL_{base_name}")

    rainfall_reals = rainfall.GetRealizations(
        parameters=IWS,
        n_realizations=cfg.processing.realisation_count,
        RealizationStartYear=cfg.temporal.start_year,
        temp_resolution=cfg.temporal.resolution,
        EASTING=easting,
        NORTHING=northing,
        n_years=cfg.temporal.period,
        base_seed=cfg.processing.seed,
    )

    if cfg.paths.write_outputs:
        print(f"{print_name}: Writing")
        rainfall_reals.to_csv(
            cfg.paths.output_directory / f"{name}.csv",
            index=False
        )

    if cfg.paths.write_outputs and cfg.plotting.create_plots:  # todo: what to do about baseline plotting?
        print(f"{print_name}: Plotting")
        # TODO: for this weather variable and others: write plots to memory and then either write to disk here or iobuffer to return to user through UI
        plot_rainfall.main(
            mean_rf_path=cfg.plotting.mean_rainfall_path,
            rprf_path=cfg.plotting.rprf_path,
            realisations=rainfall_reals,
            EASTING=easting,
            NORTHING=northing,
            n_realizations=cfg.processing.realisation_count,
            return_periods=cfg.plotting.return_periods,
            output_dir=cfg.paths.output_directory,
        )

    return rainfall_reals

def process_temperature(cfg, results, easting, northing, base_name):
    name = f"E{easting}_N{northing}_TEMPERATURE"
    print_name = f"Easting {easting}, Northing {northing} [TEMPERATURE]"
    print(f"{print_name}: Processing")

    IWS = pd.read_parquet(cfg.paths.input_weather_series / f"TEMPERATURE_{base_name}")
    r2 = pd.read_parquet(cfg.paths.parameters_directory / cfg.temperature.R2)
    SER = pd.read_parquet(cfg.paths.parameters_directory / cfg.temperature.standard_error)
    WGPAR = pd.read_parquet(cfg.paths.parameters_directory / cfg.temperature.wgen_params)
    transfix = pd.read_parquet(cfg.paths.parameters_directory / cfg.temperature.transformation_fixed)
    transtat = pd.read_parquet(cfg.paths.parameters_directory / cfg.temperature.transformed_statistics)
    offset = pd.read_parquet(cfg.paths.parameters_directory / cfg.temperature.offset)

    temp_reals_dict = temperature.GetRealizations(
        all_realizations_df=results["RAINFALL"],
        realisation_count=cfg.processing.realisation_count,
        IWS=IWS,
        r2=r2,
        SER=SER,
        WGPAR=WGPAR,
        transfix=transfix,
        transtat=transtat,
        offset=offset,
        easting=easting,
        northing=northing,
        predictors={  # todo: REMOVE HARDCODING
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
            ('wind_speed', 'DW'): ['wind_speed_lag1', 'prcp', 'temp_avg',
                                   'dtr'],
            ('wind_speed', 'WD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],
            ('wind_speed', 'WW'): ['wind_speed_lag1', 'prcp', 'temp_avg',
                                   'dtr'],
            ('sun_dur', 'DDD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],
            ('sun_dur', 'DD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],
            ('sun_dur', 'DW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr'],
            ('sun_dur', 'WD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],
            ('sun_dur', 'WW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr']
        },
        workers=cfg.processing.workers,
        seed=cfg.processing.seed
    )

    if cfg.paths.write_outputs:
        print(f"{print_name}: Writing")
        for key, value in temp_reals_dict.items():
            value.to_csv(
                cfg.paths.output_directory / f"{name}_{key}.csv",
                index=False
            )

    if cfg.paths.write_outputs and cfg.plotting.create_plots:
        print(f"{print_name}: Plotting")
        plot_temperature.main(
            obs_path=cfg.plotting.IWS_raw_statistics,
            realisations=temp_reals_dict["average_filled"],
            EASTING=easting,
            NORTHING=northing,
            n_realizations=cfg.processing.realisation_count,
            output_dir=cfg.paths.output_directory,
        )

    return temp_reals_dict

def process_wind_speed(cfg, results, easting, northing, base_name):
    pass

def process_vapour_pressure(cfg, results, easting, northing, base_name):
    pass

def process_sunshine_duration(cfg, results, easting, northing, base_name):
    pass

def process_pet(cfg, results, easting, northing, base_name):
    pass