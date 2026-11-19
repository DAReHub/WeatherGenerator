import io
import sys
from pathlib import Path
import zipfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import json
import re
import copy
import os
from Plotting import plot_rainfall
from Plotting import plot_temperature

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "WeatherGeneratorApp"))
from WeatherGeneratorApp import main as wg

WORKERS = int(os.environ.get("WORKERS", 20))
MAX_REALS = int(os.environ.get("MAX_REALS", 10000))
MAX_LOCATIONS = int(os.environ.get("MAX_LOCATIONS", 1))

app = FastAPI()

class Coordinates(BaseModel):
    easting: int
    northing: int


class RealsRequest(BaseModel):
    ensemble_member: str = "M01"
    start_year: int = 2061
    period: int = 20
    resolution: str = "HOURLY"
    grid_size_km: int = 5
    coordinates: list[Coordinates]
    realisation_count: int = 100
    variables: list = ["Rainfall"]
    workers: int = 20
    seed: int = -1

DEFAULT_CONFIG = {
    "paths": {
        "parameters_directory": "data/parameters/",
        "input_weather_series": "data/parameters/IWS",
        "output_directory": "outputs/",
        "write_outputs": False
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
        "transformed_statitics": "TRANSFORMED_STATISTICS.parquet",
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

def build_config_from_api(p: dict) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    cfg["model"]["ensemble_member"] = p["ensemble_member"]
    cfg["temporal"].update(
        start_year=p["start_year"],
        period=p["period"],
        resolution=p["resolution"].upper(),
    )
    cfg["spatial"].update(
        grid_size_km=p["grid_size_km"],
        coordinates=[[c["easting"], c["northing"]] for c in p["coordinates"]],
    )
    cfg["processing"].update(
        realisation_count=p["realisation_count"],
        variables=[v.upper() for v in p["variables"]],
        workers=p["workers"],
        seed=p["seed"],
    )
    return cfg

@app.post("/weather-generator/")
async def get_reals(params: RealsRequest):

    if not params.coordinates:
        raise HTTPException(status_code=400, detail="No locations provided")

    if len(params.coordinates) > MAX_LOCATIONS:
        raise HTTPException(status_code=400, detail=f"Exceeds maximum number of locations ({MAX_LOCATIONS}) - edit your .env file")

    if params.realisation_count > MAX_REALS:
        raise HTTPException(status_code=400, detail=f"Exceeds maximum number of realisations ({MAX_REALS}) - edit your .env file")

    config = build_config_from_api(params.model_dump())
    print(config)

    results = wg.run(config)

    zip_filename = (f"Weather_Generator_Realisations_{datetime.now().isoformat(timespec="seconds")}.zip")
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for result in results:
            csv_buffer = io.StringIO()
            results[result].to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode("utf-8")
            csv_filename = f"{result}.csv"
            zf.writestr(csv_filename, csv_bytes)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )


# @app.post("/rainfall/")
# async def get_reals(params: RealsRequest):
#     if not params.locations:
#         raise HTTPException(status_code=400, detail="No locations provided")
#
#     if len(params.locations) > LOCATIONS_MAX_RAIN:
#         raise HTTPException(status_code=400, detail="Exceeds maximum number of locations")
#
#     if params.real_count > REALS_MAX_RAIN:
#         raise HTTPException(status_code=400, detail="Exceeds maximum number of realisations")
#
#     try:
#         if params.temporal_res == "HOURLY":
#             results = generate_realisations.main(
#                 params_path=f'data/Rainfall/'
#                             f'{params.period}_{params.temporal_res}/'
#                             f'UKCP_PARAMS_{params.spatial_res}KM_{params.period}'
#                             f'_{params.model}.parquet',
#                 coordinates=[loc.model_dump() for loc in params.locations],
#                 RealizationCount=params.real_count,
#                 RealizationStartYear=int(params.period.split("_")[0])
#             )
#
#         elif params.temporal_res == "DAILY":
#             results = Future_Temperature_Rainfall_Daily.daily_rainfall(
#                 params_path=f'data/Rainfall/'
#                             f'{params.period}_{params.temporal_res}/'
#                             f'UKCP_PARAMS_{params.spatial_res}KM_{params.period}'
#                             f'_{params.model}.parquet',
#                 coordinates=[loc.model_dump() for loc in params.locations],
#                 RealizationCount=params.real_count,
#                 RealizationStartYear=int(params.period.split("_")[0])
#             )
#
#         else:
#             raise HTTPException(status_code=400, detail="Invalid temporal resolution")
#
#         plots = plot_rainfall.main(
#             mean_rf_path='data/plotting/MEAN_HUK_RF.parquet',
#             rprf_path='data/plotting/RL24H_HUK.parquet',
#             all_realisations_path=results[0],
#             coordinates=[loc.model_dump() for loc in params.locations],
#             n_realizations=params.real_count,
#             RETURN_PERIODS=[2, 3, 5, 10, 15, 20, 25, 30]
#         )
#
#         zip_filename = (
#             f"Weather_Generator_Realisations_{datetime.now().isoformat(timespec="seconds")}.zip"
#         )
#         zip_buffer = io.BytesIO()
#
#         with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
#             metadata = {
#                 "download_time": datetime.now().isoformat(timespec="seconds"),
#                 "variable": "RAINFALL",
#                 "UKCP18_ensemble_member": params.model,
#                 "period": params.period.replace("_", "-"),
#                 "temporal_resolution": params.temporal_res,
#                 "centroid_locations": [loc.model_dump() for loc in params.locations],
#                 "unit": "mm",
#                 "realisation_count": params.real_count,
#                 "grid_resolution": "5km",
#                 "crs": {
#                     "name": "OSGB36 / British National Grid",
#                     "epsg": "EPSG:27700",
#                 }
#             }
#             zf.writestr("metadata.json", json.dumps(metadata, indent=2))
#
#             for result in results:
#                 csv_buffer = io.StringIO()
#                 df_out = (
#                     result["reals"]
#                     # The below makes outputs consistent with Temperature outputs
#                     .drop(columns=["Year", "Month"], errors="ignore")
#                     .rename(columns=lambda c: re.sub(r"^Realization_(\d+)$", r"R_\1", c))
#                 )
#                 df_out.to_csv(csv_buffer, index=False)
#                 csv_bytes = csv_buffer.getvalue().encode("utf-8")
#
#                 csv_filename = f"E{result['easting']}_N{result['northing']}.csv"
#                 zf.writestr(csv_filename, csv_bytes)
#
#         zip_buffer.seek(0)
#
#         return StreamingResponse(
#             zip_buffer,
#             media_type="application/zip",
#             headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
#         )
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to generate realisations: {e}")
#
#
# @app.post("/temperature/")
# async def get_reals(params: RealsRequest):
#     if not params.locations:
#         raise HTTPException(status_code=400, detail="No locations provided")
#
#     if len(params.locations) > LOCATIONS_MAX_TEMP:
#         raise HTTPException(status_code=400, detail="Exceeds maximum number of locations")
#
#     if params.real_count > REALS_MAX_TEMP:
#         raise HTTPException(status_code=400, detail="Exceeds maximum number of realisations")
#
#     if params.temporal_res == "HOURLY":
#         raise HTTPException(status_code=400, detail="HOURLY temporal resolution not available for temperature")
#
#     try:
#         results = Future_Temperature_Rainfall_Daily.daily_temperture(
#             params_path=f'data/Rainfall/'
#                         f'{params.period}_{params.temporal_res}/'
#                         f'UKCP_PARAMS_{params.spatial_res}KM_{params.period}'
#                         f'_{params.model}.parquet',
#             input_weather_series_path=f'data/Temperature/'
#                                       f'{params.period}_InputWeatherSeries/'
#                                       f'UKCP_PARAMS_{params.spatial_res}KM_'
#                                       f'{params.period}_IWS_{params.model}.parquet',
#             r2_path="data/R2.parquet",
#             standard_error_path="data/STANDARD_ERROR.parquet",
#             wgen_params_path='data/WGENPARAMS.parquet',
#             transformation_fixed_path='data/TRANSFORMATION_FIXED.parquet',
#             transformed_statitics_path='data/TRANSFORMED_STATISTICS.parquet',
#             offset_path='data/OFFSET_DF.parquet',
#             coordinates=[loc.model_dump() for loc in params.locations],
#             RealizationCount=params.real_count,
#             RealizationStartYear=int(params.period.split("_")[0]),
#             predictors={
#                 ('temp_avg', 'DDD'): ['temp_avg_lag1'],
#                 ('temp_avg', 'DD'): ['temp_avg_lag1'],
#                 ('temp_avg', 'DW'): ['temp_avg_lag1', 'prcp'],
#                 ('temp_avg', 'WD'): ['temp_avg_lag1', 'prcp_lag1'],
#                 ('temp_avg', 'WW'): ['temp_avg_lag1'],
#                 ('dtr', 'DDD'): ['dtr_lag1'],
#                 ('dtr', 'DD'): ['dtr_lag1'],
#                 ('dtr', 'DW'): ['dtr_lag1', 'prcp'],
#                 ('dtr', 'WD'): ['dtr_lag1', 'prcp_lag1'],
#                 ('dtr', 'WW'): ['dtr_lag1'],
#                 ('vap_press', 'DDD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
#                 ('vap_press', 'DD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
#                 ('vap_press', 'DW'): ['vap_press_lag1', 'prcp', 'temp_avg', 'dtr'],
#                 ('vap_press', 'WD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
#                 ('vap_press', 'WW'): ['vap_press_lag1', 'prcp', 'temp_avg', 'dtr'],
#                 ('wind_speed', 'DDD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
#                 ('wind_speed', 'DD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
#                 ('wind_speed', 'DW'): ['wind_speed_lag1', 'prcp', 'temp_avg', 'dtr'],
#                 ('wind_speed', 'WD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
#                 ('wind_speed', 'WW'): ['wind_speed_lag1', 'prcp', 'temp_avg', 'dtr'],
#                 ('sun_dur', 'DDD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
#                 ('sun_dur', 'DD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
#                 ('sun_dur', 'DW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr'],
#                 ('sun_dur', 'WD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
#                 ('sun_dur', 'WW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr']
#             },
#             workers=WORKERS,
#         )
#
#         zip_filename = (
#             f"Weather_Generator_Realisations_{datetime.now().isoformat(timespec="seconds")}.zip"
#         )
#         zip_buffer = io.BytesIO()
#
#         with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
#             metadata = {
#                 "download_time": datetime.now().isoformat(timespec="seconds"),
#                 "variable": "TEMPERATURE",
#                 "UKCP18_ensemble_member": params.model,
#                 "period": params.period.replace("_", "-"),
#                 "temporal_resolution": params.temporal_res,
#                 "centroid_locations": [loc.model_dump() for loc in params.locations],
#                 "unit": "°C",
#                 "realisation_count": params.real_count,
#                 "grid_resolution": "5km",
#                 "crs": {
#                     "name": "OSGB36 / British National Grid",
#                     "epsg": "EPSG:27700",
#                 }
#             }
#             zf.writestr("metadata.json", json.dumps(metadata, indent=2))
#
#             for result in results:
#                 temp_avg_filled, temp_min, temp_max = result["reals"]
#
#                 variants = {
#                     "AVG": temp_avg_filled,
#                     "MIN": temp_min,
#                     "MAX": temp_max,
#                 }
#
#                 for label, df in variants.items():
#                     csv_buffer = io.StringIO()
#                     df.to_csv(csv_buffer, index=True, index_label="DateTime")
#                     csv_bytes = csv_buffer.getvalue().encode("utf-8")
#
#                     csv_filename = (
#                         f"E{result['easting']}_N{result['northing']}_{label}.csv"
#                     )
#                     zf.writestr(csv_filename, csv_bytes)
#
#         zip_buffer.seek(0)
#
#         return StreamingResponse(
#             zip_buffer,
#             media_type="application/zip",
#             headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
#         )
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to generate realisations: {e}")