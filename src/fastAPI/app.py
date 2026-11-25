import io
import sys
from pathlib import Path
import zipfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import copy
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "WeatherGeneratorApp"))
from WeatherGeneratorApp import main as wg
from WeatherGeneratorApp.config import config_from_dict, Config

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

def build_config_from_api(p: dict) -> Config:
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
    return config_from_dict(raw)

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