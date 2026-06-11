# ============================================================
# main.py — FastAPI Application with Caching
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager

from model_utils import load_model_and_scalers, get_model, get_scaler_y, get_feature_meta
from weather_client import fetch_historical_weather, fetch_merged_weather, LOCATIONS
from preprocessing import preprocess_weather_to_sequence, preprocess_forecast_windows, SEQUENCE_LENGTH, _preprocess_full_dataframe, get_disease_risks_for_day
from cache_manager import CacheManager

BASE_PATH = os.environ.get("MODEL_BASE_PATH", "/home/susilovps/pest_prediction_v2")
CACHE_DIR = os.path.join(BASE_PATH, "cache")

# Cache TTLs (in seconds)
CACHE_TTL_PREDICT = 3 * 60 * 60    # 3 hours for current day prediction
CACHE_TTL_FORECAST = 6 * 60 * 60   # 6 hours for future forecast

# Global cache instance
cache = CacheManager(CACHE_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  JEMBER PEST PREDICTION API — V2 AA-LSTM-AEA")
    print("=" * 60)
    load_model_and_scalers(BASE_PATH)
    print(f"  Cache directory: {CACHE_DIR}")
    print("=" * 60)
    print(f"  Ready: http://0.0.0.0:8000/docs")
    print("=" * 60)
    yield
    print("Shutting down...")


app = FastAPI(
    title="Jember Pest Prediction API",
    description="LSTM-based pest risk forecasting for Jember rainfed rice paddies.",
    version="2.2.0",
    lifespan=lifespan,
)


def get_risk_level(risk: float) -> str:
    if risk >= 70: return "HIGH"
    elif risk >= 40: return "MODERATE"
    elif risk >= 20: return "LOW"
    return "MINIMAL"


PEST_NAMES = {
    "bph": "Brown Planthopper (Wereng Coklat)",
    "ysb": "Yellow Stem Borer (Penggerek Batang Kuning)",
    "rlf": "Rice Leaf Folder (Penggulung Daun)",
    "wst": "Rice Bug (Walang Sangit)",
    "rat": "Field Rat (Tikus Sawah)",
    "snail": "Golden Apple Snail (Keong Mas)",
}

PEST_SCIENTIFIC = {
    "bph": "Nilaparvata lugens", "ysb": "Scirpophaga incertulas",
    "rlf": "Cnaphalocrocis medinalis", "wst": "Leptocorisa oratorius",
    "rat": "Rattus argentiventer", "snail": "Pomacea canaliculata",
}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "V2 AA-LSTM-AEA + Location Embedding", "locations": len(LOCATIONS)}


@app.get("/locations")
async def list_locations():
    return [{"id": loc_id, "name": loc["name"], "lat": loc["lat"], "lon": loc["lon"]} for loc_id, loc in sorted(LOCATIONS.items())]


@app.get("/predict/{location_id}")
async def predict_today(location_id: int):
    """Predict pest risk for TODAY (cached for 3 hours)."""
    if location_id not in LOCATIONS:
        raise HTTPException(status_code=404, detail=f"Location {location_id} not found")
    
    cache_key = f"predict_{location_id}"
    
    # 1. Check Cache
    cached_result = cache.get(cache_key, CACHE_TTL_PREDICT)
    if cached_result:
        cached_result["from_cache"] = True
        cached_result["performance_ms"] = {"total": 1.5} # ~1ms disk read
        return cached_result

    # 2. Cache Miss -> Compute
    start_time = time.time()
    try:
        weather_data = await fetch_historical_weather(location_id, days=67)
        fetch_time = time.time() - start_time
        
        X_seq, A_seq = preprocess_weather_to_sequence(weather_data, location_id)
        preprocess_time = time.time() - start_time - fetch_time
        
        model = get_model()
        loc_input = np.array([[location_id - 1]], dtype=np.int32)
        y_pred_scaled = model.predict([X_seq, A_seq, loc_input], verbose=0, batch_size=1)
        predict_time = time.time() - start_time - fetch_time - preprocess_time
        
        scaler_y = get_scaler_y()
        meta = get_feature_meta()
        y_pred = np.clip(scaler_y.inverse_transform(y_pred_scaled), 1, 99)
        
        predictions = {}
        for i, col in enumerate(meta["target_cols"]):
            pest_key = col.replace("_risk", "")
            predictions[col] = {
                "risk_percentage": round(float(y_pred[0, i]), 2),
                "risk_level": get_risk_level(y_pred[0, i]),
                "common_name": PEST_NAMES.get(pest_key, pest_key),
                "scientific_name": PEST_SCIENTIFIC.get(pest_key, ""),
            }
        
        total_time = time.time() - start_time
        
        # Calculate deterministic diseases from the preprocessed target day
        df_full = _preprocess_full_dataframe(pd.DataFrame(weather_data))
        target_day_row = df_full.iloc[-1]
        disease_risks = get_disease_risks_for_day(target_day_row)

        result = {
            "success": True,
            "from_cache": False,
            "location": {"id": location_id, "name": LOCATIONS[location_id]["name"], "lat": LOCATIONS[location_id]["lat"], "lon": LOCATIONS[location_id]["lon"]},
            "prediction_for_date": weather_data[-1]["datetime"].isoformat(),
            "lookback_days": SEQUENCE_LENGTH,
            "predictions": predictions,
            "diseases": disease_risks,
            "performance_ms": {
                "weather_fetch": round(fetch_time * 1000, 1),
                "preprocessing": round(preprocess_time * 1000, 1),
                "model_inference": round(predict_time * 1000, 1),
                "total": round(total_time * 1000, 1),
            }
        }
        
        # 3. Save to Cache
        cache.set(cache_key, result)
        
        return result
        
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Weather API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/forecast/{location_id}")
async def forecast_pest_risk(
    location_id: int,
    days: int = Query(default=14, ge=1, le=14, description="Number of days to forecast (1-14)")
):
    """Forecast pest risk for FUTURE days (cached for 6 hours)."""
    if location_id not in LOCATIONS:
        raise HTTPException(status_code=404, detail=f"Location {location_id} not found")
    
    cache_key = f"forecast_{location_id}_{days}"
    
    # 1. Check Cache
    cached_result = cache.get(cache_key, CACHE_TTL_FORECAST)
    if cached_result:
        cached_result["from_cache"] = True
        cached_result["performance_ms"] = {"total": 2.0}
        return cached_result

    # 2. Cache Miss -> Compute
    start_time = time.time()
    try:
        weather_data = await fetch_merged_weather(location_id, forecast_days=days)
        fetch_time = time.time() - start_time
        
        X_seq, A_seq, pred_dates = preprocess_forecast_windows(weather_data, location_id, n_forecast_days=days)
        preprocess_time = time.time() - start_time - fetch_time
        
        model = get_model()
        loc_input = np.full((len(pred_dates), 1), location_id - 1, dtype=np.int32)
        y_pred_scaled = model.predict([X_seq, A_seq, loc_input], verbose=0, batch_size=len(pred_dates))
        predict_time = time.time() - start_time - fetch_time - preprocess_time
        
        scaler_y = get_scaler_y()
        meta = get_feature_meta()
        y_pred = np.clip(scaler_y.inverse_transform(y_pred_scaled), 1, 99)
        
        # Preprocess full dataframe once to get disease environmental data
        df_full = _preprocess_full_dataframe(pd.DataFrame(weather_data))

        forecasts = []
        for i, pred_date in enumerate(pred_dates):
            # Get the corresponding row for disease calculation
            target_row_idx = (i + SEQUENCE_LENGTH) - 1 
            if target_row_idx < len(df_full):
                day_diseases = get_disease_risks_for_day(df_full.iloc[target_row_idx])
            else:
                day_diseases = {}

            day_data = {"day_number": i + 1, "date": pred_date.isoformat(), "day_name": pred_date.strftime("%A"), "pests": {}, "diseases": day_diseases}
            for j, col in enumerate(meta["target_cols"]):
                pest_key = col.replace("_risk", "")
                day_data["pests"][col] = {
                    "risk_percentage": round(float(y_pred[i, j]), 2),
                    "risk_level": get_risk_level(y_pred[i, j]),
                    "common_name": PEST_NAMES.get(pest_key, pest_key),
                }
            forecasts.append(day_data)
            
        total_time = time.time() - start_time
        
        result = {
            "success": True,
            "from_cache": False,
            "location": {"id": location_id, "name": LOCATIONS[location_id]["name"], "lat": LOCATIONS[location_id]["lat"], "lon": LOCATIONS[location_id]["lon"]},
            "forecast_info": {
                "days_requested": days, "days_returned": len(forecasts),
                "forecast_start": forecasts[0]["date"] if forecasts else None,
                "forecast_end": forecasts[-1]["date"] if forecasts else None,
                "lookback_per_prediction": SEQUENCE_LENGTH,
            },
            "forecasts": forecasts,
            "performance_ms": {
                "weather_fetch": round(fetch_time * 1000, 1),
                "preprocessing": round(preprocess_time * 1000, 1),
                "model_inference": round(predict_time * 1000, 1),
                "total": round(total_time * 1000, 1),
            }
        }
        
        # 3. Save to Cache
        cache.set(cache_key, result)
        
        return result
        
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Weather API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")