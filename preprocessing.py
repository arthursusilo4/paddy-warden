# ============================================================
# preprocessing.py — V2 Feature Engineering for Inference
# ============================================================

import numpy as np
import pandas as pd
from datetime import date
from typing import Tuple, List

from model_utils import get_scaler_X, get_scaler_A, get_feature_meta

# Thermal parameters for GDD calculations (from V2 generator)
THERMAL_PARAMS = {
    "bph": {"tb": 12, "tmax": 32, "tleth": 35, "gdd_per_gen": 320},
    "ysb": {"tb": 12, "tmax": 30, "tleth": 35, "gdd_per_gen": 380},
    "rlf": {"tb": 10, "tmax": 28, "tleth": 34, "gdd_per_gen": 300},
    "wst": {"tb": 10, "tmax": 32, "tleth": 36, "gdd_per_gen": 450},
}

NITROGEN_OPTIMAL = {
    "rice_blast": 80, 
    "bacterial_leaf_blight": 120,
    "sheath_blight": 100, 
    "default": 100
}

SEQUENCE_LENGTH = 14


# ==================== ENVIRONMENTAL CALCULATIONS ====================

def calculate_vpd(temp: float, humidity: float) -> float:
    es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    ea = (humidity / 100) * es
    return max(0, es - ea)


def calculate_vpd_index(vpd: float, optimal_vpd: float = 0.8, spread: float = 0.5) -> float:
    return np.exp(-((vpd - optimal_vpd) ** 2 / (2 * spread ** 2)))


def calculate_lwd(humidity: float, precip: float, growth_stage: str) -> float:
    rh_min = max(humidity - 10, 0)
    lwd_dew = 4.0 if (rh_min > 85 and humidity > 80) else (2.5 if (rh_min > 80 and humidity > 75) else 0)
    lwd_rain = min(2 + (precip / 5), 8) if precip >= 0.5 else 0
    night_rain_bonus = 2.0 if (precip > 0 and humidity > 85) else 0
    lwd_irrigation = 3.0 if growth_stage in ['vegetative', 'reproductive'] else 0
    return min(lwd_dew + lwd_rain + night_rain_bonus + lwd_irrigation, 24)


def calculate_rhsi(humidity: float) -> float:
    if 70 <= humidity <= 95:
        return 0.0
    if humidity < 70:
        return -(70 - humidity) / 30
    return (humidity - 95) / 20


def calculate_gdd(temp: float, pest_type: str) -> float:
    p = THERMAL_PARAMS.get(pest_type, {"tb": 12, "tmax": 30, "tleth": 35})
    if temp < p["tb"]:
        return 0
    elif temp <= p["tmax"]:
        return temp - p["tb"]
    elif temp <= p["tleth"]:
        return p["tmax"] - p["tb"]
    return 0


def calculate_nsf(n_applied: float, disease_type: str = "default") -> float:
    optimal_n = NITROGEN_OPTIMAL.get(disease_type, 100)
    return 0.5 + (0.5 * min(n_applied / optimal_n, 2))


# ==================== GROWTH STAGE ESTIMATION ====================

def estimate_growth_state(dt: date) -> Tuple[str, int]:
    doy = dt.dayofyear
    
    if doy >= 305:
        days_since_nov1 = doy - 305
    else:
        days_since_nov1 = (365 - 305) + doy
    
    if doy >= 305 or doy <= 84:
        dsp = days_since_nov1
        if dsp <= 45:
            return "vegetative", dsp
        elif dsp <= 80:
            return "reproductive", dsp
        elif dsp <= 115:
            return "maturity", dsp
        else:
            return "fallow", 0
    
    if 105 <= doy <= 220:
        dsp = doy - 105
        if dsp <= 45:
            return "vegetative", dsp
        elif dsp <= 80:
            return "reproductive", dsp
        elif dsp <= 115:
            return "maturity", dsp
        else:
            return "fallow", 0
    
    return "fallow", 0


# ==================== ANOMALY DETECTION ====================

def detect_anomaly_simple(temp: float, temp_rolling: float, precip: float, month: int) -> Tuple[float, int]:
    anomaly_score = 0.0
    anomaly_flag = 0
    
    if abs(temp - temp_rolling) > 6:
        anomaly_score = max(anomaly_score, min(abs(temp - temp_rolling) / 10, 1.0))
        anomaly_flag = 1
    
    wet_months = {11, 12, 1, 2, 3, 4}
    if month in wet_months and precip == 0:
        anomaly_score = max(anomaly_score, 0.5)
        anomaly_flag = 1
    
    if precip > 50:
        anomaly_score = max(anomaly_score, min(precip / 100, 1.0))
        anomaly_flag = 1
    
    return anomaly_score, anomaly_flag


# ==================== DISEASE PROBABILITY PROXIES ====================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def calculate_disease_infection_probs(row: pd.Series) -> dict:
    lwd = row.get("lwd", 0)
    temp = row.get("temp", 28)
    humidity_norm = row.get("humidity_norm", 0.8)
    rainfall_events_7d = row.get("rainfall_events_7d", 0)
    soil_moisture_pct = row.get("soil_moisture_pct", 70)
    field_age_factor = row.get("field_age_factor", 0.5)
    bph_gen_progress = row.get("bph_gen_progress", 50)
    dsp = row.get("days_since_planting", 0)
    
    temp_suit_blast = max(0, 1 - abs(temp - 23) / 10)
    blast_score = (lwd/24 * 0.4 + temp_suit_blast * 0.3 + humidity_norm * 0.3 - 0.5) * 8
    rice_blast_prob = sigmoid(blast_score)
    
    temp_suit_blb = max(0, 1 - abs(temp - 27.5) / 8)
    blb_score = (rainfall_events_7d/7 * 0.4 + humidity_norm * 0.3 + temp_suit_blb * 0.3 - 0.6) * 8
    blb_prob = sigmoid(blb_score)
    
    shb_score = (soil_moisture_pct/100 * 0.4 + humidity_norm * 0.4 - 0.5) * 8
    shb_prob = sigmoid(shb_score)
    
    temp_suit_bs = max(0, 1 - abs(temp - 26.5) / 8)
    bs_score = (field_age_factor * 0.3 + lwd/24 * 0.3 + temp_suit_bs * 0.2 + humidity_norm * 0.2 - 0.5) * 8
    bs_prob = sigmoid(bs_score)
    
    transmission_window = 1.0 if 20 <= dsp <= 60 else (0.5 if dsp < 80 else 0.1)
    tungro = min((bph_gen_progress / 100) * transmission_window * 100, 99)
    
    return {
        "rice_blast_infection_prob": rice_blast_prob,
        "bacterial_leaf_blight_infection_prob": blb_prob,
        "sheath_blight_infection_prob": shb_prob,
        "brown_spot_infection_prob": bs_prob,
        "tungro_risk": tungro,
    }


# ==================== FULL PREPROCESSING PIPELINE ====================

def _preprocess_full_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all V2 feature engineering to a DataFrame."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    
    # Rolling averages
    df["temp_rolling_avg_7d"] = df["temp"].rolling(7, min_periods=1).mean()
    df["humidity_rolling_avg_7d"] = df["humidity"].rolling(7, min_periods=1).mean()
    df["precip_rolling_avg_7d"] = df["precip"].rolling(7, min_periods=1).mean()
    df["windspeed_rolling_avg_7d"] = df["windspeed"].rolling(7, min_periods=1).mean()
    
    # Growth stage
    growth_info = df["datetime"].apply(estimate_growth_state)
    df["growth_stage"] = growth_info.apply(lambda x: x[0])
    df["days_since_planting"] = growth_info.apply(lambda x: x[1])
    
    # Water level
    water_level = np.zeros(n)
    alpha, et_coeff, max_water = 0.82, 0.045, 20.0
    for i in range(1, n):
        rain_cm = df["precip"].iloc[i] / 10.0
        et_cm = df["temp"].iloc[i] * et_coeff / 10.0
        water_level[i] = max(0, min(water_level[i-1] * alpha + rain_cm - et_cm, max_water))
    df["water_level_est"] = water_level
    
    # Anomaly detection
    anomaly_data = df.apply(
        lambda r: detect_anomaly_simple(r["temp"], r["temp_rolling_avg_7d"], r["precip"], r["datetime"].month),
        axis=1
    )
    df["anomaly_score_raw"] = anomaly_data.apply(lambda x: x[0])
    df["anomaly_flag"] = anomaly_data.apply(lambda x: x[1])
    
    # VPD and LWD
    df["vpd"] = df.apply(lambda r: calculate_vpd(r["temp"], r["humidity"]), axis=1)
    df["vpd_index"] = df["vpd"].apply(calculate_vpd_index)
    df["lwd"] = df.apply(lambda r: calculate_lwd(r["humidity"], r["precip"], r["growth_stage"]), axis=1)
    
    # Nitrogen
    df["n_applied"] = 100.0
    df["nsf"] = df["n_applied"].apply(lambda x: calculate_nsf(x))
    
    # Humidity features
    df["humidity_norm"] = df["humidity"] / 100.0
    df["rhsi"] = df["humidity"].apply(calculate_rhsi)
    df["rhsi_rolling_7d"] = df["rhsi"].rolling(7, min_periods=1).mean()
    
    # Rainfall features
    df["rainfall_event"] = (df["precip"] >= 0.5).astype(int)
    df["rainfall_events_7d"] = df["rainfall_event"].rolling(7, min_periods=1).sum()
    df["cpi_7d"] = df["precip"].rolling(7, min_periods=1).sum()
    df["cpi_14d"] = df["precip"].rolling(14, min_periods=1).sum()
    
    # GDD calculations
    for pest_key, pest_params in THERMAL_PARAMS.items():
        gdd_daily, gdd_cumulative, current_gdd, last_stage = [], [], 0, "fallow"
        for i in range(n):
            stage = df["growth_stage"].iloc[i]
            if stage == "vegetative" and last_stage == "fallow":
                current_gdd = 0
            daily_gdd = calculate_gdd(df["temp"].iloc[i], pest_key) if stage != "fallow" else 0
            current_gdd = min(current_gdd + daily_gdd, pest_params["gdd_per_gen"] * 3)
            gdd_daily.append(daily_gdd)
            gdd_cumulative.append(current_gdd)
            last_stage = stage
        df[f"{pest_key}_gdd_daily"] = gdd_daily
        df[f"{pest_key}_gdd_cumulative"] = gdd_cumulative
        df[f"{pest_key}_gen_progress"] = [min(g / pest_params["gdd_per_gen"] * 100, 300) for g in gdd_cumulative]
    
    # Static features
    df["stage_suscept"] = df["growth_stage"].map({"vegetative": 0.7, "reproductive": 1.0, "maturity": 0.6, "fallow": 0.1}).fillna(0.5)
    df["field_age_factor"] = df["days_since_planting"].apply(lambda x: min(x / 30, 1.0))
    df["plant_density"] = 35.0
    df["soil_moisture_pct"] = df["humidity"].apply(lambda x: min(x * 0.9, 100))
    df["soil_k_mg_kg"] = 100.0
    
    # Disease probabilities
    disease_probs = df.apply(calculate_disease_infection_probs, axis=1)
    for col in ["rice_blast_infection_prob", "bacterial_leaf_blight_infection_prob", 
                "sheath_blight_infection_prob", "brown_spot_infection_prob", "tungro_risk"]:
        df[col] = disease_probs.apply(lambda x: x[col])
    
    # Population states
    for pest in ["bph", "ysb", "rlf", "wst", "rat", "snail"]:
        df[f"{pest}_pop_state"] = 0.15
    
    # Yield loss
    df["estimated_yield_loss_pct"] = (
        df["rice_blast_infection_prob"] * 25 +
        df["bacterial_leaf_blight_infection_prob"] * 20 +
        df["sheath_blight_infection_prob"] * 15 +
        0.15 * 30
    ).clip(0, 100)
    
    # Datetime features
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    
    # Cyclic encodings
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["winddir_sin"] = np.sin(2 * np.pi * df["winddir"] / 360)
    df["winddir_cos"] = np.cos(2 * np.pi * df["winddir"] / 360)
    df["moonphase_sin"] = np.sin(2 * np.pi * df["moonphase"] / 1)
    df["moonphase_cos"] = np.cos(2 * np.pi * df["moonphase"] / 1)
    
    # Precip type
    df["precip_is_rain"] = (df["precip"] > 0).astype(int)
    df["precip_is_snow"] = 0
    df["precip_is_none"] = (df["precip"] == 0).astype(int)
    
    return df


def preprocess_weather_to_sequence(
    weather_data: list, 
    location_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform raw weather data into a single (1, 14, 87) sequence."""
    scaler_X = get_scaler_X()
    scaler_A = get_scaler_A()
    meta = get_feature_meta()
    feature_cols = meta["feature_cols"]
    
    df = _preprocess_full_dataframe(pd.DataFrame(weather_data))
    df_last14 = df.iloc[-SEQUENCE_LENGTH:].copy()
    
    X_scaled = scaler_X.transform(df_last14[feature_cols].fillna(0).values)
    X_seq = X_scaled.reshape(1, SEQUENCE_LENGTH, -1).astype(np.float32)
    
    A_scaled = scaler_A.transform(df_last14[["anomaly_score_raw"]].values)
    A_seq = A_scaled.reshape(1, SEQUENCE_LENGTH, 1).astype(np.float32)
    
    return X_seq, A_seq


def preprocess_forecast_windows(
    weather_data: list,
    location_id: int,
    n_forecast_days: int = 14
) -> Tuple[np.ndarray, np.ndarray, List[date]]:
    """
    Preprocess multiple 14-day windows for multi-day forecasting.
    
    Returns:
        X_seq: (n_forecast_days, 14, 87) - batched feature sequences
        A_seq: (n_forecast_days, 14, 1) - batched anomaly sequences
        dates: List of prediction dates
    """
    scaler_X = get_scaler_X()
    scaler_A = get_scaler_A()
    meta = get_feature_meta()
    feature_cols = meta["feature_cols"]
    
    # Preprocess entire dataset once
    df = _preprocess_full_dataframe(pd.DataFrame(weather_data))
    
    X_list, A_list, date_list = [], [], []
    
    # For Day N (1-indexed), use window [N-1 : N+12]
    # Day 1: indices [0:14], Day 14: indices [13:27]
    for day_num in range(1, n_forecast_days + 1):
        start_idx = day_num - 1
        end_idx = start_idx + SEQUENCE_LENGTH
        
        if end_idx > len(df):
            break  # Not enough data
        
        window = df.iloc[start_idx:end_idx].copy()
        X_scaled = scaler_X.transform(window[feature_cols].fillna(0).values)
        A_scaled = scaler_A.transform(window[["anomaly_score_raw"]].values)
        
        X_list.append(X_scaled)
        A_list.append(A_scaled)
        date_list.append(window.iloc[-1]["datetime"].date())
    
    X_seq = np.array(X_list, dtype=np.float32)
    A_seq = np.array(A_list, dtype=np.float32)
    
    return X_seq, A_seq, date_list

# ==================== DETERMINISTIC DISEASE RISK ENGINE ====================

DISEASE_META = {
    "rice_blast": {
        "common_name": "Rice Blast (Bercak Daun)",
        "pathogen": "Pyricularia oryzae",
        # Optimal: Temp 18-28°C, high LWD, high N
    },
    "bacterial_leaf_blight": {
        "common_name": "Bacterial Leaf Blight (Hawar Daun Bakteri)",
        "pathogen": "Xanthomonas oryzae pv. oryzae",
        # Optimal: Temp 25-30°C, frequent rain/wind
    },
    "sheath_blight": {
        "common_name": "Sheath Blight (Busuk Batang)",
        "pathogen": "Rhizoctonia solani",
        # Optimal: High humidity, high soil moisture, dense canopy
    },
    "brown_spot": {
        "common_name": "Brown Spot (Bercak Coklat)",
        "pathogen": "Helminthosporium oryzae",
        # Optimal: Temp 25-28°C, stressed plants (low K, old leaves)
    },
    "tungro": {
        "common_name": "Tungro Virus",
        "pathogen": "RTBV + RTSV (via Green Leafhopper)",
        # Optimal: High GLH vector population, vegetative stage
    }
}


def get_disease_risks_for_day(row: pd.Series) -> dict:
    """
    Calculate deterministic disease risk percentages for a single day.
    Uses environmental favorability indices based on V2 biological thresholds.
    Returns a dict formatted identically to the LSTM pest outputs.
    """
    # Extract base variables (with safe defaults)
    lwd = row.get("lwd", 0)
    temp = row.get("temp", 28.0)
    humidity = row.get("humidity", 80.0)
    humidity_norm = row.get("humidity_norm", 0.8)
    rainfall_events_7d = row.get("rainfall_events_7d", 0)
    soil_moisture = row.get("soil_moisture_pct", 70.0)
    field_age = row.get("field_age_factor", 0.5)
    bph_gen = row.get("bph_gen_progress", 0) # Proxy for GLH vector
    growth_stage = row.get("growth_stage", "fallow")
    vpd_index = row.get("vpd_index", 0.5)
    
    # 1. Rice Blast: Driven by Leaf Wetness + Temperature + N-status
    temp_suit_blast = max(0, 1 - abs(temp - 23) / 10) # Optimal 23°C
    lwd_factor = min(lwd / 12, 1.0) # Needs >10-12h wetness
    blast_daily = temp_suit_blast * 0.4 + lwd_factor * 0.4 + humidity_norm * 0.2
    # Scale to 0-99 range (Blast rarely hits 99% in 14-day window)
    blast_risk = round(min(blast_daily * 85, 99), 2)
    if growth_stage == "fallow": blast_risk = round(blast_risk * 0.1, 2)

    # 2. Bacterial Leaf Blight: Driven by Rain events + Wind/Humidity + Temp
    temp_suit_blb = max(0, 1 - abs(temp - 27.5) / 8) # Optimal 27.5°C
    rain_factor = min(rainfall_events_7d / 5, 1.0) # Needs frequent rain to splash bacteria
    blb_daily = rain_factor * 0.5 + temp_suit_blb * 0.3 + humidity_norm * 0.2
    blb_risk = round(min(blb_daily * 80, 99), 2)
    if growth_stage == "fallow": blb_risk = round(blb_risk * 0.1, 2)

    # 3. Sheath Blight: Driven by Soil Moisture + Canopy Humidity
    soil_factor = min(soil_moisture / 90, 1.0) # Thrives in waterlogged soils
    shb_daily = soil_factor * 0.4 + humidity_norm * 0.4 + vpd_index * 0.2
    shb_risk = round(min(shb_daily * 75, 99), 2)
    if growth_stage == "fallow": shb_risk = round(shb_risk * 0.1, 2)

    # 4. Brown Spot: Driven by Plant Age (stress) + Humidity + Temp
    temp_suit_bs = max(0, 1 - abs(temp - 26.5) / 8) # Optimal 26.5°C
    bs_daily = field_age * 0.4 + temp_suit_bs * 0.3 + humidity_norm * 0.3
    bs_risk = round(min(bs_daily * 70, 99), 2)
    if growth_stage == "fallow": bs_risk = round(bs_risk * 0.1, 2)

    # 5. Tungro: Driven by Vector (GLH) population + Growth stage
    # BPH gen progress is our proxy for Green Leafhopper activity
    vector_suit = min(bph_gen / 100, 1.0) 
    # Tungro only spreads in early vegetative stage
    if growth_stage == "vegetative": stage_mult = 1.0
    elif growth_stage == "reproductive": stage_mult = 0.3
    else: stage_mult = 0.05
        
    tungro_risk = round(min(vector_suit * stage_mult * 90, 99), 2)

    # Format to match pest output exactly
    risks = {}
    for key, meta in DISEASE_META.items():
        val = {
            "rice_blast": blast_risk,
            "bacterial_leaf_blight": blb_risk,
            "sheath_blight": shb_risk,
            "brown_spot": bs_risk,
            "tungro": tungro_risk
        }.get(key, 0.0)
        
        risks[key] = {
            "risk_percentage": val,
            "risk_level": get_risk_level(val), # Re-use the main.py helper
            "common_name": meta["common_name"],
            "scientific_name": meta["pathogen"]
        }
        
    return risks


# We need to import the risk level helper here so it's self-contained
def get_risk_level(risk: float) -> str:
    if risk >= 70: return "HIGH"
    elif risk >= 40: return "MODERATE"
    elif risk >= 20: return "LOW"
    return "MINIMAL"