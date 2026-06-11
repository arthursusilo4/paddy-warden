# ============================================================
# model_utils.py — Model & Scaler Loading (Version-Patched)
# ============================================================

import os
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Global variables (loaded at startup)
_model = None
_scaler_X = None
_scaler_y = None
_scaler_A = None
_feature_meta = None


class AnomalyEnhancedAttention(layers.Layer):
    """
    Custom attention layer with anomaly score modulation.
    Must match the exact class definition used during training.
    """
    
    def __init__(self, units, **kwargs):
        super(AnomalyEnhancedAttention, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        lstm_dim = input_shape[0][-1]
        self.Wa = self.add_weight(
            name="Wa", 
            shape=(lstm_dim, self.units), 
            initializer="glorot_uniform", 
            trainable=True
        )
        self.ba = self.add_weight(
            name="ba", 
            shape=(self.units,), 
            initializer="zeros", 
            trainable=True
        )
        self.Va = self.add_weight(
            name="Va", 
            shape=(self.units, 1), 
            initializer="glorot_uniform", 
            trainable=True
        )
        self.lambda_anomaly = self.add_weight(
            name="lambda_anomaly", 
            shape=(1,), 
            initializer="glorot_uniform", 
            trainable=True, 
            constraint=tf.keras.constraints.NonNeg()
        )
        super(AnomalyEnhancedAttention, self).build(input_shape)
    
    def call(self, inputs):
        lstm_output, anomaly_scores = inputs
        score = tf.nn.tanh(
            tf.tensordot(lstm_output, self.Wa, axes=[[2], [0]]) + self.ba
        )
        score = tf.tensordot(score, self.Va, axes=[[2], [0]])
        anomaly_multiplier = 1.0 + self.lambda_anomaly * anomaly_scores
        score = score * anomaly_multiplier
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(lstm_output * attention_weights, axis=1)
        attention_weights = tf.squeeze(attention_weights, axis=-1)
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def weighted_mse(y_true, y_pred):
    """Custom weighted MSE loss (simplified for inference loading)."""
    return tf.reduce_mean(tf.square(y_true - y_pred))


def _strip_quantization_config(config_dict):
    """
    Recursively remove 'quantization_config' keys from model config.
    This fixes version mismatch between Colab and VPS Keras versions.
    """
    if isinstance(config_dict, dict):
        return {k: _strip_quantization_config(v) for k, v in config_dict.items() if k != 'quantization_config'}
    elif isinstance(config_dict, list):
        return [_strip_quantization_config(item) for item in config_dict]
    else:
        return config_dict


def _load_model_patched(model_path: str):
    """
    Load model by patching the config to remove quantization_config.
    Handles Keras 3.x serialization inconsistencies.
    """
    import zipfile
    import tempfile
    import shutil
    
    # Create temp directory for patched model
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, 'patched_model.keras')
    
    try:
        # Extract the .keras file (it's a zip archive)
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Read and patch the config.json
        config_path = os.path.join(temp_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Recursively strip quantization_config
        clean_config = _strip_quantization_config(config)
        
        # Write patched config back
        with open(config_path, 'w') as f:
            json.dump(clean_config, f)
        
        # Re-zip the patched model
        with zipfile.ZipFile(temp_model_path, 'w', zipfile.ZIP_DEFLATED) as new_zip:
            for item in os.listdir(temp_dir):
                if item == 'patched_model.keras':
                    continue
                item_path = os.path.join(temp_dir, item)
                if os.path.isfile(item_path):
                    new_zip.write(item_path, item)
                elif os.path.isdir(item_path):
                    for root, dirs, files in os.walk(item_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_dir)
                            new_zip.write(file_path, arcname)
        
        # Load the patched model
        model = keras.models.load_model(
            temp_model_path,
            custom_objects={
                "AnomalyEnhancedAttention": AnomalyEnhancedAttention,
                "weighted_mse": weighted_mse
            }
        )
        
        return model
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_model_and_scalers(base_path: str):
    """Load model, scalers, and metadata at startup."""
    global _model, _scaler_X, _scaler_y, _scaler_A, _feature_meta
    
    model_path = os.path.join(base_path, "model", "best_model_v2.keras")
    scaler_dir = os.path.join(base_path, "scalers")
    
    print(f"Loading model from {model_path}...")
    
    # Always use patched loading to handle quantization_config issue
    _model = _load_model_patched(model_path)
    print("  Model loaded successfully.")
    
    print("Loading scalers...")
    _scaler_X = joblib.load(os.path.join(scaler_dir, "scaler_X.pkl"))
    _scaler_y = joblib.load(os.path.join(scaler_dir, "scaler_y.pkl"))
    
    # Try to load scaler_A, fallback to identity scaling if missing
    scaler_a_path = os.path.join(scaler_dir, "scaler_A.pkl")
    if os.path.exists(scaler_a_path):
        _scaler_A = joblib.load(scaler_a_path)
        print("  scaler_A loaded from file.")
    else:
        print("  WARNING: scaler_A.pkl not found. Using fallback scaling.")
        from sklearn.preprocessing import MinMaxScaler
        _scaler_A = MinMaxScaler(feature_range=(0, 1))
        _scaler_A.scale_ = np.array([1.0])
        _scaler_A.min_ = np.array([0.0])
    
    print("Loading metadata...")
    with open(os.path.join(scaler_dir, "feature_meta.json")) as f:
        _feature_meta = json.load(f)
    
    print(f"  Features: {_feature_meta['n_features']}, Targets: {_feature_meta['n_targets']}")
    print(f"  Locations: {_feature_meta['n_locations']}")


def get_model():
    return _model

def get_scaler_X():
    return _scaler_X

def get_scaler_y():
    return _scaler_y

def get_scaler_A():
    return _scaler_A

def get_feature_meta():
    return _feature_meta