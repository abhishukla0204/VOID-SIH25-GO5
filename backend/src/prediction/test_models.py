"""
Test Prediction Model Loading and Inference
==========================================

This script demonstrates how to load the trained prediction models
and make predictions on new data.
"""

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any

def load_prediction_models(model_dir: str = "outputs/models"):
    """Load all trained prediction models"""
    
    print("🔄 Loading trained prediction models...")
    
    models = {}
    scalers = {}
    
    try:
        # Load traditional ML models
        models['xgboost'] = joblib.load(f"{model_dir}/xgboost_model.joblib")
        models['random_forest'] = joblib.load(f"{model_dir}/random_forest_model.joblib")
        print("✅ XGBoost and Random Forest models loaded")
        
        # Load scaler
        scalers['main'] = joblib.load(f"{model_dir}/main_scaler.joblib")
        print("✅ Data scaler loaded")
        
        # Load metadata
        metadata = joblib.load(f"{model_dir}/model_metadata.joblib")
        feature_names = metadata['feature_names']
        model_performance = metadata['model_performance']
        print("✅ Model metadata loaded")
        
        # Load Neural Network (requires model architecture)
        try:
            # Define the same architecture as in training
            class RockfallNN(nn.Module):
                def __init__(self, input_size, hidden_size=64, dropout_rate=0.3):
                    super(RockfallNN, self).__init__()
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.fc2 = nn.Linear(hidden_size, hidden_size//2)
                    self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
                    self.fc4 = nn.Linear(hidden_size//4, 1)
                    
                    self.dropout = nn.Dropout(dropout_rate)
                    self.relu = nn.ReLU()
                    self.output_activation = nn.Sigmoid()
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = self.relu(self.fc3(x))
                    x = self.dropout(x)
                    x = self.fc4(x)
                    x = self.output_activation(x)
                    return x
            
            # Initialize and load neural network
            input_size = len(feature_names)
            nn_model = RockfallNN(input_size)
            nn_model.load_state_dict(torch.load(f"{model_dir}/neural_network_model.pth"))
            nn_model.eval()
            models['neural_network'] = nn_model
            print("✅ Neural Network model loaded")
            
        except Exception as e:
            print(f"⚠️ Could not load Neural Network: {e}")
        
        return models, scalers, feature_names, model_performance
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None, None

def predict_rockfall_risk(models: Dict, scalers: Dict, feature_names: list, 
                         input_data: np.ndarray) -> Dict[str, float]:
    """Make predictions using all loaded models"""
    
    # Scale the input data
    input_scaled = scalers['main'].transform(input_data)
    
    predictions = {}
    
    # XGBoost prediction
    if 'xgboost' in models:
        xgb_pred = models['xgboost'].predict_proba(input_data)[:, 1]
        predictions['xgboost'] = xgb_pred[0]
    
    # Random Forest prediction
    if 'random_forest' in models:
        rf_pred = models['random_forest'].predict_proba(input_data)[:, 1]
        predictions['random_forest'] = rf_pred[0]
    
    # Neural Network prediction
    if 'neural_network' in models:
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_scaled)
            nn_pred = models['neural_network'](input_tensor).numpy()[0][0]
            predictions['neural_network'] = nn_pred
    
    # Ensemble prediction (weighted average)
    if len(predictions) > 1:
        # Simple average for demonstration
        ensemble_pred = sum(predictions.values()) / len(predictions)
        predictions['ensemble'] = ensemble_pred
    
    return predictions

def create_sample_data(feature_names: list) -> np.ndarray:
    """Create sample input data for testing"""
    
    # Create realistic sample data
    sample_data = {
        'slope_degrees': 45.0,  # Steep slope
        'elevation_m': 1500.0,  # Medium elevation
        'fracture_density': 3.5,  # High fracture density
        'roughness_index': 0.7,  # Rough surface
        'rainfall_mm': 50.0,  # Heavy rainfall
        'temperature_c': 5.0,   # Cool temperature
        'wind_speed_kmh': 30.0,  # Moderate wind
        'seismic_magnitude': 2.0,  # Minor earthquake
        'freeze_thaw_cycles': 5,  # Multiple freeze-thaw cycles
        'aspect_cos': 0.5,  # South-facing slope
        'aspect_sin': 0.866,
        'geological_strength': 0.3,  # Weak rock
        'vegetation_cover': 0.2,  # Low vegetation
        'water_content': 0.8,  # High water content
        'distance_to_fault': 500.0,  # Close to fault
        'previous_events': 2,  # Previous rockfall events
        'instability_index': 0.8,  # High instability
        'weathering_rate': 0.6,  # High weathering
        'human_activity': 0.3   # Some human activity
    }
    
    # Convert to array in the correct order
    data_array = np.array([[sample_data.get(name, 0.0) for name in feature_names]])
    
    return data_array

def main():
    """Test the trained prediction models"""
    
    print("🧪 Testing Rockfall Prediction Models")
    print("="*50)
    
    # Load models
    models, scalers, feature_names, performance = load_prediction_models()
    
    if models is None:
        print("❌ Failed to load models")
        return
    
    print(f"\n📊 Loaded models: {list(models.keys())}")
    print(f"📈 Feature count: {len(feature_names)}")
    
    # Show model performance
    print(f"\n📈 Model Performance Summary:")
    for model_name, perf in performance.items():
        if 'auc_score' in perf:
            print(f"   {model_name}: AUC = {perf['auc_score']:.3f}, Accuracy = {perf['accuracy']:.3f}")
    
    # Create sample data
    print(f"\n🔮 Making predictions on sample data...")
    sample_data = create_sample_data(feature_names)
    
    # Display sample input
    print(f"\n📋 Sample Input Features:")
    for i, name in enumerate(feature_names):
        print(f"   {name}: {sample_data[0][i]:.2f}")
    
    # Make predictions
    predictions = predict_rockfall_risk(models, scalers, feature_names, sample_data)
    
    # Display results
    print(f"\n🎯 Rockfall Risk Predictions:")
    for model_name, risk_score in predictions.items():
        risk_level = "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.3 else "LOW"
        print(f"   {model_name}: {risk_score:.3f} ({risk_level} risk)")
    
    # Risk interpretation
    if 'ensemble' in predictions:
        final_risk = predictions['ensemble']
        if final_risk > 0.7:
            interpretation = "🔴 HIGH RISK - Immediate attention required"
        elif final_risk > 0.3:
            interpretation = "🟡 MEDIUM RISK - Monitor closely"
        else:
            interpretation = "🟢 LOW RISK - Normal monitoring"
        
        print(f"\n🚨 Final Assessment: {interpretation}")
        print(f"   Risk Score: {final_risk:.3f}")
    
    print(f"\n✅ Model testing completed successfully!")
    print(f"\n💡 Usage Instructions:")
    print(f"   1. Load models using load_prediction_models()")
    print(f"   2. Prepare input data with {len(feature_names)} features")
    print(f"   3. Call predict_rockfall_risk() to get risk scores")
    print(f"   4. Use ensemble prediction for best results")

if __name__ == "__main__":
    main()