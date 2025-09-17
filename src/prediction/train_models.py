"""
Training Script for Rockfall Prediction Models
==============================================

This script trains and saves all ML models for rockfall prediction.
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_models import RockfallRiskPredictor
from synthetic_data_generator import SyntheticDataGenerator, DataGenerationConfig

def main():
    """Train all prediction models and save them"""
    
    print("🚀 Starting Rockfall Risk Prediction Model Training...")
    
    # Initialize predictor
    predictor = RockfallRiskPredictor(random_state=42)
    
    # Load existing balanced synthetic data
    print("\n📊 Loading existing balanced synthetic training data...")
    try:
        df = pd.read_csv('outputs/synthetic_training_data.csv')
        print(f"✅ Loaded {len(df):,} samples with {len(df.columns)} features from existing dataset")
        
        # Display data balance
        risk_dist = df['risk_category'].value_counts().sort_index()
        print(f"📈 Risk distribution: {dict(risk_dist)}")
        event_rate = df['rockfall_event'].mean() * 100
        print(f"💥 Overall rockfall event rate: {event_rate:.1f}%")
        
    except FileNotFoundError:
        print("❌ No existing dataset found. Generating new balanced dataset...")
        config = DataGenerationConfig(n_samples=5000, random_seed=42)
        generator = SyntheticDataGenerator(config)
        df = generator.generate_complete_dataset()
        print(f"✅ Generated {len(df):,} samples with {len(df.columns)} features")
    
    # Prepare data
    print("\n🔧 Preparing data for training...")
    data_dict = predictor.prepare_data(df, target_column='rockfall_event')
    
    # Train multiple models
    print("\n" + "="*50)
    print("🎯 TRAINING MULTIPLE ML MODELS")
    print("="*50)
    
    # XGBoost
    print("\n1. Training XGBoost...")
    xgb_results = predictor.train_xgboost(data_dict, task_type='classification')
    
    # Random Forest
    print("\n2. Training Random Forest...")
    rf_results = predictor.train_random_forest(data_dict, task_type='classification')
    
    # Neural Network (if PyTorch available)
    print("\n3. Training Neural Network...")
    nn_results = predictor.train_neural_network(data_dict, task_type='classification')
    
    # Create ensemble
    print("\n4. Creating Ensemble Model...")
    ensemble_results = predictor.create_ensemble_model(data_dict)
    
    # Analyze feature importance
    print("\n" + "="*50)
    print("📊 FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    importance_analysis = predictor.analyze_feature_importance()
    
    # Save models
    print("\n" + "="*50)
    print("💾 SAVING MODELS")
    print("="*50)
    
    predictor.save_models()
    
    print("\n" + "="*50)
    print("✅ MODEL TRAINING COMPLETED!")
    print("="*50)
    
    # Print final performance summary
    print("\n📈 FINAL PERFORMANCE SUMMARY:")
    for model_name, performance in predictor.model_performance.items():
        if 'auc_score' in performance:
            print(f"   {model_name}: AUC = {performance['auc_score']:.3f}, "
                  f"Accuracy = {performance['accuracy']:.3f}")
    
    print(f"\n💾 Models saved to: outputs/models/")
    print("   Files created:")
    print("   - xgboost_model.joblib")
    print("   - random_forest_model.joblib") 
    print("   - neural_network_model.pth")
    print("   - main_scaler.joblib")
    print("   - model_metadata.joblib")
    
    return predictor, data_dict

if __name__ == "__main__":
    predictor, data = main()