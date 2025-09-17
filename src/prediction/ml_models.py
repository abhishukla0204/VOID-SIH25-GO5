"""
Machine Learning Models for Rockfall Risk Prediction
==================================================

This module implements multiple ML approaches for predicting rockfall risk
based on terrain features and environmental conditions.

Models Implemented:
- XGBoost: Gradient boosting for tabular data
- Random Forest: Ensemble method with feature importance
- Neural Network: Deep learning approach
- LSTM: Time-series prediction for temporal patterns
- Ensemble: Combined model predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, confusion_matrix, 
                           mean_squared_error, r2_score, roc_auc_score, 
                           precision_recall_curve, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb

# Deep Learning Libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Neural Network models will be disabled.")

class RockfallRiskPredictor:
    """Main class for rockfall risk prediction using multiple ML approaches"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.feature_names = []
        
        # Set random seeds
        np.random.seed(random_state)
        
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'rockfall_event',
                    test_size: float = 0.2) -> Dict[str, Any]:
        """Prepare data for training and testing"""
        
        print(f"🔄 Preparing data for ML training...")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Select feature columns (exclude target and non-feature columns)
        exclude_columns = [target_column, 'timestamp', 'risk_category', 'x', 'y']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        print(f"📈 Features: {len(self.feature_names)}")
        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y if len(y.unique()) > 1 else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'feature_names': self.feature_names
        }
        
        print(f"✅ Data prepared successfully!")
        print(f"   Training set: {X_train.shape[0]:,} samples")
        print(f"   Test set: {X_test.shape[0]:,} samples")
        
        return data_dict
    
    def train_xgboost(self, data_dict: Dict[str, Any], task_type: str = 'classification') -> Dict:
        """Train XGBoost model for rockfall prediction"""
        
        print(f"🚀 Training XGBoost model ({task_type})...")
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        if task_type == 'classification':
            # Classification parameters
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state
            }
            
            model = xgb.XGBClassifier(**params)
            
        else:  # regression
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state
            }
            
            model = xgb.XGBRegressor(**params)
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            accuracy = (y_pred == y_test).mean()
            
            performance = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   AUC Score: {auc_score:.3f}")
            
        else:
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            performance = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2
            }
            
            print(f"   RMSE: {np.sqrt(mse):.3f}")
            print(f"   R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        self.feature_importance['xgboost'] = feature_importance
        
        # Store model and performance
        self.models['xgboost'] = model
        self.model_performance['xgboost'] = performance
        
        print(f"✅ XGBoost training completed!")
        
        return {
            'model': model,
            'performance': performance,
            'feature_importance': feature_importance,
            'predictions': y_pred
        }
    
    def train_random_forest(self, data_dict: Dict[str, Any], task_type: str = 'classification') -> Dict:
        """Train Random Forest model for rockfall prediction"""
        
        print(f"🌲 Training Random Forest model ({task_type})...")
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        if task_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            accuracy = (y_pred == y_test).mean()
            
            performance = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   AUC Score: {auc_score:.3f}")
            
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            performance = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2
            }
            
            print(f"   RMSE: {np.sqrt(mse):.3f}")
            print(f"   R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        self.feature_importance['random_forest'] = feature_importance
        
        # Store model and performance
        self.models['random_forest'] = model
        self.model_performance['random_forest'] = performance
        
        print(f"✅ Random Forest training completed!")
        
        return {
            'model': model,
            'performance': performance,
            'feature_importance': feature_importance,
            'predictions': y_pred
        }
    
    def train_neural_network(self, data_dict: Dict[str, Any], task_type: str = 'classification') -> Dict:
        """Train Neural Network model using PyTorch"""
        
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available. Skipping Neural Network training.")
            return {}
        
        print(f"🧠 Training Neural Network model ({task_type})...")
        
        X_train_scaled = data_dict['X_train_scaled']
        X_test_scaled = data_dict['X_test_scaled']
        y_train = data_dict['y_train'].values
        y_test = data_dict['y_test'].values
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Define neural network architecture
        class RockfallNN(nn.Module):
            def __init__(self, input_size, hidden_size=64, dropout_rate=0.3):
                super(RockfallNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size//2)
                self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
                self.fc4 = nn.Linear(hidden_size//4, 1)
                
                self.dropout = nn.Dropout(dropout_rate)
                self.relu = nn.ReLU()
                
                if task_type == 'classification':
                    self.output_activation = nn.Sigmoid()
                else:
                    self.output_activation = nn.Identity()
            
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
        
        # Initialize model
        input_size = X_train_scaled.shape[1]
        model = RockfallNN(input_size)
        
        # Loss function and optimizer
        if task_type == 'classification':
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Training loop
        model.train()
        epochs = 100
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_test_tensor)
            y_pred = y_pred_tensor.numpy().flatten()
        
        if task_type == 'classification':
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = (y_pred_binary == y_test).mean()
            auc_score = roc_auc_score(y_test, y_pred)
            
            performance = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'classification_report': classification_report(y_test, y_pred_binary, output_dict=True)
            }
            
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   AUC Score: {auc_score:.3f}")
            
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            performance = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2,
                'train_losses': train_losses
            }
            
            print(f"   RMSE: {np.sqrt(mse):.3f}")
            print(f"   R² Score: {r2:.3f}")
        
        # Store model and performance
        self.models['neural_network'] = model
        self.model_performance['neural_network'] = performance
        
        print(f"✅ Neural Network training completed!")
        
        return {
            'model': model,
            'performance': performance,
            'predictions': y_pred
        }
    
    def create_ensemble_model(self, data_dict: Dict[str, Any], models_to_combine: List[str] = None) -> Dict:
        """Create ensemble model combining multiple trained models"""
        
        if models_to_combine is None:
            models_to_combine = list(self.models.keys())
        
        print(f"🎯 Creating ensemble model with: {models_to_combine}")
        
        X_test = data_dict['X_test']
        X_test_scaled = data_dict['X_test_scaled']
        y_test = data_dict['y_test']
        
        predictions = []
        weights = []
        
        for model_name in models_to_combine:
            if model_name not in self.models:
                continue
                
            model = self.models[model_name]
            
            # Get predictions based on model type
            if model_name == 'neural_network' and TORCH_AVAILABLE:
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test_scaled)
                    pred = model(X_tensor).numpy().flatten()
            else:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_test)[:, 1]
                else:
                    pred = model.predict(X_test)
            
            predictions.append(pred)
            
            # Weight based on model performance (AUC or R2)
            if model_name in self.model_performance:
                perf = self.model_performance[model_name]
                weight = perf.get('auc_score', perf.get('r2_score', 0.5))
                weights.append(max(weight, 0.1))  # Minimum weight of 0.1
            else:
                weights.append(0.5)
        
        if not predictions:
            print("❌ No valid model predictions found for ensemble")
            return {}
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Create weighted ensemble prediction
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        # Calculate ensemble performance
        if len(np.unique(y_test)) == 2:  # Classification
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            accuracy = (ensemble_pred_binary == y_test).mean()
            auc_score = roc_auc_score(y_test, ensemble_pred)
            
            performance = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'model_weights': dict(zip(models_to_combine, weights)),
                'classification_report': classification_report(y_test, ensemble_pred_binary, output_dict=True)
            }
            
            print(f"   Ensemble Accuracy: {accuracy:.3f}")
            print(f"   Ensemble AUC Score: {auc_score:.3f}")
            
        else:  # Regression
            mse = mean_squared_error(y_test, ensemble_pred)
            r2 = r2_score(y_test, ensemble_pred)
            
            performance = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2,
                'model_weights': dict(zip(models_to_combine, weights))
            }
            
            print(f"   Ensemble RMSE: {np.sqrt(mse):.3f}")
            print(f"   Ensemble R² Score: {r2:.3f}")
        
        # Store ensemble performance
        self.model_performance['ensemble'] = performance
        
        print(f"✅ Ensemble model created successfully!")
        print(f"   Model weights: {dict(zip(models_to_combine, weights))}")
        
        return {
            'predictions': ensemble_pred,
            'performance': performance,
            'model_weights': dict(zip(models_to_combine, weights))
        }
    
    def analyze_feature_importance(self) -> Dict:
        """Analyze and visualize feature importance across models"""
        
        if not self.feature_importance:
            print("❌ No feature importance data found. Train models first.")
            return {}
        
        print("📊 Analyzing feature importance across models...")
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame(self.feature_importance).fillna(0)
        
        # Calculate average importance across models
        importance_df['average'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('average', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot top 15 features
        top_features = importance_df.head(15)
        
        x = np.arange(len(top_features))
        width = 0.25
        
        models = [col for col in top_features.columns if col != 'average']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
        
        for i, model in enumerate(models):
            if i < len(colors):
                plt.bar(x + i * width, top_features[model], width, 
                       label=model, color=colors[i], alpha=0.8)
        
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance Comparison Across Models')
        plt.xticks(x + width, top_features.index, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Feature importance summary
        summary = {
            'top_10_features': top_features.head(10)['average'].to_dict(),
            'model_agreement': {},
            'feature_rankings': {}
        }
        
        # Calculate model agreement on top features
        for model in models:
            model_top_5 = importance_df.nlargest(5, model).index.tolist()
            overall_top_5 = importance_df.nlargest(5, 'average').index.tolist()
            
            agreement = len(set(model_top_5) & set(overall_top_5)) / 5
            summary['model_agreement'][model] = agreement
        
        # Feature rankings by model
        for model in models:
            rankings = importance_df[model].rank(ascending=False).to_dict()
            summary['feature_rankings'][model] = rankings
        
        print(f"✅ Feature importance analysis completed!")
        
        return summary
    
    def save_models(self, save_dir: str = "outputs/models"):
        """Save all trained models and scalers"""
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save traditional ML models
        for model_name, model in self.models.items():
            if model_name != 'neural_network':
                model_path = os.path.join(save_dir, f"{model_name}_model.joblib")
                joblib.dump(model, model_path)
                print(f"💾 {model_name} model saved to: {model_path}")
        
        # Save PyTorch model separately
        if 'neural_network' in self.models and TORCH_AVAILABLE:
            nn_path = os.path.join(save_dir, "neural_network_model.pth")
            torch.save(self.models['neural_network'].state_dict(), nn_path)
            print(f"💾 Neural network model saved to: {nn_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = os.path.join(save_dir, f"{scaler_name}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
            print(f"💾 {scaler_name} scaler saved to: {scaler_path}")
        
        # Save feature names and performance metrics
        metadata = {
            'feature_names': self.feature_names,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance
        }
        
        metadata_path = os.path.join(save_dir, "model_metadata.joblib")
        joblib.dump(metadata, metadata_path)
        print(f"💾 Model metadata saved to: {metadata_path}")
    
    def predict_risk(self, X: np.ndarray, model_name: str = 'ensemble') -> np.ndarray:
        """Make risk predictions using specified model"""
        
        if model_name == 'ensemble':
            # Use all available models for ensemble prediction
            if len(self.models) == 0:
                raise ValueError("No trained models found")
            
            predictions = []
            weights = []
            
            for name, model in self.models.items():
                if name == 'neural_network' and TORCH_AVAILABLE:
                    model.eval()
                    with torch.no_grad():
                        X_scaled = self.scalers['main'].transform(X)
                        X_tensor = torch.FloatTensor(X_scaled)
                        pred = model(X_tensor).numpy().flatten()
                else:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]
                    else:
                        pred = model.predict(X)
                
                predictions.append(pred)
                
                # Get model weight from performance
                perf = self.model_performance.get(name, {})
                weight = perf.get('auc_score', perf.get('r2_score', 0.5))
                weights.append(max(weight, 0.1))
            
            # Normalize weights and create ensemble
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            return ensemble_pred
            
        else:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")
            
            model = self.models[model_name]
            
            if model_name == 'neural_network' and TORCH_AVAILABLE:
                model.eval()
                with torch.no_grad():
                    X_scaled = self.scalers['main'].transform(X)
                    X_tensor = torch.FloatTensor(X_scaled)
                    predictions = model(X_tensor).numpy().flatten()
            else:
                if hasattr(model, 'predict_proba'):
                    predictions = model.predict_proba(X)[:, 1]
                else:
                    predictions = model.predict(X)
            
            return predictions


def main():
    """Demonstrate ML model training for rockfall prediction"""
    
    # Initialize predictor
    predictor = RockfallRiskPredictor(random_state=42)
    
    print("🚀 Starting Rockfall Risk Prediction Model Training...")
    
    # For demonstration, we'll create a small synthetic dataset
    # In practice, you would load your actual data
    from synthetic_data_generator import SyntheticDataGenerator, DataGenerationConfig
    
    # Generate synthetic data
    config = DataGenerationConfig(n_samples=2000, random_seed=42)
    generator = SyntheticDataGenerator(config)
    df = generator.generate_complete_dataset()
    
    # Prepare data
    data_dict = predictor.prepare_data(df, target_column='rockfall_event')
    
    # Train multiple models
    print("\n" + "="*50)
    print("🎯 TRAINING MULTIPLE ML MODELS")
    print("="*50)
    
    # XGBoost
    xgb_results = predictor.train_xgboost(data_dict, task_type='classification')
    
    # Random Forest
    rf_results = predictor.train_random_forest(data_dict, task_type='classification')
    
    # Neural Network (if PyTorch available)
    nn_results = predictor.train_neural_network(data_dict, task_type='classification')
    
    # Create ensemble
    ensemble_results = predictor.create_ensemble_model(data_dict)
    
    # Analyze feature importance
    print("\n" + "="*50)
    print("📊 FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    importance_analysis = predictor.analyze_feature_importance()
    
    # Save models
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
    
    return predictor, data_dict


if __name__ == "__main__":
    predictor, data = main()