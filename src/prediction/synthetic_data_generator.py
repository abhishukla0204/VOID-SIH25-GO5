"""
Synthetic Data Generator for Rockfall Prediction
==============================================

This module generates realistic synthetic training data for rockfall prediction
based on terrain features and environmental triggers.

Key Features:
- Critical terrain features (slope, fracture density, instability index, etc.)
- Environmental triggers (rainfall, freeze-thaw cycles, seismic activity, etc.)
- Realistic correlations between features
- Configurable data generation parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation"""
    n_samples: int = 10000
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    random_seed: int = 42
    
    # Terrain feature ranges
    slope_range: Tuple[float, float] = (0, 90)  # degrees
    elevation_range: Tuple[float, float] = (100, 3000)  # meters
    fracture_density_range: Tuple[float, float] = (0, 10)  # fractures per m²
    roughness_range: Tuple[float, float] = (0, 1)  # normalized roughness
    
    # Environmental ranges
    rainfall_range: Tuple[float, float] = (0, 200)  # mm per day
    temperature_range: Tuple[float, float] = (-30, 45)  # Celsius
    wind_speed_range: Tuple[float, float] = (0, 120)  # km/h
    seismic_magnitude_range: Tuple[float, float] = (0, 7)  # Richter scale

class SyntheticDataGenerator:
    """Generates realistic synthetic data for rockfall prediction"""
    
    def __init__(self, config: DataGenerationConfig = None):
        self.config = config or DataGenerationConfig()
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        
        # Initialize feature correlations and weights
        self._initialize_feature_relationships()
    
    def _initialize_feature_relationships(self):
        """Define realistic relationships between features and rockfall risk"""
        
        # Feature importance weights (higher = more important for rockfall risk)
        self.feature_weights = {
            'slope': 0.25,
            'fracture_density': 0.20,
            'rainfall': 0.15,
            'freeze_thaw_cycles': 0.12,
            'seismic_activity': 0.10,
            'instability_index': 0.08,
            'temperature_variation': 0.05,
            'wind_speed': 0.03,
            'roughness': 0.02
        }
        
        # Risk thresholds for different features
        self.risk_thresholds = {
            'slope': {'low': 30, 'medium': 45, 'high': 60},
            'fracture_density': {'low': 2, 'medium': 5, 'high': 8},
            'rainfall': {'low': 20, 'medium': 50, 'high': 100},
            'freeze_thaw_cycles': {'low': 5, 'medium': 15, 'high': 30},
            'seismic_activity': {'low': 2, 'medium': 4, 'high': 6},
            'temperature_variation': {'low': 10, 'medium': 20, 'high': 35}
        }
    
    def generate_terrain_features(self) -> pd.DataFrame:
        """Generate critical terrain features"""
        n = self.config.n_samples
        
        # Generate base terrain features
        terrain_data = {}
        
        # 1. Slope (primary risk factor)
        # Higher slopes have exponentially higher risk
        terrain_data['slope'] = np.random.exponential(scale=15, size=n)
        terrain_data['slope'] = np.clip(terrain_data['slope'], *self.config.slope_range)
        
        # 2. Elevation (affects stability processes)
        terrain_data['elevation'] = np.random.normal(loc=1200, scale=600, size=n)
        terrain_data['elevation'] = np.clip(terrain_data['elevation'], *self.config.elevation_range)
        
        # 3. Fracture Density (correlated with slope)
        base_fracture = np.random.exponential(scale=2, size=n)
        slope_influence = terrain_data['slope'] / 90 * 3  # Higher slopes = more fractures
        terrain_data['fracture_density'] = base_fracture + slope_influence
        terrain_data['fracture_density'] = np.clip(terrain_data['fracture_density'], 
                                                  *self.config.fracture_density_range)
        
        # 4. Roughness (surface irregularity)
        terrain_data['roughness'] = np.random.beta(a=2, b=5, size=n)  # Skewed toward lower values
        
        # 5. Slope Variability (local terrain complexity)
        terrain_data['slope_variability'] = np.random.gamma(shape=2, scale=5, size=n)
        terrain_data['slope_variability'] = np.clip(terrain_data['slope_variability'], 0, 50)
        
        # 6. Instability Index (combined terrain assessment)
        instability = (
            terrain_data['slope'] / 90 * 0.4 +
            terrain_data['fracture_density'] / 10 * 0.3 +
            terrain_data['roughness'] * 0.2 +
            terrain_data['slope_variability'] / 50 * 0.1
        )
        terrain_data['instability_index'] = instability + np.random.normal(0, 0.1, n)
        terrain_data['instability_index'] = np.clip(terrain_data['instability_index'], 0, 1)
        
        # 7. Topographic Wetness Index (water accumulation effects)
        # Higher in lower areas and flatter regions
        slope_radians = np.radians(terrain_data['slope'] + 0.1)  # Avoid division by zero
        contributing_area = np.random.lognormal(mean=5, sigma=1, size=n)
        terrain_data['wetness_index'] = np.log(contributing_area / np.tan(slope_radians))
        terrain_data['wetness_index'] = np.clip(terrain_data['wetness_index'], 0, 20)
        
        return pd.DataFrame(terrain_data)
    
    def generate_environmental_features(self) -> pd.DataFrame:
        """Generate environmental trigger features"""
        n = self.config.n_samples
        
        # Generate temporal component
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        dates = pd.date_range(start=start_date, end=end_date, periods=n)
        
        env_data = {'timestamp': dates}
        
        # Extract temporal features
        env_data['month'] = dates.month
        env_data['day_of_year'] = dates.dayofyear
        env_data['season'] = ((dates.month - 1) // 3) + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
        
        # 1. Rainfall (main trigger) - seasonal patterns
        seasonal_rainfall = np.where(
            env_data['season'] == 1, 50,  # Winter
            np.where(env_data['season'] == 2, 80,  # Spring
                    np.where(env_data['season'] == 3, 30, 70))  # Summer, Fall
        )
        rainfall_base = np.random.exponential(scale=seasonal_rainfall)
        env_data['rainfall'] = np.clip(rainfall_base, *self.config.rainfall_range)
        
        # 2. Temperature Variation (thermal stress)
        # Base temperature with seasonal variation
        seasonal_temp = 20 + 15 * np.sin(2 * np.pi * env_data['day_of_year'] / 365)
        daily_temp = seasonal_temp + np.random.normal(0, 5, n)
        env_data['temperature'] = np.clip(daily_temp, *self.config.temperature_range)
        
        # Temperature variation (daily range)
        env_data['temperature_variation'] = np.abs(np.random.normal(10, 5, n))
        env_data['temperature_variation'] = np.clip(env_data['temperature_variation'], 0, 40)
        
        # 3. Freeze-Thaw Cycles (winter weathering)
        # More likely in winter and spring, and at higher elevations
        freeze_thaw_base = np.where(
            env_data['season'].isin([1, 2]), 
            np.random.poisson(lam=8, size=n),  # Winter/Spring
            np.random.poisson(lam=2, size=n)   # Summer/Fall
        )
        env_data['freeze_thaw_cycles'] = np.clip(freeze_thaw_base, 0, 50)
        
        # 4. Seismic Activity (earthquake trigger)
        # Most earthquakes are small, few are large
        seismic_base = np.random.exponential(scale=1.5, size=n)
        env_data['seismic_activity'] = np.clip(seismic_base, *self.config.seismic_magnitude_range)
        
        # 5. Wind Speed (environmental loading)
        wind_seasonal = np.where(
            env_data['season'].isin([1, 4]), 
            np.random.gamma(shape=3, scale=15, size=n),  # Winter/Fall - windier
            np.random.gamma(shape=2, scale=10, size=n)   # Spring/Summer
        )
        env_data['wind_speed'] = np.clip(wind_seasonal, *self.config.wind_speed_range)
        
        # 6. Precipitation intensity (rate of rainfall)
        env_data['precipitation_intensity'] = np.where(
            env_data['rainfall'] > 0,
            env_data['rainfall'] / np.random.uniform(1, 24, n),  # mm/hour
            0
        )
        
        # 7. Humidity (affects rock weathering)
        env_data['humidity'] = np.random.beta(a=6, b=4, size=n) * 100  # 0-100%
        
        return pd.DataFrame(env_data)
    
    def calculate_risk_score(self, terrain_df: pd.DataFrame, env_df: pd.DataFrame) -> np.ndarray:
        """Calculate rockfall risk score based on all features"""
        
        # Normalize features to 0-1 scale for risk calculation
        normalized_features = {}
        
        # Terrain features (higher values = higher risk)
        normalized_features['slope'] = terrain_df['slope'] / 90
        normalized_features['fracture_density'] = terrain_df['fracture_density'] / 10
        normalized_features['instability_index'] = terrain_df['instability_index']
        normalized_features['roughness'] = terrain_df['roughness']
        normalized_features['slope_variability'] = terrain_df['slope_variability'] / 50
        normalized_features['wetness_index'] = np.clip(terrain_df['wetness_index'] / 20, 0, 1)
        
        # Environmental features
        normalized_features['rainfall'] = np.clip(env_df['rainfall'] / 200, 0, 1)
        normalized_features['freeze_thaw_cycles'] = np.clip(env_df['freeze_thaw_cycles'] / 50, 0, 1)
        normalized_features['seismic_activity'] = env_df['seismic_activity'] / 7
        normalized_features['temperature_variation'] = np.clip(env_df['temperature_variation'] / 40, 0, 1)
        normalized_features['wind_speed'] = env_df['wind_speed'] / 120
        
        # Calculate weighted risk score
        risk_score = np.zeros(len(terrain_df))
        
        for feature, weight in self.feature_weights.items():
            if feature in normalized_features:
                risk_score += normalized_features[feature] * weight
        
        # Add interaction effects
        # Rainfall + slope interaction (very dangerous combination)
        rainfall_slope_interaction = (normalized_features['rainfall'] * 
                                    normalized_features['slope'] * 0.15)
        
        # Freeze-thaw + fracture density interaction
        freeze_fracture_interaction = (normalized_features['freeze_thaw_cycles'] * 
                                     normalized_features['fracture_density'] * 0.10)
        
        # Seismic + instability interaction
        seismic_instability_interaction = (normalized_features['seismic_activity'] * 
                                         normalized_features['instability_index'] * 0.08)
        
        risk_score += (rainfall_slope_interaction + 
                      freeze_fracture_interaction + 
                      seismic_instability_interaction)
        
        # Add some noise and ensure 0-1 range
        risk_score += np.random.normal(0, 0.05, len(terrain_df))
        risk_score = np.clip(risk_score, 0, 1)
        
        return risk_score
    
    def generate_rockfall_events(self, risk_scores: np.ndarray, 
                                threshold: float = 0.7) -> np.ndarray:
        """Generate binary rockfall events based on risk scores"""
        
        # Higher risk = higher probability of rockfall
        # Use sigmoid function to convert risk to probability
        probability = 1 / (1 + np.exp(-10 * (risk_scores - threshold)))
        
        # Generate events based on probability
        events = np.random.binomial(1, probability)
        
        # Ensure we have some events for training
        if events.sum() < 50:  # Minimum number of positive cases
            # Force some high-risk cases to be positive
            high_risk_indices = np.where(risk_scores > 0.6)[0]
            if len(high_risk_indices) > 0:
                selected_indices = np.random.choice(
                    high_risk_indices, 
                    size=min(100, len(high_risk_indices)), 
                    replace=False
                )
                events[selected_indices] = 1
        
        return events
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """Generate complete synthetic dataset with all features"""
        
        print("🏔️ Generating synthetic rockfall prediction dataset...")
        print(f"📊 Creating {self.config.n_samples:,} samples...")
        
        # Generate terrain features
        print("🗻 Generating terrain features...")
        terrain_df = self.generate_terrain_features()
        
        # Generate environmental features
        print("🌤️ Generating environmental features...")
        env_df = self.generate_environmental_features()
        
        # Combine datasets
        combined_df = pd.concat([terrain_df, env_df], axis=1)
        
        # Calculate risk scores
        print("⚠️ Calculating risk scores...")
        risk_scores = self.calculate_risk_score(terrain_df, env_df)
        combined_df['risk_score'] = risk_scores
        
        # Generate rockfall events
        print("💥 Generating rockfall events...")
        rockfall_events = self.generate_rockfall_events(risk_scores)
        combined_df['rockfall_event'] = rockfall_events
        
        # Add risk categories
        combined_df['risk_category'] = pd.cut(
            risk_scores, 
            bins=[0, 0.3, 0.6, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        print(f"✅ Dataset generated successfully!")
        print(f"📈 Risk distribution: {combined_df['risk_category'].value_counts().to_dict()}")
        print(f"💥 Rockfall events: {rockfall_events.sum():,} ({rockfall_events.mean()*100:.1f}%)")
        
        return combined_df
    
    def visualize_dataset(self, df: pd.DataFrame, save_path: str = None):
        """Create comprehensive visualizations of the synthetic dataset"""
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Synthetic Rockfall Prediction Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Risk Score Distribution
        axes[0,0].hist(df['risk_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Risk Score Distribution')
        axes[0,0].set_xlabel('Risk Score')
        axes[0,0].set_ylabel('Frequency')
        
        # 2. Slope vs Risk Score
        scatter = axes[0,1].scatter(df['slope'], df['risk_score'], 
                                  c=df['rockfall_event'], cmap='coolwarm', alpha=0.6)
        axes[0,1].set_title('Slope vs Risk Score')
        axes[0,1].set_xlabel('Slope (degrees)')
        axes[0,1].set_ylabel('Risk Score')
        plt.colorbar(scatter, ax=axes[0,1])
        
        # 3. Rainfall vs Risk Score
        axes[0,2].scatter(df['rainfall'], df['risk_score'], 
                         c=df['rockfall_event'], cmap='coolwarm', alpha=0.6)
        axes[0,2].set_title('Rainfall vs Risk Score')
        axes[0,2].set_xlabel('Rainfall (mm)')
        axes[0,2].set_ylabel('Risk Score')
        
        # 4. Fracture Density Distribution
        axes[0,3].hist(df['fracture_density'], bins=30, alpha=0.7, color='lightcoral')
        axes[0,3].set_title('Fracture Density Distribution')
        axes[0,3].set_xlabel('Fracture Density (per m²)')
        axes[0,3].set_ylabel('Frequency')
        
        # 5. Seasonal Risk Patterns
        seasonal_risk = df.groupby('season')['risk_score'].mean()
        axes[1,0].bar(['Winter', 'Spring', 'Summer', 'Fall'], seasonal_risk.values, 
                     color=['lightblue', 'lightgreen', 'yellow', 'orange'])
        axes[1,0].set_title('Seasonal Risk Patterns')
        axes[1,0].set_ylabel('Average Risk Score')
        
        # 6. Temperature Variation Impact
        axes[1,1].scatter(df['temperature_variation'], df['risk_score'], 
                         alpha=0.5, color='purple')
        axes[1,1].set_title('Temperature Variation vs Risk')
        axes[1,1].set_xlabel('Temperature Variation (°C)')
        axes[1,1].set_ylabel('Risk Score')
        
        # 7. Seismic Activity Distribution
        axes[1,2].hist(df['seismic_activity'], bins=30, alpha=0.7, color='red')
        axes[1,2].set_title('Seismic Activity Distribution')
        axes[1,2].set_xlabel('Magnitude')
        axes[1,2].set_ylabel('Frequency')
        
        # 8. Instability Index vs Rockfall Events
        rockfall_yes = df[df['rockfall_event'] == 1]['instability_index']
        rockfall_no = df[df['rockfall_event'] == 0]['instability_index']
        axes[1,3].hist([rockfall_no, rockfall_yes], bins=30, alpha=0.7, 
                      label=['No Rockfall', 'Rockfall'], color=['blue', 'red'])
        axes[1,3].set_title('Instability Index vs Rockfall Events')
        axes[1,3].set_xlabel('Instability Index')
        axes[1,3].set_ylabel('Frequency')
        axes[1,3].legend()
        
        # 9. Feature Correlation Matrix
        corr_features = ['slope', 'fracture_density', 'rainfall', 'freeze_thaw_cycles', 
                        'seismic_activity', 'risk_score', 'rockfall_event']
        corr_matrix = df[corr_features].corr()
        im = axes[2,0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[2,0].set_title('Feature Correlation Matrix')
        axes[2,0].set_xticks(range(len(corr_features)))
        axes[2,0].set_yticks(range(len(corr_features)))
        axes[2,0].set_xticklabels(corr_features, rotation=45)
        axes[2,0].set_yticklabels(corr_features)
        plt.colorbar(im, ax=axes[2,0])
        
        # 10. Wind Speed vs Risk
        axes[2,1].scatter(df['wind_speed'], df['risk_score'], alpha=0.5, color='green')
        axes[2,1].set_title('Wind Speed vs Risk Score')
        axes[2,1].set_xlabel('Wind Speed (km/h)')
        axes[2,1].set_ylabel('Risk Score')
        
        # 11. Elevation vs Risk Score
        axes[2,2].scatter(df['elevation'], df['risk_score'], alpha=0.5, color='brown')
        axes[2,2].set_title('Elevation vs Risk Score')
        axes[2,2].set_xlabel('Elevation (m)')
        axes[2,2].set_ylabel('Risk Score')
        
        # 12. Risk Category Distribution
        risk_counts = df['risk_category'].value_counts()
        axes[2,3].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                     colors=['green', 'yellow', 'red'])
        axes[2,3].set_title('Risk Category Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        
        plt.show()
    
    def generate_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary of the synthetic dataset"""
        
        summary = {
            'dataset_overview': {
                'total_samples': len(df),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                'rockfall_events': int(df['rockfall_event'].sum()),
                'rockfall_rate': f"{df['rockfall_event'].mean()*100:.2f}%"
            },
            'terrain_features': {},
            'environmental_features': {},
            'risk_analysis': {},
            'correlations': {}
        }
        
        # Terrain feature statistics
        terrain_features = ['slope', 'fracture_density', 'instability_index', 
                          'elevation', 'roughness', 'slope_variability', 'wetness_index']
        
        for feature in terrain_features:
            if feature in df.columns:
                summary['terrain_features'][feature] = {
                    'mean': f"{df[feature].mean():.2f}",
                    'std': f"{df[feature].std():.2f}",
                    'min': f"{df[feature].min():.2f}",
                    'max': f"{df[feature].max():.2f}",
                    'high_risk_correlation': f"{df[feature].corr(df['rockfall_event']):.3f}"
                }
        
        # Environmental feature statistics
        env_features = ['rainfall', 'freeze_thaw_cycles', 'seismic_activity', 
                       'temperature_variation', 'wind_speed', 'precipitation_intensity']
        
        for feature in env_features:
            if feature in df.columns:
                summary['environmental_features'][feature] = {
                    'mean': f"{df[feature].mean():.2f}",
                    'std': f"{df[feature].std():.2f}",
                    'min': f"{df[feature].min():.2f}",
                    'max': f"{df[feature].max():.2f}",
                    'high_risk_correlation': f"{df[feature].corr(df['rockfall_event']):.3f}"
                }
        
        # Risk analysis
        summary['risk_analysis'] = {
            'average_risk_score': f"{df['risk_score'].mean():.3f}",
            'high_risk_samples': f"{(df['risk_score'] > 0.7).sum():,} ({(df['risk_score'] > 0.7).mean()*100:.1f}%)",
            'medium_risk_samples': f"{((df['risk_score'] > 0.3) & (df['risk_score'] <= 0.7)).sum():,}",
            'low_risk_samples': f"{(df['risk_score'] <= 0.3).sum():,}",
            'risk_categories': df['risk_category'].value_counts().to_dict()
        }
        
        # Key correlations with rockfall events
        correlation_features = ['slope', 'fracture_density', 'rainfall', 'risk_score']
        correlations = {}
        for feature in correlation_features:
            if feature in df.columns:
                correlations[feature] = f"{df[feature].corr(df['rockfall_event']):.3f}"
        
        summary['correlations'] = correlations
        
        return summary


def main():
    """Demonstrate synthetic data generation"""
    
    # Configure data generation
    config = DataGenerationConfig(
        n_samples=5000,
        random_seed=42
    )
    
    # Create generator
    generator = SyntheticDataGenerator(config)
    
    # Generate dataset
    dataset = generator.generate_complete_dataset()
    
    # Save dataset
    output_path = "outputs/synthetic_training_data.csv"
    dataset.to_csv(output_path, index=False)
    print(f"💾 Dataset saved to: {output_path}")
    
    # Generate visualizations
    viz_path = "outputs/synthetic_data_analysis.png"
    generator.visualize_dataset(dataset, viz_path)
    
    # Generate summary
    summary = generator.generate_feature_summary(dataset)
    
    print("\n📊 DATASET SUMMARY")
    print("="*50)
    for category, data in summary.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {data}")
    
    return dataset, summary


if __name__ == "__main__":
    dataset, summary = main()