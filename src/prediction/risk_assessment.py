"""
Comprehensive Risk Assessment Framework for Rockfall Prediction
============================================================

This module provides a complete risk assessment system that combines:
- Real-time risk scoring and probability forecasting
- Risk zone mapping and spatial analysis
- Alert generation and management
- Historical risk trend analysis
- Integration with ML prediction models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Geospatial libraries
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.features import shapes
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    print("⚠️ Geospatial libraries not fully available. Some features may be limited.")

class RiskLevel:
    """Risk level definitions and thresholds"""
    
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3
    
    NAMES = {
        LOW: 'Low',
        MEDIUM: 'Medium', 
        HIGH: 'High',
        CRITICAL: 'Critical'
    }
    
    COLORS = {
        LOW: '#00FF00',      # Green
        MEDIUM: '#FFFF00',   # Yellow
        HIGH: '#FF8000',     # Orange
        CRITICAL: '#FF0000'  # Red
    }
    
    THRESHOLDS = {
        'risk_score': [0.0, 0.3, 0.6, 0.8, 1.0],
        'probability': [0.0, 0.2, 0.5, 0.8, 1.0]
    }

class RiskAssessmentFramework:
    """Comprehensive risk assessment system for rockfall prediction"""
    
    def __init__(self, ml_predictor=None):
        self.ml_predictor = ml_predictor
        self.risk_history = []
        self.alert_history = []
        self.risk_zones = {}
        self.current_risk_state = {}
        
        # Risk assessment parameters
        self.risk_weights = {
            'ml_prediction': 0.40,      # ML model prediction
            'terrain_risk': 0.25,       # Static terrain-based risk
            'environmental_risk': 0.20, # Current environmental conditions
            'historical_risk': 0.10,    # Historical patterns
            'proximity_risk': 0.05      # Proximity to previous events
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'immediate': 0.8,    # Immediate evacuation
            'high': 0.6,         # High alert - prepare for evacuation
            'medium': 0.4,       # Medium alert - increased monitoring
            'low': 0.2           # Low alert - routine monitoring
        }
        
        print("🛡️ Risk Assessment Framework initialized")
    
    def calculate_terrain_risk(self, terrain_features: Dict[str, float]) -> Dict:
        """Calculate static terrain-based risk score"""
        
        risk_factors = {}
        weights = {}
        
        # Slope risk (primary factor)
        if 'slope' in terrain_features:
            slope = terrain_features['slope']
            if slope < 20:
                risk_factors['slope'] = 0.1
            elif slope < 35:
                risk_factors['slope'] = 0.3
            elif slope < 50:
                risk_factors['slope'] = 0.6
            elif slope < 70:
                risk_factors['slope'] = 0.8
            else:
                risk_factors['slope'] = 1.0
            weights['slope'] = 0.35
        
        # Fracture density risk
        if 'fracture_density' in terrain_features:
            fracture_density = terrain_features['fracture_density']
            risk_factors['fracture_density'] = min(fracture_density / 10, 1.0)
            weights['fracture_density'] = 0.25
        
        # Instability index risk
        if 'instability_index' in terrain_features:
            instability = terrain_features['instability_index']
            risk_factors['instability_index'] = min(instability, 1.0)
            weights['instability_index'] = 0.20
        
        # Roughness risk (inverse relationship - smoother = higher risk)
        if 'roughness' in terrain_features:
            roughness = terrain_features['roughness']
            risk_factors['roughness'] = 1.0 - min(roughness, 1.0)
            weights['roughness'] = 0.10
        
        # Wetness index risk
        if 'wetness_index' in terrain_features:
            wetness = terrain_features['wetness_index']
            risk_factors['wetness_index'] = min(wetness / 20, 1.0)
            weights['wetness_index'] = 0.10
        
        # Calculate weighted terrain risk
        if risk_factors and weights:
            total_weight = sum(weights.values())
            terrain_risk = sum(risk_factors[factor] * weights[factor] 
                             for factor in risk_factors) / total_weight
        else:
            terrain_risk = 0.5  # Default moderate risk
        
        return {
            'terrain_risk_score': terrain_risk,
            'risk_factors': risk_factors,
            'factor_weights': weights
        }
    
    def calculate_environmental_risk(self, env_conditions: Dict[str, float]) -> Dict:
        """Calculate environmental trigger-based risk score"""
        
        risk_factors = {}
        weights = {}
        
        # Rainfall risk (primary environmental trigger)
        if 'rainfall' in env_conditions:
            rainfall = env_conditions['rainfall']
            if rainfall < 5:
                risk_factors['rainfall'] = 0.1
            elif rainfall < 20:
                risk_factors['rainfall'] = 0.3
            elif rainfall < 50:
                risk_factors['rainfall'] = 0.6
            elif rainfall < 100:
                risk_factors['rainfall'] = 0.8
            else:
                risk_factors['rainfall'] = 1.0
            weights['rainfall'] = 0.40
        
        # Freeze-thaw cycle risk
        if 'freeze_thaw_cycles' in env_conditions:
            freeze_thaw = env_conditions['freeze_thaw_cycles']
            risk_factors['freeze_thaw'] = min(freeze_thaw / 50, 1.0)
            weights['freeze_thaw'] = 0.25
        
        # Seismic activity risk
        if 'seismic_activity' in env_conditions:
            seismic = env_conditions['seismic_activity']
            if seismic < 2:
                risk_factors['seismic'] = 0.0
            elif seismic < 4:
                risk_factors['seismic'] = 0.3
            elif seismic < 6:
                risk_factors['seismic'] = 0.7
            else:
                risk_factors['seismic'] = 1.0
            weights['seismic'] = 0.20
        
        # Temperature variation risk
        if 'temperature_variation' in env_conditions:
            temp_var = env_conditions['temperature_variation']
            risk_factors['temperature'] = min(temp_var / 40, 1.0)
            weights['temperature'] = 0.10
        
        # Wind speed risk
        if 'wind_speed' in env_conditions:
            wind = env_conditions['wind_speed']
            risk_factors['wind'] = min(wind / 120, 1.0)
            weights['wind'] = 0.05
        
        # Calculate weighted environmental risk
        if risk_factors and weights:
            total_weight = sum(weights.values())
            env_risk = sum(risk_factors[factor] * weights[factor] 
                          for factor in risk_factors) / total_weight
        else:
            env_risk = 0.3  # Default low-moderate risk
        
        return {
            'environmental_risk_score': env_risk,
            'risk_factors': risk_factors,
            'factor_weights': weights
        }
    
    def calculate_historical_risk(self, location: Tuple[float, float], 
                                 time_window_days: int = 30) -> Dict:
        """Calculate risk based on historical patterns at location"""
        
        historical_events = 0
        recent_events = 0
        avg_risk_score = 0.3
        
        # Analyze historical risk data if available
        if self.risk_history:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=time_window_days)
            
            for entry in self.risk_history:
                entry_time = entry.get('timestamp', current_time)
                entry_location = entry.get('location', (0, 0))
                
                # Calculate distance to current location
                distance = np.sqrt((entry_location[0] - location[0])**2 + 
                                 (entry_location[1] - location[1])**2)
                
                # Consider events within 1km radius
                if distance < 0.01:  # Approximately 1km in degrees
                    historical_events += 1
                    
                    if entry_time >= cutoff_time:
                        recent_events += 1
                    
                    # Update average risk
                    event_risk = entry.get('final_risk_score', 0.3)
                    avg_risk_score = (avg_risk_score + event_risk) / 2
        
        # Calculate historical risk score
        if historical_events == 0:
            historical_risk = 0.2  # Low risk for areas with no history
        else:
            base_risk = min(historical_events / 10, 0.6)  # Cap at 0.6
            recent_multiplier = 1 + (recent_events / max(historical_events, 1))
            historical_risk = min(base_risk * recent_multiplier, 1.0)
        
        return {
            'historical_risk_score': historical_risk,
            'historical_events': historical_events,
            'recent_events': recent_events,
            'average_historical_risk': avg_risk_score
        }
    
    def calculate_proximity_risk(self, location: Tuple[float, float], 
                               radius_km: float = 5.0) -> Dict:
        """Calculate risk based on proximity to recent rockfall events"""
        
        proximity_risk = 0.0
        nearby_events = 0
        closest_distance = float('inf')
        
        if self.alert_history:
            current_time = datetime.now()
            recent_cutoff = current_time - timedelta(hours=24)  # Last 24 hours
            
            for alert in self.alert_history:
                alert_time = alert.get('timestamp', current_time)
                alert_location = alert.get('location', (0, 0))
                
                if alert_time >= recent_cutoff:
                    # Calculate distance in kilometers (rough approximation)
                    distance_deg = np.sqrt((alert_location[0] - location[0])**2 + 
                                         (alert_location[1] - location[1])**2)
                    distance_km = distance_deg * 111.32  # Approximate km per degree
                    
                    if distance_km <= radius_km:
                        nearby_events += 1
                        closest_distance = min(closest_distance, distance_km)
                        
                        # Proximity risk decreases with distance
                        event_proximity_risk = max(0, 1 - (distance_km / radius_km))
                        proximity_risk = max(proximity_risk, event_proximity_risk)
        
        return {
            'proximity_risk_score': proximity_risk,
            'nearby_events': nearby_events,
            'closest_event_distance_km': closest_distance if closest_distance != float('inf') else None
        }
    
    def assess_comprehensive_risk(self, location: Tuple[float, float],
                                terrain_features: Dict[str, float],
                                env_conditions: Dict[str, float],
                                use_ml_prediction: bool = True) -> Dict:
        """Perform comprehensive risk assessment combining all factors"""
        
        risk_components = {}
        
        # 1. Terrain-based risk
        terrain_risk = self.calculate_terrain_risk(terrain_features)
        risk_components['terrain'] = terrain_risk
        
        # 2. Environmental risk
        env_risk = self.calculate_environmental_risk(env_conditions)
        risk_components['environmental'] = env_risk
        
        # 3. Historical risk
        historical_risk = self.calculate_historical_risk(location)
        risk_components['historical'] = historical_risk
        
        # 4. Proximity risk
        proximity_risk = self.calculate_proximity_risk(location)
        risk_components['proximity'] = proximity_risk
        
        # 5. ML prediction risk (if available)
        ml_risk_score = 0.5  # Default
        if use_ml_prediction and self.ml_predictor:
            try:
                # Prepare feature vector for ML prediction
                feature_vector = {**terrain_features, **env_conditions}
                feature_df = pd.DataFrame([feature_vector])
                
                # Get ML prediction
                ml_prediction = self.ml_predictor.predict_risk(feature_df.values, 'ensemble')
                ml_risk_score = float(ml_prediction[0])
                
            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}")
                ml_risk_score = 0.5
        
        risk_components['ml_prediction'] = {'ml_risk_score': ml_risk_score}
        
        # Calculate final weighted risk score
        final_risk_score = (
            terrain_risk['terrain_risk_score'] * self.risk_weights['terrain_risk'] +
            env_risk['environmental_risk_score'] * self.risk_weights['environmental_risk'] +
            historical_risk['historical_risk_score'] * self.risk_weights['historical_risk'] +
            proximity_risk['proximity_risk_score'] * self.risk_weights['proximity_risk'] +
            ml_risk_score * self.risk_weights['ml_prediction']
        )
        
        # Determine risk level
        risk_level = self._get_risk_level(final_risk_score)
        
        # Calculate probability forecast
        probability_forecast = self._calculate_probability_forecast(
            final_risk_score, risk_components
        )
        
        # Generate risk assessment summary
        risk_assessment = {
            'timestamp': datetime.now(),
            'location': location,
            'final_risk_score': final_risk_score,
            'risk_level': risk_level,
            'risk_level_name': RiskLevel.NAMES[risk_level],
            'probability_forecast': probability_forecast,
            'risk_components': risk_components,
            'component_weights': self.risk_weights,
            'confidence_score': self._calculate_confidence_score(risk_components)
        }
        
        # Store in risk history
        self.risk_history.append(risk_assessment)
        
        # Check for alert conditions
        alert_info = self._check_alert_conditions(risk_assessment)
        if alert_info:
            risk_assessment['alert_generated'] = alert_info
            self.alert_history.append(alert_info)
        
        return risk_assessment
    
    def _get_risk_level(self, risk_score: float) -> int:
        """Convert risk score to discrete risk level"""
        
        thresholds = RiskLevel.THRESHOLDS['risk_score']
        
        for i in range(len(thresholds) - 1):
            if thresholds[i] <= risk_score < thresholds[i + 1]:
                return i
        
        return RiskLevel.CRITICAL  # Highest level for scores >= 0.8
    
    def _calculate_probability_forecast(self, risk_score: float, 
                                      risk_components: Dict) -> Dict:
        """Calculate probability forecasts for different time horizons"""
        
        # Base probability from risk score
        base_prob = min(risk_score * 1.2, 1.0)  # Slight amplification
        
        # Adjust based on environmental urgency
        env_risk = risk_components['environmental']['environmental_risk_score']
        urgency_multiplier = 1 + (env_risk * 0.5)  # Up to 50% increase
        
        # Calculate probabilities for different time windows
        probabilities = {
            '1_hour': base_prob * urgency_multiplier * 0.1,      # Very low short-term
            '6_hours': base_prob * urgency_multiplier * 0.3,     # Low-moderate
            '24_hours': base_prob * urgency_multiplier * 0.6,    # Moderate-high
            '7_days': base_prob * urgency_multiplier * 0.8,      # High
            '30_days': base_prob * urgency_multiplier            # Full probability
        }
        
        # Ensure probabilities are within valid range
        for timeframe in probabilities:
            probabilities[timeframe] = min(probabilities[timeframe], 1.0)
        
        return probabilities
    
    def _calculate_confidence_score(self, risk_components: Dict) -> float:
        """Calculate confidence in the risk assessment"""
        
        confidence_factors = []
        
        # ML model confidence (if available)
        ml_component = risk_components.get('ml_prediction', {})
        if 'ml_risk_score' in ml_component:
            # Confidence is higher when prediction is decisive (close to 0 or 1)
            ml_score = ml_component['ml_risk_score']
            ml_confidence = 1 - 2 * abs(ml_score - 0.5)  # Distance from 0.5
            confidence_factors.append(ml_confidence * 0.4)
        
        # Data availability confidence
        terrain_factors = len(risk_components.get('terrain', {}).get('risk_factors', {}))
        env_factors = len(risk_components.get('environmental', {}).get('risk_factors', {}))
        
        data_completeness = (terrain_factors + env_factors) / 10  # Assume max 10 factors
        confidence_factors.append(min(data_completeness, 1.0) * 0.3)
        
        # Historical data confidence
        historical_events = risk_components.get('historical', {}).get('historical_events', 0)
        historical_confidence = min(historical_events / 5, 1.0)  # Max confidence at 5+ events
        confidence_factors.append(historical_confidence * 0.2)
        
        # Recent events confidence
        nearby_events = risk_components.get('proximity', {}).get('nearby_events', 0)
        proximity_confidence = min(nearby_events / 3, 1.0)  # Max confidence at 3+ nearby events
        confidence_factors.append(proximity_confidence * 0.1)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors) if confidence_factors else 0.5
        return min(overall_confidence, 1.0)
    
    def _check_alert_conditions(self, risk_assessment: Dict) -> Optional[Dict]:
        """Check if alert conditions are met and generate alert"""
        
        risk_score = risk_assessment['final_risk_score']
        risk_level = risk_assessment['risk_level']
        location = risk_assessment['location']
        
        # Determine alert level
        alert_level = None
        if risk_score >= self.alert_thresholds['immediate']:
            alert_level = 'immediate'
        elif risk_score >= self.alert_thresholds['high']:
            alert_level = 'high'
        elif risk_score >= self.alert_thresholds['medium']:
            alert_level = 'medium'
        elif risk_score >= self.alert_thresholds['low']:
            alert_level = 'low'
        
        if alert_level:
            alert_info = {
                'timestamp': datetime.now(),
                'alert_level': alert_level,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'location': location,
                'message': self._generate_alert_message(alert_level, risk_assessment),
                'recommended_actions': self._get_recommended_actions(alert_level),
                'confidence': risk_assessment['confidence_score']
            }
            
            return alert_info
        
        return None
    
    def _generate_alert_message(self, alert_level: str, risk_assessment: Dict) -> str:
        """Generate human-readable alert message"""
        
        location = risk_assessment['location']
        risk_score = risk_assessment['final_risk_score']
        confidence = risk_assessment['confidence_score']
        
        messages = {
            'immediate': f"🚨 IMMEDIATE EVACUATION REQUIRED! Critical rockfall risk detected at location ({location[0]:.4f}, {location[1]:.4f}). Risk score: {risk_score:.2f} (Confidence: {confidence:.2f})",
            
            'high': f"⚠️ HIGH ALERT: Significant rockfall risk at location ({location[0]:.4f}, {location[1]:.4f}). Prepare for possible evacuation. Risk score: {risk_score:.2f}",
            
            'medium': f"🔶 MEDIUM ALERT: Elevated rockfall risk detected at location ({location[0]:.4f}, {location[1]:.4f}). Increase monitoring. Risk score: {risk_score:.2f}",
            
            'low': f"🔸 LOW ALERT: Slight increase in rockfall risk at location ({location[0]:.4f}, {location[1]:.4f}). Continue routine monitoring. Risk score: {risk_score:.2f}"
        }
        
        return messages.get(alert_level, "Unknown alert level")
    
    def _get_recommended_actions(self, alert_level: str) -> List[str]:
        """Get recommended actions for each alert level"""
        
        actions = {
            'immediate': [
                "Evacuate all personnel from the area immediately",
                "Establish 500m safety perimeter",
                "Contact emergency services",
                "Activate emergency response protocol",
                "Monitor continuously until risk subsides"
            ],
            'high': [
                "Prepare evacuation plans and routes",
                "Restrict access to high-risk zones",
                "Increase monitoring frequency to every 15 minutes",
                "Alert all on-site personnel",
                "Position emergency equipment"
            ],
            'medium': [
                "Increase monitoring frequency to hourly",
                "Brief personnel on evacuation procedures",
                "Restrict non-essential activities in risk zone",
                "Prepare monitoring equipment",
                "Review weather forecasts"
            ],
            'low': [
                "Continue routine monitoring",
                "Check equipment functionality",
                "Review recent sensor data",
                "Update risk assessment records"
            ]
        }
        
        return actions.get(alert_level, ["No specific actions required"])
    
    def create_risk_map(self, area_bounds: Tuple[float, float, float, float],
                       resolution: float = 0.001) -> Dict:
        """Create spatial risk map for specified area"""
        
        min_x, min_y, max_x, max_y = area_bounds
        
        # Create grid of points
        x_points = np.arange(min_x, max_x, resolution)
        y_points = np.arange(min_y, max_y, resolution)
        
        risk_grid = np.zeros((len(y_points), len(x_points)))
        
        print(f"🗺️ Creating risk map for area: {area_bounds}")
        print(f"📏 Grid size: {len(y_points)} x {len(x_points)} = {len(y_points) * len(x_points):,} points")
        
        # Calculate risk for each grid point
        for i, y in enumerate(y_points):
            for j, x in enumerate(x_points):
                # Use simplified risk calculation for mapping
                # In practice, you would extract actual terrain/environmental data
                
                # Simulate terrain features (normally from DEM)
                terrain_features = {
                    'slope': np.random.uniform(0, 90),
                    'fracture_density': np.random.uniform(0, 10),
                    'instability_index': np.random.uniform(0, 1),
                    'roughness': np.random.uniform(0, 1)
                }
                
                # Simulate environmental conditions
                env_conditions = {
                    'rainfall': np.random.uniform(0, 100),
                    'freeze_thaw_cycles': np.random.uniform(0, 30),
                    'seismic_activity': np.random.uniform(0, 5),
                    'temperature_variation': np.random.uniform(0, 30)
                }
                
                # Calculate risk (simplified)
                risk_assessment = self.assess_comprehensive_risk(
                    (x, y), terrain_features, env_conditions, use_ml_prediction=False
                )
                
                risk_grid[i, j] = risk_assessment['final_risk_score']
        
        # Create risk zones
        risk_zones = self._create_risk_zones(risk_grid, x_points, y_points)
        
        return {
            'risk_grid': risk_grid,
            'x_coordinates': x_points,
            'y_coordinates': y_points,
            'bounds': area_bounds,
            'risk_zones': risk_zones,
            'resolution': resolution
        }
    
    def _create_risk_zones(self, risk_grid: np.ndarray, 
                          x_coords: np.ndarray, y_coords: np.ndarray) -> Dict:
        """Create discrete risk zones from continuous risk grid"""
        
        zones = {}
        
        for level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            thresholds = RiskLevel.THRESHOLDS['risk_score']
            
            if level < len(thresholds) - 1:
                mask = (risk_grid >= thresholds[level]) & (risk_grid < thresholds[level + 1])
            else:
                mask = risk_grid >= thresholds[level]
            
            zone_pixels = np.sum(mask)
            total_pixels = risk_grid.size
            coverage_percent = (zone_pixels / total_pixels) * 100
            
            zones[RiskLevel.NAMES[level]] = {
                'pixel_count': int(zone_pixels),
                'coverage_percent': coverage_percent,
                'risk_level': level,
                'color': RiskLevel.COLORS[level]
            }
        
        return zones
    
    def visualize_risk_assessment(self, risk_assessment: Dict, save_path: str = None):
        """Create comprehensive visualization of risk assessment"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Rockfall Risk Assessment', fontsize=16, fontweight='bold')
        
        # 1. Risk Score Breakdown
        components = risk_assessment['risk_components']
        risk_scores = [
            components['terrain']['terrain_risk_score'],
            components['environmental']['environmental_risk_score'],
            components['historical']['historical_risk_score'],
            components['proximity']['proximity_risk_score'],
            components['ml_prediction']['ml_risk_score']
        ]
        
        component_names = ['Terrain', 'Environmental', 'Historical', 'Proximity', 'ML Prediction']
        colors = ['brown', 'skyblue', 'gray', 'orange', 'purple']
        
        axes[0,0].bar(component_names, risk_scores, color=colors, alpha=0.7)
        axes[0,0].set_title('Risk Component Breakdown')
        axes[0,0].set_ylabel('Risk Score')
        axes[0,0].set_ylim(0, 1)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Final Risk Level
        risk_level = risk_assessment['risk_level']
        risk_score = risk_assessment['final_risk_score']
        
        # Create gauge-style plot
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Color segments by risk level
        colors_gauge = ['green', 'yellow', 'orange', 'red']
        thresholds = [0, 0.3, 0.6, 0.8, 1.0]
        
        for i in range(len(thresholds) - 1):
            theta_segment = theta[(theta >= thresholds[i] * np.pi) & (theta <= thresholds[i+1] * np.pi)]
            r_segment = np.ones_like(theta_segment)
            axes[0,1].fill_between(theta_segment, 0, r_segment, color=colors_gauge[i], alpha=0.7)
        
        # Add risk score needle
        risk_angle = risk_score * np.pi
        axes[0,1].plot([risk_angle, risk_angle], [0, 1], 'k-', linewidth=3)
        axes[0,1].set_title(f'Risk Level: {RiskLevel.NAMES[risk_level]}\nScore: {risk_score:.3f}')
        axes[0,1].set_ylim(0, 1.2)
        
        # 3. Probability Forecast
        prob_forecast = risk_assessment['probability_forecast']
        timeframes = list(prob_forecast.keys())
        probabilities = list(prob_forecast.values())
        
        axes[0,2].plot(timeframes, probabilities, 'ro-', linewidth=2, markersize=8)
        axes[0,2].set_title('Probability Forecast')
        axes[0,2].set_ylabel('Probability')
        axes[0,2].set_ylim(0, 1)
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Risk Factor Analysis (Terrain)
        terrain_factors = components['terrain']['risk_factors']
        if terrain_factors:
            factor_names = list(terrain_factors.keys())
            factor_values = list(terrain_factors.values())
            
            axes[1,0].barh(factor_names, factor_values, color='brown', alpha=0.7)
            axes[1,0].set_title('Terrain Risk Factors')
            axes[1,0].set_xlabel('Risk Score')
            axes[1,0].set_xlim(0, 1)
        
        # 5. Environmental Conditions
        env_factors = components['environmental']['risk_factors']
        if env_factors:
            factor_names = list(env_factors.keys())
            factor_values = list(env_factors.values())
            
            axes[1,1].barh(factor_names, factor_values, color='skyblue', alpha=0.7)
            axes[1,1].set_title('Environmental Risk Factors')
            axes[1,1].set_xlabel('Risk Score')
            axes[1,1].set_xlim(0, 1)
        
        # 6. Historical and Proximity Analysis
        hist_data = components['historical']
        prox_data = components['proximity']
        
        info_text = f"""
Historical Analysis:
• Events in area: {hist_data['historical_events']}
• Recent events: {hist_data['recent_events']}
• Avg historical risk: {hist_data['average_historical_risk']:.3f}

Proximity Analysis:
• Nearby events (24h): {prox_data['nearby_events']}
• Closest distance: {prox_data.get('closest_event_distance_km', 'N/A')} km

Assessment Info:
• Timestamp: {risk_assessment['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
• Location: ({risk_assessment['location'][0]:.4f}, {risk_assessment['location'][1]:.4f})
• Confidence: {risk_assessment['confidence_score']:.3f}
        """
        
        axes[1,2].text(0.1, 0.9, info_text, transform=axes[1,2].transAxes, 
                      fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1,2].set_title('Historical & Proximity Analysis')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Risk assessment visualization saved to: {save_path}")
        
        plt.show()
    
    def export_risk_report(self, risk_assessment: Dict, file_path: str):
        """Export comprehensive risk assessment report"""
        
        report = {
            'risk_assessment_report': {
                'metadata': {
                    'timestamp': risk_assessment['timestamp'].isoformat(),
                    'location': risk_assessment['location'],
                    'report_version': '1.0'
                },
                'summary': {
                    'final_risk_score': risk_assessment['final_risk_score'],
                    'risk_level': risk_assessment['risk_level_name'],
                    'confidence_score': risk_assessment['confidence_score']
                },
                'detailed_analysis': risk_assessment['risk_components'],
                'probability_forecast': risk_assessment['probability_forecast'],
                'alert_information': risk_assessment.get('alert_generated', None),
                'recommended_actions': (
                    risk_assessment.get('alert_generated', {}).get('recommended_actions', [])
                )
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Risk assessment report exported to: {file_path}")


def main():
    """Demonstrate risk assessment framework"""
    
    # Initialize risk assessment framework
    risk_framework = RiskAssessmentFramework()
    
    print("🛡️ Demonstrating Comprehensive Risk Assessment Framework")
    print("="*60)
    
    # Example location and conditions
    test_location = (40.7128, -74.0060)  # NYC coordinates for demo
    
    # Sample terrain features
    terrain_features = {
        'slope': 65.0,                    # High slope
        'fracture_density': 7.5,          # High fracture density
        'instability_index': 0.8,         # High instability
        'roughness': 0.3,                 # Moderate roughness
        'wetness_index': 12.0             # High wetness
    }
    
    # Sample environmental conditions
    env_conditions = {
        'rainfall': 85.0,                 # Heavy rainfall
        'freeze_thaw_cycles': 15,         # Moderate freeze-thaw
        'seismic_activity': 3.2,          # Moderate seismic activity
        'temperature_variation': 25.0,    # High temperature variation
        'wind_speed': 45.0                # Moderate wind
    }
    
    # Perform comprehensive risk assessment
    print("🔍 Performing comprehensive risk assessment...")
    risk_assessment = risk_framework.assess_comprehensive_risk(
        test_location, terrain_features, env_conditions, use_ml_prediction=False
    )
    
    # Display results
    print("\n📊 RISK ASSESSMENT RESULTS")
    print("="*40)
    print(f"Location: {risk_assessment['location']}")
    print(f"Final Risk Score: {risk_assessment['final_risk_score']:.3f}")
    print(f"Risk Level: {risk_assessment['risk_level_name']}")
    print(f"Confidence Score: {risk_assessment['confidence_score']:.3f}")
    
    if 'alert_generated' in risk_assessment:
        alert = risk_assessment['alert_generated']
        print(f"\n🚨 ALERT GENERATED: {alert['alert_level'].upper()}")
        print(f"Message: {alert['message']}")
    
    # Create visualizations
    viz_path = "outputs/risk_assessment_analysis.png"
    risk_framework.visualize_risk_assessment(risk_assessment, viz_path)
    
    # Export report
    report_path = "outputs/risk_assessment_report.json"
    risk_framework.export_risk_report(risk_assessment, report_path)
    
    # Demonstrate risk mapping
    print("\n🗺️ Creating spatial risk map...")
    area_bounds = (40.70, -74.02, 40.72, -73.99)  # Small area around NYC
    risk_map = risk_framework.create_risk_map(area_bounds, resolution=0.002)
    
    print(f"Risk map created with {risk_map['risk_grid'].size:,} grid points")
    print("Risk zone distribution:")
    for zone_name, zone_info in risk_map['risk_zones'].items():
        print(f"  {zone_name}: {zone_info['coverage_percent']:.1f}% coverage")
    
    return risk_framework, risk_assessment


if __name__ == "__main__":
    framework, assessment = main()