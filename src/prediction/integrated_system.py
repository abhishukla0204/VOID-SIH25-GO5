"""
Integrated Rockfall Monitoring System
====================================

This module integrates the proactive prediction system with the reactive YOLO detection system
to provide comprehensive rockfall monitoring combining:

PROACTIVE MONITORING:
- ML-based risk prediction using terrain and environmental data
- Real-time risk assessment and probability forecasting
- Predictive alerts and risk zone mapping

REACTIVE MONITORING:
- YOLOv8-based real-time rockfall detection from video streams
- Instant event detection and alert generation
- Visual confirmation of rockfall events

INTEGRATION FEATURES:
- Unified dashboard and monitoring interface
- Combined alert system with both predictive and reactive alerts
- Historical analysis combining predictions and actual detections
- Enhanced decision support through multi-modal data fusion
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import threading
import time
import queue
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Import existing system components
import sys
sys.path.append('src')
sys.path.append('src/prediction')
sys.path.append('src/detection')
sys.path.append('src/sensors')
sys.path.append('src/dem_analysis')

try:
    from detection.realtime_detector import RockfallDetector
    from sensors.sensor_alerts import SensorDataProcessor
    from dem_analysis.dem_processor import DEMAnalyzer
    from prediction.ml_models import RockfallRiskPredictor
    from prediction.risk_assessment import RiskAssessmentFramework
    from prediction.terrain_feature_extractor import DEMTerrainExtractor
    from prediction.synthetic_data_generator import SyntheticDataGenerator
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some modules not available: {e}")
    MODULES_AVAILABLE = False

class IntegratedRockfallSystem:
    """Main class for integrated proactive and reactive rockfall monitoring"""
    
    def __init__(self):
        self.prediction_system = None
        self.detection_system = None
        self.sensor_system = None
        self.dem_analyzer = None
        self.risk_framework = None
        
        # System state
        self.is_running = False
        self.monitoring_threads = []
        self.alert_queue = queue.Queue()
        self.system_status = {}
        
        # Data storage
        self.prediction_history = []
        self.detection_history = []
        self.combined_alerts = []
        
        # Configuration
        self.config = {
            'prediction_interval': 300,    # 5 minutes between predictions
            'detection_active': True,      # Real-time detection enabled
            'sensor_interval': 60,         # 1 minute between sensor readings
            'risk_threshold': 0.6,         # Risk threshold for alerts
            'alert_cooldown': 1800,        # 30 minutes between similar alerts
            'location': (40.7128, -74.0060),  # Default location
            'dem_file': 'data/DEM/Bingham_Canyon_Mine.tif'
        }
        
        print("🔗 Integrated Rockfall Monitoring System initialized")
    
    def initialize_systems(self) -> bool:
        """Initialize all subsystems"""
        
        if not MODULES_AVAILABLE:
            print("❌ Required modules not available")
            return False
        
        try:
            print("🚀 Initializing integrated monitoring systems...")
            
            # Initialize prediction system
            print("🧠 Initializing ML prediction system...")
            self.prediction_system = RockfallRiskPredictor()
            
            # Load pre-trained models if available
            try:
                import joblib
                model_path = "outputs/models"
                if os.path.exists(f"{model_path}/xgboost_model.joblib"):
                    self.prediction_system.models['xgboost'] = joblib.load(f"{model_path}/xgboost_model.joblib")
                    self.prediction_system.scalers['main'] = joblib.load(f"{model_path}/main_scaler.joblib")
                    metadata = joblib.load(f"{model_path}/model_metadata.joblib")
                    self.prediction_system.feature_names = metadata['feature_names']
                    self.prediction_system.model_performance = metadata['model_performance']
                    print("✅ Pre-trained prediction models loaded")
                else:
                    print("⚠️ No pre-trained models found, will use default predictions")
            except Exception as e:
                print(f"⚠️ Could not load pre-trained models: {e}")
            
            # Initialize risk assessment framework
            print("🛡️ Initializing risk assessment framework...")
            self.risk_framework = RiskAssessmentFramework(self.prediction_system)
            
            # Initialize YOLO detection system
            print("👁️ Initializing YOLO detection system...")
            self.detection_system = RockfallDetector()
            
            # Initialize sensor system
            print("📡 Initializing sensor monitoring system...")
            self.sensor_system = SensorDataProcessor()
            
            # Initialize DEM analyzer
            print("🗻 Initializing DEM analysis system...")
            self.dem_analyzer = DEMAnalyzer()
            
            # Load DEM data if available
            try:
                if os.path.exists(self.config['dem_file']):
                    self.dem_analyzer.load_dem(self.config['dem_file'])
                    print("✅ DEM data loaded successfully")
                else:
                    print("⚠️ DEM file not found, using synthetic terrain data")
            except Exception as e:
                print(f"⚠️ Could not load DEM: {e}")
            
            self.system_status = {
                'prediction_system': True,
                'detection_system': True,
                'sensor_system': True,
                'dem_analyzer': True,
                'risk_framework': True,
                'initialization_time': datetime.now()
            }
            
            print("✅ All systems initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
    
    def start_integrated_monitoring(self):
        """Start integrated monitoring with all systems"""
        
        if not self.system_status.get('prediction_system', False):
            print("❌ Systems not initialized. Call initialize_systems() first.")
            return
        
        print("🚀 Starting integrated rockfall monitoring...")
        self.is_running = True
        
        # Start prediction monitoring thread
        prediction_thread = threading.Thread(
            target=self._prediction_monitoring_loop,
            name="PredictionMonitoring",
            daemon=True
        )
        prediction_thread.start()
        self.monitoring_threads.append(prediction_thread)
        
        # Start detection monitoring thread (if detection is active)
        if self.config['detection_active']:
            detection_thread = threading.Thread(
                target=self._detection_monitoring_loop,
                name="DetectionMonitoring", 
                daemon=True
            )
            detection_thread.start()
            self.monitoring_threads.append(detection_thread)
        
        # Start sensor monitoring thread
        sensor_thread = threading.Thread(
            target=self._sensor_monitoring_loop,
            name="SensorMonitoring",
            daemon=True
        )
        sensor_thread.start()
        self.monitoring_threads.append(sensor_thread)
        
        # Start alert processing thread
        alert_thread = threading.Thread(
            target=self._alert_processing_loop,
            name="AlertProcessing",
            daemon=True
        )
        alert_thread.start()
        self.monitoring_threads.append(alert_thread)
        
        print(f"✅ Integrated monitoring started with {len(self.monitoring_threads)} active threads")
        
        # Print status summary
        self._print_system_status()
    
    def _prediction_monitoring_loop(self):
        """Main loop for predictive risk assessment"""
        
        print("🧠 Prediction monitoring thread started")
        
        while self.is_running:
            try:
                # Get current environmental conditions
                env_conditions = self._get_current_environmental_conditions()
                
                # Get terrain features for current location
                terrain_features = self._get_terrain_features()
                
                # Perform comprehensive risk assessment
                risk_assessment = self.risk_framework.assess_comprehensive_risk(
                    location=self.config['location'],
                    terrain_features=terrain_features,
                    env_conditions=env_conditions,
                    use_ml_prediction=True
                )
                
                # Store prediction result
                self.prediction_history.append(risk_assessment)
                
                # Check for alerts
                if risk_assessment['final_risk_score'] >= self.config['risk_threshold']:
                    alert_data = {
                        'type': 'prediction',
                        'timestamp': datetime.now(),
                        'location': self.config['location'],
                        'risk_score': risk_assessment['final_risk_score'],
                        'risk_level': risk_assessment['risk_level_name'],
                        'alert_data': risk_assessment,
                        'message': f"Predictive Alert: {risk_assessment['risk_level_name']} risk detected (Score: {risk_assessment['final_risk_score']:.3f})"
                    }
                    self.alert_queue.put(alert_data)
                
                print(f"📊 Prediction update: Risk={risk_assessment['final_risk_score']:.3f}, Level={risk_assessment['risk_level_name']}")
                
                # Wait for next prediction cycle
                time.sleep(self.config['prediction_interval'])
                
            except Exception as e:
                print(f"❌ Prediction monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _detection_monitoring_loop(self):
        """Main loop for real-time YOLO detection"""
        
        print("👁️ Detection monitoring thread started")
        
        try:
            # Start video detection (using default camera)
            detections = self.detection_system.process_video_stream(
                source=0,  # Default camera
                output_path="outputs/detection_stream.mp4",
                save_video=False  # Don't save to reduce overhead
            )
            
            for detection in detections:
                if not self.is_running:
                    break
                
                # Store detection result
                detection_record = {
                    'timestamp': datetime.now(),
                    'location': self.config['location'],  # Would get from GPS in real system
                    'confidence': detection.get('confidence', 0.0),
                    'detection_data': detection
                }
                self.detection_history.append(detection_record)
                
                # Generate immediate alert for rockfall detection
                if detection.get('confidence', 0.0) > 0.5:  # Confidence threshold
                    alert_data = {
                        'type': 'detection',
                        'timestamp': datetime.now(),
                        'location': self.config['location'],
                        'confidence': detection['confidence'],
                        'alert_data': detection,
                        'message': f"🚨 ROCKFALL DETECTED! Confidence: {detection['confidence']:.2f}"
                    }
                    self.alert_queue.put(alert_data)
                    
                    print(f"🚨 ROCKFALL DETECTED: Confidence={detection['confidence']:.2f}")
                
        except Exception as e:
            print(f"❌ Detection monitoring error: {e}")
    
    def _sensor_monitoring_loop(self):
        """Main loop for sensor data monitoring"""
        
        print("📡 Sensor monitoring thread started")
        
        while self.is_running:
            try:
                # Generate and analyze sensor data
                sensor_data = self.sensor_system.generate_sensor_data(duration_hours=1)
                risk_assessment = self.sensor_system.assess_risk(sensor_data)
                
                # Check for sensor-based alerts
                if risk_assessment['overall_risk_score'] > 0.7:
                    alert_data = {
                        'type': 'sensor',
                        'timestamp': datetime.now(),
                        'location': self.config['location'],
                        'risk_score': risk_assessment['overall_risk_score'],
                        'alert_data': risk_assessment,
                        'message': f"Sensor Alert: High risk detected (Score: {risk_assessment['overall_risk_score']:.3f})"
                    }
                    self.alert_queue.put(alert_data)
                
                print(f"📡 Sensor update: Risk={risk_assessment['overall_risk_score']:.3f}")
                
                time.sleep(self.config['sensor_interval'])
                
            except Exception as e:
                print(f"❌ Sensor monitoring error: {e}")
                time.sleep(60)
    
    def _alert_processing_loop(self):
        """Main loop for processing and managing alerts"""
        
        print("🚨 Alert processing thread started")
        recent_alerts = {}
        
        while self.is_running:
            try:
                # Get alert from queue (with timeout)
                try:
                    alert = self.alert_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check alert cooldown to prevent spam
                alert_key = f"{alert['type']}_{alert['location']}"
                current_time = datetime.now()
                
                if alert_key in recent_alerts:
                    time_since_last = (current_time - recent_alerts[alert_key]).seconds
                    if time_since_last < self.config['alert_cooldown']:
                        continue  # Skip this alert due to cooldown
                
                # Process and store alert
                recent_alerts[alert_key] = current_time
                self.combined_alerts.append(alert)
                
                # Log alert
                self._log_alert(alert)
                
                # Take appropriate actions based on alert type and severity
                self._handle_alert(alert)
                
            except Exception as e:
                print(f"❌ Alert processing error: {e}")
    
    def _get_current_environmental_conditions(self) -> Dict[str, float]:
        """Get current environmental conditions (real or simulated)"""
        
        # In a real system, this would connect to weather APIs and sensors
        # For demonstration, we'll use simulated data with some realistic patterns
        
        current_hour = datetime.now().hour
        current_month = datetime.now().month
        
        # Simulate realistic environmental conditions
        conditions = {
            'rainfall': max(0, np.random.exponential(20) * (1 + 0.5 * np.sin(current_month * np.pi / 6))),
            'freeze_thaw_cycles': max(0, np.random.poisson(5) if current_month in [11, 12, 1, 2, 3] else np.random.poisson(1)),
            'seismic_activity': np.random.exponential(1.5),
            'temperature_variation': abs(np.random.normal(15, 8)),
            'wind_speed': np.random.gamma(2, 15),
            'temperature': 20 + 15 * np.sin((current_month - 3) * np.pi / 6) + np.random.normal(0, 5),
            'humidity': np.random.beta(6, 4) * 100
        }
        
        # Clip values to realistic ranges
        conditions['rainfall'] = min(conditions['rainfall'], 200)
        conditions['seismic_activity'] = min(conditions['seismic_activity'], 7)
        conditions['wind_speed'] = min(conditions['wind_speed'], 120)
        conditions['temperature'] = max(-30, min(conditions['temperature'], 45))
        
        return conditions
    
    def _get_terrain_features(self) -> Dict[str, float]:
        """Get terrain features for current location"""
        
        # In a real system, this would extract features from DEM at specific location
        # For demonstration, we'll use representative values for the area
        
        if self.dem_analyzer and hasattr(self.dem_analyzer, 'dem_data'):
            # Use actual DEM data if available
            try:
                # Get center point of DEM for demonstration
                rows, cols = self.dem_analyzer.dem_data.shape
                center_row, center_col = rows // 2, cols // 2
                
                # Extract features at center point (simplified)
                features = {
                    'slope': float(self.dem_analyzer.slope[center_row, center_col]) if hasattr(self.dem_analyzer, 'slope') else 45.0,
                    'elevation': float(self.dem_analyzer.dem_data[center_row, center_col]),
                    'fracture_density': np.random.uniform(3, 8),  # Would come from geological survey
                    'instability_index': np.random.uniform(0.4, 0.8),
                    'roughness': np.random.uniform(0.2, 0.6),
                    'slope_variability': np.random.uniform(5, 25),
                    'wetness_index': np.random.uniform(5, 15)
                }
                
                # Handle NaN values
                for key, value in features.items():
                    if np.isnan(value):
                        features[key] = {
                            'slope': 45.0,
                            'elevation': 1500.0,
                            'fracture_density': 5.0,
                            'instability_index': 0.6,
                            'roughness': 0.4,
                            'slope_variability': 15.0,
                            'wetness_index': 10.0
                        }[key]
                
                return features
                
            except Exception as e:
                print(f"⚠️ Could not extract DEM features: {e}")
        
        # Default terrain features for demonstration
        return {
            'slope': 65.0,                    # High slope
            'elevation': 1200.0,              # Moderate elevation
            'fracture_density': 6.5,          # Moderate-high fracture density
            'instability_index': 0.7,         # High instability
            'roughness': 0.3,                 # Moderate roughness
            'slope_variability': 18.0,        # Moderate variability
            'wetness_index': 12.0             # High wetness
        }
    
    def _log_alert(self, alert: Dict):
        """Log alert to file and console"""
        
        timestamp = alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        alert_type = alert['type'].upper()
        message = alert['message']
        
        # Console output
        print(f"🚨 [{timestamp}] {alert_type} ALERT: {message}")
        
        # File logging
        os.makedirs('outputs/alerts', exist_ok=True)
        log_file = f"outputs/alerts/integrated_alerts_{datetime.now().strftime('%Y%m%d')}.log"
        
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {alert_type}: {message}\n")
            f.write(f"  Location: {alert['location']}\n")
            if 'risk_score' in alert:
                f.write(f"  Risk Score: {alert['risk_score']:.3f}\n")
            if 'confidence' in alert:
                f.write(f"  Confidence: {alert['confidence']:.3f}\n")
            f.write("  ---\n")
    
    def _handle_alert(self, alert: Dict):
        """Handle alert based on type and severity"""
        
        alert_type = alert['type']
        
        if alert_type == 'detection':
            # Immediate rockfall detected - highest priority
            self._handle_detection_alert(alert)
        elif alert_type == 'prediction':
            # Predictive alert - prepare for potential event
            self._handle_prediction_alert(alert)
        elif alert_type == 'sensor':
            # Sensor-based alert - environmental trigger
            self._handle_sensor_alert(alert)
    
    def _handle_detection_alert(self, alert: Dict):
        """Handle immediate rockfall detection alert"""
        
        # This would trigger immediate response procedures
        confidence = alert.get('confidence', 0.0)
        
        if confidence > 0.8:
            # High confidence detection - immediate evacuation
            print("🚨 HIGH CONFIDENCE ROCKFALL - IMMEDIATE EVACUATION REQUIRED!")
            # Would trigger emergency protocols
        elif confidence > 0.6:
            # Medium confidence - prepare for evacuation
            print("⚠️ MEDIUM CONFIDENCE ROCKFALL - PREPARE FOR EVACUATION")
        else:
            # Low confidence - increased monitoring
            print("🔶 LOW CONFIDENCE ROCKFALL - INCREASE MONITORING")
    
    def _handle_prediction_alert(self, alert: Dict):
        """Handle predictive risk alert"""
        
        risk_score = alert.get('risk_score', 0.0)
        
        if risk_score > 0.8:
            print("🚨 CRITICAL RISK PREDICTION - EVACUATE AREA")
        elif risk_score > 0.6:
            print("⚠️ HIGH RISK PREDICTION - PREPARE FOR EVACUATION")
        else:
            print("🔶 MODERATE RISK PREDICTION - INCREASE MONITORING")
    
    def _handle_sensor_alert(self, alert: Dict):
        """Handle sensor-based environmental alert"""
        
        risk_score = alert.get('risk_score', 0.0)
        
        if risk_score > 0.8:
            print("📡 CRITICAL SENSOR READINGS - ENVIRONMENTAL TRIGGER DETECTED")
        else:
            print("📡 ELEVATED SENSOR READINGS - MONITOR CONDITIONS")
    
    def _print_system_status(self):
        """Print current system status"""
        
        print("\n" + "="*60)
        print("🔗 INTEGRATED ROCKFALL MONITORING SYSTEM STATUS")
        print("="*60)
        print(f"🟢 System Running: {self.is_running}")
        print(f"📊 Active Threads: {len(self.monitoring_threads)}")
        print(f"📍 Location: {self.config['location']}")
        print(f"⏱️ Prediction Interval: {self.config['prediction_interval']}s")
        print(f"👁️ Detection Active: {self.config['detection_active']}")
        print(f"⚠️ Risk Threshold: {self.config['risk_threshold']}")
        print(f"📈 Predictions Logged: {len(self.prediction_history)}")
        print(f"👁️ Detections Logged: {len(self.detection_history)}")
        print(f"🚨 Alerts Generated: {len(self.combined_alerts)}")
        print("="*60)
    
    def stop_monitoring(self):
        """Stop all monitoring threads"""
        
        print("🛑 Stopping integrated monitoring...")
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.monitoring_threads:
            thread.join(timeout=5.0)
        
        print("✅ Monitoring stopped")
    
    def get_system_summary(self) -> Dict:
        """Get comprehensive system summary"""
        
        current_time = datetime.now()
        
        # Calculate recent activity
        recent_predictions = [p for p in self.prediction_history 
                            if (current_time - p['timestamp']).seconds < 3600]
        recent_detections = [d for d in self.detection_history 
                           if (current_time - d['timestamp']).seconds < 3600]
        recent_alerts = [a for a in self.combined_alerts 
                        if (current_time - a['timestamp']).seconds < 3600]
        
        # Calculate average risk score
        avg_risk = np.mean([p['final_risk_score'] for p in recent_predictions]) if recent_predictions else 0.0
        
        summary = {
            'system_status': {
                'is_running': self.is_running,
                'uptime_minutes': (current_time - self.system_status.get('initialization_time', current_time)).seconds // 60,
                'active_threads': len(self.monitoring_threads)
            },
            'recent_activity': {
                'predictions_last_hour': len(recent_predictions),
                'detections_last_hour': len(recent_detections),
                'alerts_last_hour': len(recent_alerts),
                'average_risk_score': avg_risk
            },
            'cumulative_stats': {
                'total_predictions': len(self.prediction_history),
                'total_detections': len(self.detection_history),
                'total_alerts': len(self.combined_alerts)
            },
            'current_status': {
                'location': self.config['location'],
                'last_prediction': self.prediction_history[-1] if self.prediction_history else None,
                'last_detection': self.detection_history[-1] if self.detection_history else None,
                'last_alert': self.combined_alerts[-1] if self.combined_alerts else None
            }
        }
        
        return summary
    
    def run_demonstration(self, duration_minutes: int = 5):
        """Run a demonstration of the integrated system"""
        
        print(f"🎬 Starting {duration_minutes}-minute demonstration...")
        
        if not self.initialize_systems():
            print("❌ Failed to initialize systems")
            return
        
        # Start monitoring
        self.start_integrated_monitoring()
        
        try:
            # Run for specified duration
            start_time = time.time()
            while (time.time() - start_time) < (duration_minutes * 60):
                time.sleep(30)  # Update every 30 seconds
                
                # Print periodic status
                elapsed_minutes = (time.time() - start_time) / 60
                print(f"\n⏱️ Demo Progress: {elapsed_minutes:.1f}/{duration_minutes} minutes")
                
                # Show recent activity
                summary = self.get_system_summary()
                recent = summary['recent_activity']
                print(f"📊 Recent Activity: {recent['predictions_last_hour']} predictions, "
                      f"{recent['detections_last_hour']} detections, {recent['alerts_last_hour']} alerts")
                print(f"⚠️ Current Risk Level: {recent['average_risk_score']:.3f}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Demo interrupted by user")
        
        finally:
            # Stop monitoring
            self.stop_monitoring()
            
            # Generate final report
            self._generate_demo_report()
    
    def _generate_demo_report(self):
        """Generate demonstration report"""
        
        summary = self.get_system_summary()
        
        print("\n" + "="*60)
        print("📊 DEMONSTRATION REPORT")
        print("="*60)
        
        # System performance
        print(f"⏱️ System Uptime: {summary['system_status']['uptime_minutes']} minutes")
        print(f"📈 Total Predictions: {summary['cumulative_stats']['total_predictions']}")
        print(f"👁️ Total Detections: {summary['cumulative_stats']['total_detections']}")
        print(f"🚨 Total Alerts: {summary['cumulative_stats']['total_alerts']}")
        
        # Alert breakdown
        alert_types = {}
        for alert in self.combined_alerts:
            alert_type = alert['type']
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        
        print(f"\n🚨 Alert Breakdown:")
        for alert_type, count in alert_types.items():
            print(f"  {alert_type.title()}: {count}")
        
        # Save detailed report
        report_path = f"outputs/integrated_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        detailed_report = {
            'demonstration_summary': summary,
            'prediction_history': [
                {**p, 'timestamp': p['timestamp'].isoformat()} 
                for p in self.prediction_history
            ],
            'detection_history': [
                {**d, 'timestamp': d['timestamp'].isoformat()} 
                for d in self.detection_history
            ],
            'combined_alerts': [
                {**a, 'timestamp': a['timestamp'].isoformat()} 
                for a in self.combined_alerts
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved to: {report_path}")
        print("="*60)


def main():
    """Demonstrate integrated rockfall monitoring system"""
    
    print("🔗 Integrated Rockfall Monitoring System Demo")
    print("=" * 50)
    
    # Create integrated system
    system = IntegratedRockfallSystem()
    
    # Run demonstration
    system.run_demonstration(duration_minutes=2)  # Short demo
    
    return system


if __name__ == "__main__":
    integrated_system = main()