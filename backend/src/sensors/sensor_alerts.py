#!/usr/bin/env python3
"""
Sensor-Based Rockfall Alert System
=================================

This module provides rule-based rockfall prediction using sensor data analysis.
Features:
- Vibration sensor monitoring
- Environmental data analysis
- Threshold-based alerting
- Time-series pattern detection
- Predictive risk scoring
"""

import os
import numpy as np
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


class SensorDataProcessor:
    """Process and analyze sensor data for rockfall prediction"""
    
    def __init__(self, config_path: str = None):
        """Initialize sensor data processor"""
        self.load_config(config_path)
        self.setup_logging()
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.sensor_history = []
        self.alert_history = []
        
        print("Sensor-based Alert System initialized")
        print(f"   Vibration threshold: {self.config['vibration_threshold']} mm/s")
        print(f"   Temperature range: {self.config['temp_min']}°C - {self.config['temp_max']}°C")
        print(f"   Humidity threshold: {self.config['humidity_threshold']}%")
    
    def load_config(self, config_path: str = None):
        """Load sensor alert configuration"""
        # Default thresholds based on research
        self.config = {
            'vibration_threshold': 2.5,  # mm/s - threshold for rockfall risk
            'acceleration_threshold': 5.0,  # m/s² - seismic activity
            'temp_min': -10,  # °C - freeze-thaw cycles
            'temp_max': 40,   # °C - thermal expansion
            'humidity_threshold': 85,  # % - water infiltration risk
            'pressure_change_threshold': 5,  # hPa/hour - weather front
            'wind_speed_threshold': 15,  # m/s - erosion factor
            'precipitation_threshold': 20,  # mm/day - water saturation
            'alert_cooldown': 300,  # seconds between alerts
            'risk_levels': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8
            }
        }
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                self.config.update(custom_config)
    
    def setup_logging(self):
        """Setup logging for sensor alerts"""
        log_dir = Path(__file__).parent.parent.parent / "outputs" / "alerts"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"sensor_alerts_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__ + "_sensor")
        self.logger.info("Sensor alert system started")
    
    def generate_synthetic_sensor_data(self, duration_hours: int = 24, 
                                     interval_minutes: int = 5) -> pd.DataFrame:
        """
        Generate synthetic sensor data for demonstration
        
        Args:
            duration_hours: Duration of data generation in hours
            interval_minutes: Data collection interval in minutes
            
        Returns:
            DataFrame with synthetic sensor readings
        """
        # Calculate number of samples
        samples = int(duration_hours * 60 / interval_minutes)
        
        # Time series
        start_time = datetime.now() - timedelta(hours=duration_hours)
        timestamps = [start_time + timedelta(minutes=i*interval_minutes) for i in range(samples)]
        
        # Base patterns
        time_hours = np.array([(ts - start_time).total_seconds() / 3600 for ts in timestamps])
        
        # Generate realistic sensor data with patterns
        np.random.seed(42)  # For reproducible results
        
        # Vibration (seismic activity) - normally low with occasional spikes
        vibration_base = 0.5 + 0.3 * np.sin(time_hours * 2 * np.pi / 24)  # Daily cycle
        vibration_noise = np.random.normal(0, 0.2, samples)
        vibration_spikes = np.random.exponential(0.1, samples)
        vibration = vibration_base + vibration_noise + vibration_spikes
        vibration = np.clip(vibration, 0, 10)
        
        # Acceleration (seismic) - correlated with vibration but different pattern
        acceleration = vibration * 1.5 + np.random.normal(0, 0.5, samples)
        acceleration = np.clip(acceleration, 0, 15)
        
        # Temperature - daily cycle with random variations
        temp_daily = 15 + 10 * np.sin((time_hours - 6) * 2 * np.pi / 24)  # Peak at noon
        temp_noise = np.random.normal(0, 2, samples)
        temperature = temp_daily + temp_noise
        
        # Humidity - inverse correlation with temperature
        humidity_base = 60 - (temperature - 15) * 1.5
        humidity_noise = np.random.normal(0, 5, samples)
        humidity = humidity_base + humidity_noise
        humidity = np.clip(humidity, 20, 95)
        
        # Atmospheric pressure - slow variations
        pressure_base = 1013.25
        pressure_trend = np.cumsum(np.random.normal(0, 0.1, samples))
        pressure = pressure_base + pressure_trend
        pressure = np.clip(pressure, 990, 1030)
        
        # Wind speed - random with daily pattern
        wind_base = 3 + 2 * np.sin(time_hours * 2 * np.pi / 24)  # Stronger during day
        wind_noise = np.random.exponential(2, samples)
        wind_speed = wind_base + wind_noise
        wind_speed = np.clip(wind_speed, 0, 25)
        
        # Precipitation - occasional events
        precipitation = np.random.exponential(0.5, samples)
        precipitation = np.where(np.random.random(samples) > 0.9, precipitation * 10, 0)
        precipitation = np.clip(precipitation, 0, 50)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'vibration_velocity': vibration,  # mm/s
            'acceleration': acceleration,      # m/s²
            'temperature': temperature,        # °C
            'humidity': humidity,             # %
            'pressure': pressure,             # hPa
            'wind_speed': wind_speed,         # m/s
            'precipitation': precipitation,    # mm/hour
        })
        
        return df
    
    def analyze_sensor_data(self, df: pd.DataFrame) -> Dict:
        """
        Analyze sensor data and calculate risk metrics
        
        Args:
            df: DataFrame with sensor readings
            
        Returns:
            Dictionary with analysis results and risk scores
        """
        if df.empty:
            return {'risk_score': 0, 'alerts': [], 'analysis': {}}
        
        analysis = {}
        alerts = []
        risk_factors = []
        
        # Get latest readings
        latest = df.iloc[-1]
        
        # 1. Vibration Analysis
        vibration = latest['vibration_velocity']
        if vibration > self.config['vibration_threshold']:
            severity = min(vibration / self.config['vibration_threshold'], 3.0)
            risk_factors.append(severity * 0.4)  # High weight for vibration
            alerts.append({
                'type': 'vibration',
                'level': 'HIGH' if severity > 2 else 'MEDIUM',
                'value': vibration,
                'threshold': self.config['vibration_threshold'],
                'message': f"High vibration detected: {vibration:.2f} mm/s"
            })
        
        analysis['vibration'] = {
            'current': vibration,
            'threshold': self.config['vibration_threshold'],
            'status': 'ALERT' if vibration > self.config['vibration_threshold'] else 'NORMAL'
        }
        
        # 2. Seismic Activity Analysis
        acceleration = latest['acceleration']
        if acceleration > self.config['acceleration_threshold']:
            severity = min(acceleration / self.config['acceleration_threshold'], 3.0)
            risk_factors.append(severity * 0.3)  # Medium weight
            alerts.append({
                'type': 'seismic',
                'level': 'HIGH' if severity > 2 else 'MEDIUM',
                'value': acceleration,
                'threshold': self.config['acceleration_threshold'],
                'message': f"Seismic activity detected: {acceleration:.2f} m/s²"
            })
        
        # 3. Weather Pattern Analysis
        temperature = latest['temperature']
        humidity = latest['humidity']
        
        # Freeze-thaw cycles
        if temperature < self.config['temp_min'] or temperature > self.config['temp_max']:
            risk_factors.append(0.2)
            alerts.append({
                'type': 'temperature',
                'level': 'MEDIUM',
                'value': temperature,
                'message': f"Extreme temperature: {temperature:.1f}°C"
            })
        
        # High humidity (water infiltration)
        if humidity > self.config['humidity_threshold']:
            risk_factors.append(0.25)
            alerts.append({
                'type': 'humidity',
                'level': 'MEDIUM',
                'value': humidity,
                'threshold': self.config['humidity_threshold'],
                'message': f"High humidity: {humidity:.1f}%"
            })
        
        # 4. Precipitation Analysis
        precipitation = latest['precipitation']
        if precipitation > self.config['precipitation_threshold']:
            severity = min(precipitation / self.config['precipitation_threshold'], 2.0)
            risk_factors.append(severity * 0.3)
            alerts.append({
                'type': 'precipitation',
                'level': 'HIGH' if severity > 1.5 else 'MEDIUM',
                'value': precipitation,
                'threshold': self.config['precipitation_threshold'],
                'message': f"Heavy precipitation: {precipitation:.1f} mm/hour"
            })
        
        # 5. Pressure Change Analysis (if we have history)
        if len(df) > 1:
            pressure_change = latest['pressure'] - df.iloc[-2]['pressure']
            if abs(pressure_change) > self.config['pressure_change_threshold']:
                risk_factors.append(0.15)
                alerts.append({
                    'type': 'pressure',
                    'level': 'LOW',
                    'value': pressure_change,
                    'message': f"Rapid pressure change: {pressure_change:.1f} hPa"
                })
        
        # 6. Wind Analysis
        wind_speed = latest['wind_speed']
        if wind_speed > self.config['wind_speed_threshold']:
            severity = min(wind_speed / self.config['wind_speed_threshold'], 2.0)
            risk_factors.append(severity * 0.1)  # Low weight
            alerts.append({
                'type': 'wind',
                'level': 'LOW',
                'value': wind_speed,
                'threshold': self.config['wind_speed_threshold'],
                'message': f"High wind speed: {wind_speed:.1f} m/s"
            })
        
        # Calculate overall risk score
        if risk_factors:
            risk_score = min(sum(risk_factors), 1.0)  # Cap at 1.0
        else:
            risk_score = 0.0
        
        # Time-series analysis for trends
        if len(df) >= 10:
            # Trend analysis for vibration
            recent_vibration = df['vibration_velocity'].tail(10)
            vibration_trend = np.polyfit(range(len(recent_vibration)), recent_vibration, 1)[0]
            
            if vibration_trend > 0.1:  # Increasing trend
                risk_score = min(risk_score + 0.2, 1.0)
                analysis['vibration_trend'] = 'INCREASING'
            elif vibration_trend < -0.1:
                analysis['vibration_trend'] = 'DECREASING'
            else:
                analysis['vibration_trend'] = 'STABLE'
        
        # Store current data
        self.sensor_history.append({
            'timestamp': latest['timestamp'],
            'data': latest.to_dict(),
            'risk_score': risk_score,
            'alerts': len(alerts)
        })
        
        return {
            'timestamp': latest['timestamp'].isoformat(),
            'risk_score': risk_score,
            'risk_level': self.get_risk_level(risk_score),
            'alerts': alerts,
            'analysis': analysis,
            'sensor_readings': latest.to_dict()
        }
    
    def get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= self.config['risk_levels']['high']:
            return 'HIGH'
        elif risk_score >= self.config['risk_levels']['medium']:
            return 'MEDIUM'
        elif risk_score >= self.config['risk_levels']['low']:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def generate_alert(self, analysis_result: Dict):
        """Generate and log sensor-based alerts"""
        if not analysis_result['alerts']:
            return
        
        # Check alert cooldown
        current_time = datetime.now()
        if (self.alert_history and 
            (current_time - self.alert_history[-1]['timestamp']).total_seconds() < self.config['alert_cooldown']):
            return
        
        alert_data = {
            'timestamp': current_time,
            'alert_id': f"SENSOR_{int(time.time())}",
            'risk_score': analysis_result['risk_score'],
            'risk_level': analysis_result['risk_level'],
            'alerts': analysis_result['alerts'],
            'sensor_data': analysis_result['sensor_readings']
        }
        
        # Log alert
        high_alerts = [a for a in analysis_result['alerts'] if a['level'] == 'HIGH']
        if high_alerts:
            self.logger.warning(f"🚨 HIGH RISK SENSOR ALERT: {alert_data['alert_id']}")
            self.logger.warning(f"   Risk Score: {analysis_result['risk_score']:.3f}")
            self.logger.warning(f"   Risk Level: {analysis_result['risk_level']}")
            for alert in high_alerts:
                self.logger.warning(f"   - {alert['message']}")
        else:
            self.logger.info(f"⚠️ Sensor Alert: {alert_data['alert_id']} - {analysis_result['risk_level']}")
        
        # Store alert
        self.alert_history.append(alert_data)
        
        print(f"\n⚠️ SENSOR ALERT TRIGGERED! ⚠️")
        print(f"Alert ID: {alert_data['alert_id']}")
        print(f"Risk Level: {analysis_result['risk_level']}")
        print(f"Risk Score: {analysis_result['risk_score']:.3f}")
        print(f"Active Alerts: {len(analysis_result['alerts'])}")
        for alert in analysis_result['alerts']:
            print(f"  - {alert['message']}")
        print("="*50)
    
    def create_risk_visualization(self, df: pd.DataFrame, save_path: str = None) -> str:
        """Create visualization of sensor data and risk analysis"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Rockfall Risk Analysis - Sensor Data', fontsize=16)
        
        # Convert timestamp for plotting
        if 'timestamp' in df.columns:
            times = pd.to_datetime(df['timestamp'])
        else:
            times = range(len(df))
        
        # Plot 1: Vibration and Acceleration
        ax1 = axes[0, 0]
        ax1.plot(times, df['vibration_velocity'], label='Vibration (mm/s)', color='red')
        ax1.axhline(y=self.config['vibration_threshold'], color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Seismic Activity')
        ax1.set_ylabel('Vibration (mm/s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Temperature and Humidity
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        ax2.plot(times, df['temperature'], color='orange', label='Temperature (°C)')
        ax2_twin.plot(times, df['humidity'], color='blue', label='Humidity (%)')
        ax2.set_title('Weather Conditions')
        ax2.set_ylabel('Temperature (°C)', color='orange')
        ax2_twin.set_ylabel('Humidity (%)', color='blue')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Precipitation
        ax3 = axes[1, 0]
        ax3.bar(times, df['precipitation'], alpha=0.7, color='lightblue')
        ax3.axhline(y=self.config['precipitation_threshold'], color='blue', linestyle='--')
        ax3.set_title('Precipitation')
        ax3.set_ylabel('Precipitation (mm/hour)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Pressure and Wind
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        ax4.plot(times, df['pressure'], color='green', label='Pressure (hPa)')
        ax4_twin.plot(times, df['wind_speed'], color='purple', label='Wind Speed (m/s)')
        ax4.set_title('Atmospheric Conditions')
        ax4.set_ylabel('Pressure (hPa)', color='green')
        ax4_twin.set_ylabel('Wind Speed (m/s)', color='purple')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Risk Score Over Time
        ax5 = axes[2, 0]
        if self.sensor_history:
            history_times = [h['timestamp'] for h in self.sensor_history]
            risk_scores = [h['risk_score'] for h in self.sensor_history]
            ax5.plot(history_times, risk_scores, color='red', linewidth=2)
            ax5.fill_between(history_times, risk_scores, alpha=0.3, color='red')
            ax5.axhline(y=self.config['risk_levels']['high'], color='red', linestyle='--', label='High Risk')
            ax5.axhline(y=self.config['risk_levels']['medium'], color='orange', linestyle='--', label='Medium Risk')
            ax5.axhline(y=self.config['risk_levels']['low'], color='yellow', linestyle='--', label='Low Risk')
        ax5.set_title('Risk Score Timeline')
        ax5.set_ylabel('Risk Score')
        ax5.set_ylim(0, 1)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Alert Summary
        ax6 = axes[2, 1]
        if self.alert_history:
            alert_times = [a['timestamp'] for a in self.alert_history]
            alert_levels = [a['risk_level'] for a in self.alert_history]
            level_colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow', 'MINIMAL': 'green'}
            colors = [level_colors.get(level, 'gray') for level in alert_levels]
            ax6.scatter(alert_times, range(len(alert_times)), c=colors, s=100)
            ax6.set_title('Alert History')
            ax6.set_ylabel('Alert Number')
        else:
            ax6.text(0.5, 0.5, 'No alerts yet', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Alert History')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if not save_path:
            output_dir = Path(__file__).parent.parent.parent / "outputs"
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / f"sensor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def run_continuous_monitoring(self, duration_minutes: int = 60, 
                                interval_seconds: int = 30):
        """Run continuous sensor monitoring simulation"""
        print(f"\n🔄 Starting continuous sensor monitoring...")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Interval: {interval_seconds} seconds")
        print("="*50)
        
        start_time = time.time()
        iteration = 0
        
        try:
            while time.time() - start_time < duration_minutes * 60:
                iteration += 1
                
                # Generate fresh sensor data (simulate real-time readings)
                df = self.generate_synthetic_sensor_data(duration_hours=0.1, interval_minutes=1)
                
                # Analyze current data
                analysis_result = self.analyze_sensor_data(df)
                
                # Check for alerts
                if analysis_result['alerts']:
                    self.generate_alert(analysis_result)
                
                # Print status update
                if iteration % 10 == 0:  # Every 10 iterations
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Risk: {analysis_result['risk_level']} "
                          f"(Score: {analysis_result['risk_score']:.3f}) "
                          f"Alerts: {len(analysis_result['alerts'])}")
                
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        # Generate final report
        print(f"\n📊 Monitoring Summary:")
        print(f"   Duration: {(time.time() - start_time)/60:.1f} minutes")
        print(f"   Iterations: {iteration}")
        print(f"   Total alerts: {len(self.alert_history)}")
        
        if self.sensor_history:
            avg_risk = np.mean([h['risk_score'] for h in self.sensor_history])
            max_risk = max([h['risk_score'] for h in self.sensor_history])
            print(f"   Average risk score: {avg_risk:.3f}")
            print(f"   Maximum risk score: {max_risk:.3f}")
        
        return self.alert_history


def main():
    """Main function for testing sensor alert system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sensor-based Rockfall Alert System')
    parser.add_argument('--duration', type=int, default=24,
                       help='Data duration in hours for analysis')
    parser.add_argument('--monitor', type=int, default=0,
                       help='Run continuous monitoring for N minutes (0 to disable)')
    parser.add_argument('--save-plot', action='store_true',
                       help='Save analysis plots')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = SensorDataProcessor()
        
        # Generate sensor data
        print(f"Generating {args.duration} hours of synthetic sensor data...")
        df = processor.generate_synthetic_sensor_data(duration_hours=args.duration)
        
        # Analyze data
        print("Analyzing sensor data for rockfall risk...")
        analysis_result = processor.analyze_sensor_data(df)
        
        # Print results
        print(f"\n📊 Analysis Results:")
        print(f"   Risk Level: {analysis_result['risk_level']}")
        print(f"   Risk Score: {analysis_result['risk_score']:.3f}")
        print(f"   Active Alerts: {len(analysis_result['alerts'])}")
        
        for alert in analysis_result['alerts']:
            print(f"   - {alert['level']}: {alert['message']}")
        
        # Generate alerts if needed
        if analysis_result['alerts']:
            processor.generate_alert(analysis_result)
        
        # Create visualization
        if args.save_plot:
            plot_path = processor.create_risk_visualization(df)
            print(f"   Visualization saved: {plot_path}")
        
        # Run continuous monitoring if requested
        if args.monitor > 0:
            processor.run_continuous_monitoring(duration_minutes=args.monitor)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())