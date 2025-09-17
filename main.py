#!/usr/bin/env python3
"""
Integrated Rockfall Detection and Prediction System
==================================================

This script integrates all components of the rockfall detection system:
- YOLOv8 video detection
- Sensor-based risk assessment
- DEM terrain analysis
- Real-time alerting
- Web dashboard interface

Usage:
    python main.py --mode dashboard    # Launch web dashboard
    python main.py --mode train       # Train YOLOv8 model
    python main.py --mode detect      # Run video detection
    python main.py --mode sensor      # Run sensor monitoring
    python main.py --mode dem         # Analyze DEM files
    python main.py --mode all         # Run complete system
"""

import os
import sys
import argparse
import subprocess
import threading
import time
import json
from datetime import datetime
from pathlib import Path
import logging

# Add src directories to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "src" / "training"))
sys.path.append(str(Path(__file__).parent / "src" / "detection"))
sys.path.append(str(Path(__file__).parent / "src" / "sensors"))
sys.path.append(str(Path(__file__).parent / "src" / "dem_analysis"))
sys.path.append(str(Path(__file__).parent / "src" / "dashboard"))


class RockfallSystem:
    """Integrated rockfall detection and prediction system"""
    
    def __init__(self):
        """Initialize the system"""
        self.setup_logging()
        self.project_root = Path(__file__).parent
        self.running_processes = {}
        
        print("Initializing Rockfall Detection System")
        print("="*60)
        
    def setup_logging(self):
        """Setup system logging"""
        log_dir = Path(__file__).parent / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Rockfall Detection System initialized")
    
    def train_model(self, **kwargs):
        """Train YOLOv8 model"""
        print("\n🚀 Training YOLOv8 Model")
        print("-" * 40)
        
        try:
            from src.training.train_yolo import RockfallTrainer
            
            # Default training configuration
            config = {
                'model_size': kwargs.get('model_size', 'yolov8n.pt'),
                'epochs': kwargs.get('epochs', 50),
                'batch_size': kwargs.get('batch_size', 16),
                'device': kwargs.get('device', 'cpu')
            }
            
            trainer = RockfallTrainer()
            trainer.config.update(config)
            
            # Train model
            results, model = trainer.train_model()
            
            # Evaluate and export
            eval_results = trainer.evaluate_model(model)
            export_paths = trainer.export_model(model)
            
            print(f"✅ Training completed successfully!")
            print(f"📁 Model saved in: {trainer.models_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            print(f"❌ Training failed: {e}")
            return False
    
    def run_video_detection(self, **kwargs):
        """Run video detection system"""
        print("\n🎥 Starting Video Detection")
        print("-" * 40)
        
        try:
            from src.detection.realtime_detector import RockfallDetector
            
            detector = RockfallDetector(
                model_path=kwargs.get('model_path'),
                confidence=kwargs.get('confidence', 0.5),
                device=kwargs.get('device', 'cpu'),
                alert_threshold=kwargs.get('alert_threshold', 3)
            )
            
            # Start detection
            detector.process_video_stream(
                source=kwargs.get('source', 0),
                display=kwargs.get('display', True),
                save_video=kwargs.get('save_video', False)
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Video detection failed: {e}")
            print(f"❌ Video detection failed: {e}")
            return False
    
    def run_sensor_monitoring(self, **kwargs):
        """Run sensor monitoring system"""
        print("\n📊 Starting Sensor Monitoring")
        print("-" * 40)
        
        try:
            from src.sensors.sensor_alerts import SensorDataProcessor
            
            processor = SensorDataProcessor()
            
            # Generate and analyze sensor data
            duration = kwargs.get('duration', 24)
            df = processor.generate_synthetic_sensor_data(duration_hours=duration)
            
            # Analyze for risks
            analysis_result = processor.analyze_sensor_data(df)
            
            # Generate alerts if needed
            if analysis_result['alerts']:
                processor.generate_alert(analysis_result)
            
            # Create visualization
            plot_path = processor.create_risk_visualization(df)
            print(f"📊 Sensor analysis plot saved: {plot_path}")
            
            # Run continuous monitoring if requested
            if kwargs.get('continuous', False):
                processor.run_continuous_monitoring(
                    duration_minutes=kwargs.get('monitor_duration', 60)
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sensor monitoring failed: {e}")
            print(f"❌ Sensor monitoring failed: {e}")
            return False
    
    def analyze_dem(self, dem_path: str, **kwargs):
        """Analyze DEM file for risk assessment"""
        print(f"\n🗺️ Analyzing DEM: {Path(dem_path).name}")
        print("-" * 40)
        
        try:
            from src.dem_analysis.dem_processor import DEMAnalyzer
            
            analyzer = DEMAnalyzer(dem_path)
            
            # Perform risk assessment
            risk_results = analyzer.assess_rockfall_risk()
            
            # Identify critical zones
            critical_zones = analyzer.identify_critical_zones(
                risk_results, 
                min_area=kwargs.get('min_zone_size', 100)
            )
            
            # Create visualization
            plot_path = analyzer.create_risk_visualization(risk_results, critical_zones)
            print(f"📊 Risk analysis plot saved: {plot_path}")
            
            # Generate report
            report = analyzer.generate_report(risk_results, critical_zones)
            
            # Save report
            report_path = self.project_root / "outputs" / f"risk_report_{Path(dem_path).stem}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📋 Risk report saved: {report_path}")
            print(f"📈 High risk areas: {report['risk_distribution']['high_risk']['percentage']:.1f}%")
            print(f"🎯 Critical zones: {len(critical_zones)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"DEM analysis failed: {e}")
            print(f"❌ DEM analysis failed: {e}")
            return False
    
    def launch_dashboard(self, **kwargs):
        """Launch Streamlit dashboard"""
        print("\n🖥️ Launching Web Dashboard")
        print("-" * 40)
        
        try:
            dashboard_script = self.project_root / "src" / "dashboard" / "app.py"
            
            # Launch Streamlit app
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                str(dashboard_script),
                "--server.port", str(kwargs.get('port', 8501)),
                "--server.address", kwargs.get('host', 'localhost')
            ]
            
            print(f"🚀 Starting dashboard at http://{kwargs.get('host', 'localhost')}:{kwargs.get('port', 8501)}")
            print("Press Ctrl+C to stop the dashboard")
            
            # Run in subprocess
            process = subprocess.run(cmd, check=True)
            
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
            return True
        except Exception as e:
            self.logger.error(f"Dashboard launch failed: {e}")
            print(f"❌ Dashboard launch failed: {e}")
            return False
    
    def run_complete_system(self, **kwargs):
        """Run the complete integrated system"""
        print("\n🌟 Starting Complete Rockfall Detection System")
        print("="*60)
        
        # Launch dashboard in background
        dashboard_thread = threading.Thread(
            target=self.launch_dashboard,
            kwargs={'port': kwargs.get('port', 8501)}
        )
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        print("✅ Dashboard started in background")
        
        # Run DEM analysis for all available files
        dem_dir = self.project_root / "data" / "DEM"
        if dem_dir.exists():
            dem_files = list(dem_dir.glob("*.tif"))
            for dem_file in dem_files:
                print(f"\n📍 Analyzing {dem_file.name}...")
                self.analyze_dem(str(dem_file))
        
        # Run sensor monitoring
        print("\n📊 Running sensor analysis...")
        self.run_sensor_monitoring(duration=24, continuous=False)
        
        # Show system status
        print("\n" + "="*60)
        print("🎉 System Status:")
        print("✅ Dashboard: Running at http://localhost:8501")
        print("✅ DEM Analysis: Completed")
        print("✅ Sensor Monitoring: Completed")
        print("⚠️ Video Detection: Available (start from dashboard)")
        print("⚠️ Model Training: Available (use --mode train)")
        print("="*60)
        print("\n🔗 Access the dashboard to monitor the system:")
        print("   http://localhost:8501")
        print("\nPress Ctrl+C to stop the system")
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 System stopped by user")
        
        return True
    
    def create_sample_data(self):
        """Create sample data files for testing"""
        print("\n📂 Creating sample data files")
        print("-" * 40)
        
        sample_dir = self.project_root / "sample_data"
        
        # Create sensor data
        print("📊 Generating sample sensor data...")
        try:
            from src.sensors.sensor_alerts import SensorDataProcessor
            processor = SensorDataProcessor()
            
            # Generate 7 days of data
            df = processor.generate_synthetic_sensor_data(duration_hours=168)
            
            sensor_file = sample_dir / "sensor_data" / "sample_sensor_data.csv"
            sensor_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(sensor_file, index=False)
            
            print(f"✅ Sample sensor data saved: {sensor_file}")
            
        except Exception as e:
            print(f"❌ Failed to create sensor data: {e}")
        
        # Create weather data
        weather_file = sample_dir / "weather_data" / "sample_weather.json"
        weather_file.parent.mkdir(parents=True, exist_ok=True)
        
        sample_weather = {
            "timestamp": datetime.now().isoformat(),
            "temperature": 15.5,
            "humidity": 72.3,
            "pressure": 1013.2,
            "wind_speed": 5.2,
            "precipitation": 0.0,
            "weather_condition": "partly_cloudy"
        }
        
        with open(weather_file, 'w') as f:
            json.dump(sample_weather, f, indent=2)
        
        print(f"✅ Sample weather data saved: {weather_file}")
        
        print("✅ Sample data creation completed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Rockfall Detection and Prediction System')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'detect', 'sensor', 'dem', 'dashboard', 'all', 'setup'],
                       help='System operation mode')
    
    # Training arguments
    parser.add_argument('--model-size', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    
    # Detection arguments
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model for detection')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (webcam=0, file path, or URL)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold')
    
    # DEM arguments
    parser.add_argument('--dem-path', type=str, default=None,
                       help='Path to DEM file for analysis')
    
    # Sensor arguments
    parser.add_argument('--duration', type=int, default=24,
                       help='Sensor data duration in hours')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous sensor monitoring')
    
    # Dashboard arguments
    parser.add_argument('--port', type=int, default=8501,
                       help='Dashboard port')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Dashboard host')
    
    # General arguments
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for processing')
    
    args = parser.parse_args()
    
    # Initialize system
    system = RockfallSystem()
    
    try:
        if args.mode == 'setup':
            system.create_sample_data()
            
        elif args.mode == 'train':
            system.train_model(
                model_size=args.model_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device
            )
            
        elif args.mode == 'detect':
            system.run_video_detection(
                model_path=args.model_path,
                source=args.source,
                confidence=args.confidence,
                device=args.device
            )
            
        elif args.mode == 'sensor':
            system.run_sensor_monitoring(
                duration=args.duration,
                continuous=args.continuous
            )
            
        elif args.mode == 'dem':
            if args.dem_path:
                system.analyze_dem(args.dem_path)
            else:
                # Analyze all DEM files
                dem_dir = Path("data/DEM")
                if dem_dir.exists():
                    dem_files = list(dem_dir.glob("*.tif"))
                    for dem_file in dem_files:
                        system.analyze_dem(str(dem_file))
                else:
                    print("❌ No DEM files found. Use --dem-path to specify a file.")
                    
        elif args.mode == 'dashboard':
            system.launch_dashboard(
                port=args.port,
                host=args.host
            )
            
        elif args.mode == 'all':
            system.run_complete_system(
                port=args.port
            )
            
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
    
    print("\n✅ System operation completed")
    return 0


if __name__ == "__main__":
    exit(main())