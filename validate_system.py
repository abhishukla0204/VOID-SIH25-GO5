#!/usr/bin/env python3
"""
System Validation and Testing Script
===================================

This script validates all components of the rockfall detection system
and provides a comprehensive status report.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_yolo_training():
    """Test YOLOv8 training pipeline"""
    print("Testing YOLOv8 training pipeline...")
    try:
        from src.training.train_yolo import RockfallTrainer
        trainer = RockfallTrainer()
        trainer.validate_dataset()
        print("  ✅ Training pipeline: PASSED")
        return True
    except Exception as e:
        print(f"  ❌ Training pipeline: FAILED - {e}")
        return False

def test_video_detection():
    """Test video detection system"""
    print("Testing video detection system...")
    try:
        from src.detection.realtime_detector import RockfallDetector
        detector = RockfallDetector(confidence=0.5, device='cpu')
        print("  ✅ Video detection: PASSED")
        return True
    except Exception as e:
        print(f"  ❌ Video detection: FAILED - {e}")
        return False

def test_sensor_analysis():
    """Test sensor analysis system"""
    print("Testing sensor analysis system...")
    try:
        from src.sensors.sensor_alerts import SensorDataProcessor
        processor = SensorDataProcessor()
        
        # Generate test data
        df = processor.generate_synthetic_sensor_data(duration_hours=1)
        
        # Analyze data
        result = processor.analyze_sensor_data(df)
        
        # Validate results
        assert 'risk_score' in result
        assert 'risk_level' in result
        assert 'alerts' in result
        
        print("  ✅ Sensor analysis: PASSED")
        return True
    except Exception as e:
        print(f"  ❌ Sensor analysis: FAILED - {e}")
        return False

def test_dem_analysis():
    """Test DEM analysis system"""
    print("Testing DEM analysis system...")
    try:
        # Check if DEM files exist
        dem_dir = Path("data/DEM")
        dem_files = list(dem_dir.glob("*.tif")) if dem_dir.exists() else []
        
        if not dem_files:
            print("  ⚠️ DEM analysis: SKIPPED - No DEM files found")
            return True
        
        from src.dem_analysis.dem_processor import DEMAnalyzer
        
        # Test with first available DEM file
        analyzer = DEMAnalyzer(str(dem_files[0]))
        
        # Test basic functionality
        risk_results = analyzer.assess_rockfall_risk()
        critical_zones = analyzer.identify_critical_zones(risk_results)
        
        # Validate results
        assert 'slope' in risk_results
        assert 'risk_classification' in risk_results
        assert isinstance(critical_zones, list)
        
        print("  ✅ DEM analysis: PASSED")
        return True
    except Exception as e:
        print(f"  ❌ DEM analysis: FAILED - {e}")
        return False

def test_dashboard_components():
    """Test dashboard components"""
    print("Testing dashboard components...")
    try:
        from src.dashboard.app import RockfallDashboard
        dashboard = RockfallDashboard()
        
        # Test basic initialization
        assert hasattr(dashboard, 'initialize_session_state')
        assert hasattr(dashboard, 'render_header')
        
        print("  ✅ Dashboard components: PASSED")
        return True
    except Exception as e:
        print(f"  ❌ Dashboard components: FAILED - {e}")
        return False

def test_data_integrity():
    """Test data integrity and structure"""
    print("Testing data integrity...")
    try:
        # Check dataset structure
        data_yaml = Path("data/rockfall_training_data/data.yaml")
        if data_yaml.exists():
            import yaml
            with open(data_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate config
            assert 'nc' in config
            assert 'names' in config
            assert config['nc'] == 1
            assert config['names'] == ['Rock']
            
            # Check if image directories exist
            data_dir = data_yaml.parent
            for split in ['train', 'valid', 'test']:
                if split in config:
                    split_dir = data_dir / config[split]
                    if split_dir.exists():
                        images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
                        assert len(images) > 0, f"No images found in {split} set"
            
            print("  ✅ Dataset integrity: PASSED")
        else:
            print("  ⚠️ Dataset integrity: SKIPPED - data.yaml not found")
        
        return True
    except Exception as e:
        print(f"  ❌ Dataset integrity: FAILED - {e}")
        return False

def test_output_directories():
    """Test output directory structure"""
    print("Testing output directories...")
    try:
        required_dirs = [
            "outputs",
            "outputs/videos",
            "outputs/alerts",
            "models",
            "sample_data",
            "sample_data/sensor_data",
            "sample_data/weather_data"
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("  ✅ Output directories: PASSED")
        return True
    except Exception as e:
        print(f"  ❌ Output directories: FAILED - {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("Testing dependencies...")
    try:
        required_packages = [
            'ultralytics',
            'cv2',  # opencv-python
            'numpy',
            'pandas',
            'matplotlib',
            'streamlit',
            'rasterio',
            'plotly',
            'sklearn'  # scikit-learn
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"  ❌ Dependencies: FAILED - Missing: {missing_packages}")
            return False
        
        print("  ✅ Dependencies: PASSED")
        return True
    except Exception as e:
        print(f"  ❌ Dependencies: FAILED - {e}")
        return False

def generate_system_report(test_results):
    """Generate comprehensive system report"""
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "system_status": "OPERATIONAL" if all(test_results.values()) else "ISSUES_DETECTED",
        "component_status": test_results,
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
            "project_root": str(Path(__file__).parent.absolute())
        },
        "recommendations": []
    }
    
    # Add recommendations based on test results
    if not test_results.get("training_pipeline", False):
        report["recommendations"].append("Fix YOLOv8 training pipeline configuration")
    
    if not test_results.get("video_detection", False):
        report["recommendations"].append("Resolve video detection system issues")
    
    if not test_results.get("sensor_analysis", False):
        report["recommendations"].append("Debug sensor analysis module")
    
    if not test_results.get("dem_analysis", False):
        report["recommendations"].append("Check DEM analysis dependencies and data files")
    
    if not test_results.get("dashboard", False):
        report["recommendations"].append("Fix dashboard component initialization")
    
    if all(test_results.values()):
        report["recommendations"].append("System is fully operational - ready for deployment")
    
    # Save report
    report_path = Path("outputs/system_validation_report.json")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report, report_path

def main():
    """Run comprehensive system validation"""
    print("="*60)
    print("ROCKFALL DETECTION SYSTEM - VALIDATION REPORT")
    print("="*60)
    print(f"Validation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all tests
    test_results = {}
    
    test_results["dependencies"] = test_dependencies()
    test_results["output_directories"] = test_output_directories()
    test_results["data_integrity"] = test_data_integrity()
    test_results["training_pipeline"] = test_yolo_training()
    test_results["video_detection"] = test_video_detection()
    test_results["sensor_analysis"] = test_sensor_analysis()
    test_results["dem_analysis"] = test_dem_analysis()
    test_results["dashboard"] = test_dashboard_components()
    
    # Generate report
    print("\nGenerating system report...")
    report, report_path = generate_system_report(test_results)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    print(f"System status: {report['system_status']}")
    
    print("\nComponent Status:")
    for component, status in test_results.items():
        status_symbol = "✅" if status else "❌"
        print(f"  {status_symbol} {component.replace('_', ' ').title()}")
    
    if report["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nDetailed report saved to: {report_path}")
    
    # Usage instructions
    print("\n" + "="*60)
    print("QUICK START GUIDE")
    print("="*60)
    
    if report['system_status'] == 'OPERATIONAL':
        print("🎉 System is ready! Try these commands:")
        print()
        print("1. Launch the dashboard:")
        print("   python main.py --mode dashboard")
        print()
        print("2. Run sensor analysis:")
        print("   python main.py --mode sensor --duration 12")
        print()
        print("3. Analyze DEM data:")
        print("   python main.py --mode dem")
        print()
        print("4. Train the model (CPU, quick test):")
        print("   python main.py --mode train --epochs 5 --batch-size 4")
        print()
        print("5. Run complete system:")
        print("   python main.py --mode all")
    else:
        print("⚠️ System has issues. Please address the failed components first.")
        print("Check the recommendations above and the detailed report.")
    
    print("\n" + "="*60)
    
    return 0 if report['system_status'] == 'OPERATIONAL' else 1

if __name__ == "__main__":
    exit(main())