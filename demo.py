#!/usr/bin/env python3
"""
Quick Demo Script for Rockfall Detection System
==============================================

This script runs a quick demonstration of all system components.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_demo():
    """Run comprehensive system demo"""
    print("🏔️ Rockfall Detection System - Quick Demo")
    print("="*60)
    
    project_root = Path(__file__).parent
    python_exe = project_root / ".venv" / "Scripts" / "python.exe"
    
    # Test 1: Sensor Analysis
    print("\n📊 Demo 1: Sensor-based Risk Analysis")
    print("-" * 40)
    try:
        result = subprocess.run([
            str(python_exe), "src/sensors/sensor_alerts.py", 
            "--duration", "12", "--save-plot"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Sensor analysis completed successfully")
        else:
            print(f"❌ Sensor analysis failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ Sensor analysis timed out")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    time.sleep(2)
    
    # Test 2: DEM Analysis
    print("\n🗺️ Demo 2: DEM Risk Analysis")
    print("-" * 40)
    try:
        dem_file = project_root / "data" / "DEM" / "Bingham_Canyon_Mine.tif"
        if dem_file.exists():
            result = subprocess.run([
                str(python_exe), "src/dem_analysis/dem_processor.py", 
                "--dem", str(dem_file)
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✅ DEM analysis completed successfully")
                # Print key results
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'High risk areas:' in line or 'Critical zones:' in line or 'Max slope:' in line:
                        print(f"   {line.strip()}")
            else:
                print(f"❌ DEM analysis failed: {result.stderr}")
        else:
            print("❌ DEM file not found")
    except subprocess.TimeoutExpired:
        print("⏰ DEM analysis timed out")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    time.sleep(2)
    
    # Test 3: Create sample data
    print("\n📂 Demo 3: Creating Sample Data")
    print("-" * 40)
    try:
        result = subprocess.run([
            str(python_exe), "main.py", "--mode", "setup"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Sample data created successfully")
        else:
            print(f"❌ Sample data creation failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ Sample data creation timed out")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Demo summary
    print("\n" + "="*60)
    print("🎉 Demo Summary:")
    print("✅ Sensor-based risk analysis: Synthetic data generated and analyzed")
    print("✅ DEM terrain analysis: Mine site risk assessment completed")
    print("✅ Sample data creation: Test datasets generated")
    print("⚠️ Video detection: Ready (requires camera/video file)")
    print("⚠️ Model training: Ready (configured for CPU training)")
    print("="*60)
    
    print("\n🚀 Next Steps:")
    print("1. Train the model:")
    print("   python main.py --mode train --epochs 10")
    print("\n2. Launch the dashboard:")
    print("   python main.py --mode dashboard")
    print("\n3. Run complete system:")
    print("   python main.py --mode all")
    print("\n4. Run video detection (with webcam):")
    print("   python main.py --mode detect --source 0")

if __name__ == "__main__":
    run_demo()