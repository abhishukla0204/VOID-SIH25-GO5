#!/usr/bin/env python3
"""
Test script to verify the risk assessment API is working with correct field mappings.
"""

import requests
import json
from datetime import datetime

def test_risk_assessment():
    """Test the risk assessment API with sample environmental data."""
    
    # Sample data matching the EnvironmentalData model
    test_data = {
        "slope": 45.0,
        "elevation": 1500.0,
        "fracture_density": 3.2,
        "roughness": 0.7,
        "slope_variability": 0.4,
        "instability_index": 0.6,
        "wetness_index": 0.3,
        "month": datetime.now().month,
        "day_of_year": datetime.now().timetuple().tm_yday,
        "season": (datetime.now().month - 1) // 3,
        "rainfall": 75.0,
        "temperature": 15.0,
        "temperature_variation": 8.0,
        "freeze_thaw_cycles": 12.0,
        "seismic_activity": 3.5,
        "wind_speed": 25.0,
        "precipitation_intensity": 8.0,
        "humidity": 65.0,
        "risk_score": 0.0
    }
    
    try:
        # Test API status first
        print("📡 Testing API status...")
        status_response = requests.get("http://localhost:8000/api/status", timeout=10)
        if status_response.status_code == 200:
            print("✅ API status check passed")
        else:
            print(f"❌ API status check failed: {status_response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Is it running on port 8000?")
        return False
    except Exception as e:
        print(f"❌ Error checking API status: {e}")
        return False
    
    try:
        # Test risk assessment
        print("🔍 Testing risk assessment...")
        print(f"📊 Sending data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(
            "http://localhost:8000/api/predict-risk",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            print("✅ Risk assessment API test passed!")
            print(f"📈 Results:")
            print(f"   Risk Score: {results.get('risk_score', 'N/A')}")
            print(f"   Risk Level: {results.get('risk_level', 'N/A')}")
            print(f"   Confidence: {results.get('confidence', 'N/A')}")
            print(f"   Model Predictions: {results.get('model_predictions', {})}")
            print(f"   Recommendations: {results.get('recommendations', [])}")
            return True
        else:
            print(f"❌ Risk assessment API failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing risk assessment: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Risk Assessment API")
    print("=" * 50)
    success = test_risk_assessment()
    print("=" * 50)
    if success:
        print("🎉 Risk assessment API is working correctly!")
    else:
        print("💥 Risk assessment tests failed. Check the backend server logs.")