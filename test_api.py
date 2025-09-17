#!/usr/bin/env python3
"""
Simple test script to verify the rock detection API is working.
"""

import requests
import os
from pathlib import Path

def test_api():
    """Test the rock detection API with a sample image."""
    # Find a test image
    test_image_dir = Path("data/rockfall_training_data/test/images")
    
    if not test_image_dir.exists():
        print(f"❌ Test image directory not found: {test_image_dir}")
        return False
    
    # Get the first available test image
    test_images = list(test_image_dir.glob("*.jpg"))
    if not test_images:
        print("❌ No test images found")
        return False
    
    test_image = test_images[0]
    print(f"🔍 Testing with image: {test_image.name}")
    
    try:
        # Test API status first
        print("📡 Testing API status...")
        status_response = requests.get("http://localhost:8000/api/status", timeout=10)
        if status_response.status_code == 200:
            print("✅ API status check passed")
            print(f"Status: {status_response.json()}")
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
        # Test rock detection
        print("🪨 Testing rock detection...")
        with open(test_image, 'rb') as f:
            files = {'file': (test_image.name, f, 'image/jpeg')}
            response = requests.post(
                "http://localhost:8000/api/detect-rocks",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            results = response.json()
            print("✅ Rock detection API test passed!")
            print(f"Results: {results}")
            return True
        else:
            print(f"❌ Rock detection API failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing rock detection: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Rock Detection API")
    print("=" * 40)
    success = test_api()
    print("=" * 40)
    if success:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("💥 Tests failed. Check the backend server logs.")