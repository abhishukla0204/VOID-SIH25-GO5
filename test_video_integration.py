#!/usr/bin/env python3
"""
Video Integration Test Script
=============================

Test script to verify that the camera video files are properly integrated
and can be processed by the backend system.
"""

import cv2
import os
import sys
from pathlib import Path

def test_video_files():
    """Test the video files in data/camera_data/"""
    
    # Get project root
    project_root = Path(__file__).parent
    video_dir = project_root / "data" / "camera_data"
    
    print("🎥 Testing Camera Video Files")
    print("=" * 50)
    
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return False
    
    video_files = ["1.mp4", "2.mp4", "3.mp4"]
    camera_mapping = {
        "1.mp4": "East Camera",
        "2.mp4": "West Camera", 
        "3.mp4": "North Camera"
    }
    
    all_tests_passed = True
    
    for video_file in video_files:
        video_path = video_dir / video_file
        camera_name = camera_mapping[video_file]
        
        print(f"\n📹 Testing {camera_name} ({video_file}):")
        print("-" * 30)
        
        if not video_path.exists():
            print(f"❌ File not found: {video_path}")
            all_tests_passed = False
            continue
        
        # Test with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Cannot open video file: {video_path}")
            all_tests_passed = False
            continue
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Successfully opened video")
        print(f"   📊 Duration: {duration:.1f} seconds")
        print(f"   🎞️  FPS: {fps:.1f}")
        print(f"   📏 Resolution: {width}x{height}")
        print(f"   🖼️  Frame Count: {frame_count}")
        
        # Test reading a few frames
        frames_read = 0
        for i in range(10):  # Test first 10 frames
            ret, frame = cap.read()
            if ret:
                frames_read += 1
            else:
                break
        
        print(f"   ✅ Successfully read {frames_read}/10 test frames")
        
        # Test looping (go back to start)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            print(f"   ✅ Video looping works correctly")
        else:
            print(f"   ❌ Video looping failed")
            all_tests_passed = False
        
        cap.release()
        
        # File size check
        file_size = video_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   💾 File Size: {file_size:.1f} MB")
        
        if duration < 7 or duration > 9:
            print(f"   ⚠️  Warning: Duration {duration:.1f}s is not close to 8 seconds")
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✅ All video tests passed! Ready for integration.")
        print("\n🚀 To start the system:")
        print("   1. Backend: cd backend && python main.py")
        print("   2. Frontend: cd frontend && npm run dev")
        print("   3. Open: http://localhost:3000/live-monitoring")
    else:
        print("❌ Some video tests failed. Check the files and try again.")
    
    return all_tests_passed

if __name__ == "__main__":
    test_video_files()