#!/usr/bin/env python3
"""
Real-time Rockfall Detection System
==================================

This module provides real-time rockfall detection using YOLOv8 models.
Features:
- Video stream processing (webcam, IP camera, video files)
- Real-time object detection and tracking
- Alert generation and logging
- Annotated video output
- Performance monitoring
"""

import os
import cv2
import numpy as np
import time
import argparse
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import logging
from typing import Dict, List, Tuple, Optional
import threading
import queue


class RockfallDetector:
    """Real-time rockfall detection using YOLOv8"""
    
    def __init__(self, model_path: str = None, confidence: float = 0.5, 
                 device: str = 'cpu', alert_threshold: int = 3):
        """
        Initialize the rockfall detector
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence: Detection confidence threshold
            device: Device for inference ('cpu', 'cuda')
            alert_threshold: Number of consecutive detections to trigger alert
        """
        self.confidence = confidence
        self.device = device
        self.alert_threshold = alert_threshold
        
        # Detection tracking
        self.detection_count = 0
        self.consecutive_detections = 0
        self.alert_active = False
        self.last_alert_time = 0
        self.alert_cooldown = 30  # seconds
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.start_time = time.time()
        
        # Alert logging
        self.setup_logging()
        self.alerts_log = []
        
        # Load model
        self.load_model(model_path)
        
        # Initialize alert queue for real-time processing
        self.alert_queue = queue.Queue()
        
        print("Rockfall Detector initialized")
        print(f"   Model: {model_path or 'yolov8n.pt (pretrained)'}")
        print(f"   Confidence: {confidence}")
        print(f"   Device: {device}")
        print(f"   Alert threshold: {alert_threshold} consecutive detections")
    
    def setup_logging(self):
        """Setup logging for alerts and system events"""
        log_dir = Path(__file__).parent.parent.parent / "outputs" / "alerts"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"rockfall_alerts_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Rockfall detection system started")
    
    def load_model(self, model_path: str = None):
        """Load YOLOv8 model"""
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"✅ Loaded custom model: {model_path}")
            else:
                # Use pretrained model for demonstration
                self.model = YOLO('yolov8n.pt')
                print("⚠️ Using pretrained YOLOv8n model (not trained on rockfall data)")
                print("   For actual rockfall detection, train a custom model first")
            
            # Set model device
            self.model.to(self.device)
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect objects in frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            detections: List of detection dictionaries
            annotated_frame: Frame with bounding boxes drawn
        """
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence, verbose=False)
            
            detections = []
            annotated_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[cls]
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class': class_name,
                            'class_id': cls,
                            'timestamp': datetime.now().isoformat()
                        }
                        detections.append(detection)
                        
                        # Draw bounding box
                        color = (0, 0, 255) if class_name.lower() in ['rock', 'stone', 'boulder'] else (0, 255, 0)
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10),
                                    (int(x1) + label_size[0], int(y1)), color, -1)
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return detections, annotated_frame
            
        except Exception as e:
            self.logger.error(f"Error in object detection: {e}")
            return [], frame
    
    def check_rockfall_alert(self, detections: List[Dict]) -> bool:
        """
        Check if rockfall alert should be triggered
        
        Args:
            detections: List of current detections
            
        Returns:
            alert_triggered: True if alert should be triggered
        """
        # Check for rock-like objects
        rock_detections = [d for d in detections if 
                          d['class'].lower() in ['rock', 'stone', 'boulder', 'person'] or  # person as proxy for rocks
                          d['confidence'] > self.confidence]
        
        if rock_detections:
            self.consecutive_detections += 1
            self.detection_count += 1
        else:
            self.consecutive_detections = 0
        
        # Trigger alert if threshold reached and not in cooldown
        current_time = time.time()
        if (self.consecutive_detections >= self.alert_threshold and 
            current_time - self.last_alert_time > self.alert_cooldown):
            
            self.last_alert_time = current_time
            self.alert_active = True
            return True
        
        return False
    
    def generate_alert(self, detections: List[Dict], frame: np.ndarray):
        """Generate and log rockfall alert"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'alert_id': f"ROCKFALL_{int(time.time())}",
            'detection_count': len(detections),
            'consecutive_detections': self.consecutive_detections,
            'detections': detections,
            'alert_level': 'HIGH' if len(detections) > 2 else 'MEDIUM'
        }
        
        # Log alert
        self.logger.warning(f"🚨 ROCKFALL ALERT: {alert_data['alert_id']}")
        self.logger.warning(f"   Level: {alert_data['alert_level']}")
        self.logger.warning(f"   Detections: {alert_data['detection_count']}")
        self.logger.warning(f"   Consecutive: {alert_data['consecutive_detections']}")
        
        # Store alert
        self.alerts_log.append(alert_data)
        
        # Add to alert queue for dashboard
        try:
            self.alert_queue.put_nowait(alert_data)
        except queue.Full:
            pass  # Queue full, skip this alert
        
        # Save alert frame
        self.save_alert_frame(frame, alert_data['alert_id'])
        
        print(f"\n🚨 ROCKFALL ALERT TRIGGERED! 🚨")
        print(f"Alert ID: {alert_data['alert_id']}")
        print(f"Alert Level: {alert_data['alert_level']}")
        print(f"Detections: {alert_data['detection_count']}")
        print("="*50)
    
    def save_alert_frame(self, frame: np.ndarray, alert_id: str):
        """Save frame that triggered alert"""
        try:
            alerts_dir = Path(__file__).parent.parent.parent / "outputs" / "alerts"
            alerts_dir.mkdir(parents=True, exist_ok=True)
            
            filename = alerts_dir / f"{alert_id}.jpg"
            cv2.imwrite(str(filename), frame)
            self.logger.info(f"Alert frame saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving alert frame: {e}")
    
    def add_overlay_info(self, frame: np.ndarray) -> np.ndarray:
        """Add system information overlay to frame"""
        overlay = frame.copy()
        
        # Calculate FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
        else:
            fps = 0
        
        # System info
        info_text = [
            f"Rockfall Detection System",
            f"FPS: {fps:.1f}",
            f"Detections: {self.detection_count}",
            f"Consecutive: {self.consecutive_detections}",
            f"Confidence: {self.confidence}",
            f"Status: {'🚨 ALERT' if self.alert_active else '✅ Normal'}"
        ]
        
        # Draw info box
        box_height = len(info_text) * 30 + 20
        cv2.rectangle(overlay, (10, 10), (350, box_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (350, box_height), (255, 255, 255), 2)
        
        # Add text
        for i, text in enumerate(info_text):
            color = (0, 0, 255) if "ALERT" in text else (255, 255, 255)
            cv2.putText(overlay, text, (20, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Reset alert status after displaying
        if self.alert_active:
            self.alert_active = False
        
        return overlay
    
    def process_video_stream(self, source: str = 0, output_path: str = None, 
                           display: bool = True, save_video: bool = False):
        """
        Process video stream for rockfall detection
        
        Args:
            source: Video source (0 for webcam, path for video file, URL for IP camera)
            output_path: Path to save output video
            display: Whether to display video in window
            save_video: Whether to save processed video
        """
        print(f"\n🎥 Starting video processing...")
        print(f"Source: {source}")
        print(f"Display: {display}")
        print(f"Save video: {save_video}")
        print("="*50)
        
        # Initialize video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video properties: {width}x{height} @ {original_fps:.1f} FPS")
        
        # Initialize video writer if saving
        video_writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
            print(f"Saving video to: {output_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect objects
                detections, annotated_frame = self.detect_objects(frame)
                
                # Check for alerts
                if self.check_rockfall_alert(detections):
                    self.generate_alert(detections, annotated_frame)
                
                # Add overlay information
                final_frame = self.add_overlay_info(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Rockfall Detection', final_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nStopping detection...")
                        break
                    elif key == ord('s'):
                        # Save current frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"detection_frame_{timestamp}.jpg"
                        cv2.imwrite(filename, final_frame)
                        print(f"Frame saved: {filename}")
                    elif key == ord('r'):
                        # Reset counters
                        self.detection_count = 0
                        self.consecutive_detections = 0
                        print("Counters reset")
                
                # Save to video file
                if video_writer:
                    video_writer.write(final_frame)
        
        except KeyboardInterrupt:
            print("\nDetection interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            
            # Print summary
            elapsed_time = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n📊 Detection Summary:")
            print(f"   Frames processed: {self.frame_count}")
            print(f"   Runtime: {elapsed_time:.1f}s")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Total detections: {self.detection_count}")
            print(f"   Alerts triggered: {len(self.alerts_log)}")
            
            self.logger.info(f"Detection session ended - {self.frame_count} frames, {len(self.alerts_log)} alerts")
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent alerts for dashboard"""
        return self.alerts_log[-count:] if self.alerts_log else []
    
    def get_alert_queue(self) -> queue.Queue:
        """Get alert queue for real-time dashboard updates"""
        return self.alert_queue


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Real-time Rockfall Detection')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained YOLOv8 model')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--save-video', action='store_true',
                       help='Save processed video')
    parser.add_argument('--alert-threshold', type=int, default=3,
                       help='Consecutive detections to trigger alert')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a number (webcam)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Create output path if saving video
    if args.save_video and not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"rockfall_detection_{timestamp}.mp4")
    
    try:
        # Initialize detector
        detector = RockfallDetector(
            model_path=args.model,
            confidence=args.confidence,
            device=args.device,
            alert_threshold=args.alert_threshold
        )
        
        # Start detection
        detector.process_video_stream(
            source=source,
            output_path=args.output,
            display=not args.no_display,
            save_video=args.save_video
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())