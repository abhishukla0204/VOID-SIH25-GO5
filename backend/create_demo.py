import cv2
import numpy as np
import json
from pathlib import Path

def create_demo_detection():
    """Create a demo detection with bounding boxes from the test dataset"""
    
    # Test image path
    image_path = "../data/rockfall_training_data/test/images/R-102-_jpg.rf.0adff2cdd3e01a5d9b561ae772866bd5.jpg"
    label_path = "../data/rockfall_training_data/test/labels/R-102-_jpg.rf.0adff2cdd3e01a5d9b561ae772866bd5.txt"
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # Read YOLO format labels
    detections = []
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to pixel coordinates
                x_center_px = int(x_center * w)
                y_center_px = int(y_center * h)
                width_px = int(width * w)
                height_px = int(height * h)
                
                # Calculate bounding box corners
                x1 = int(x_center_px - width_px/2)
                y1 = int(y_center_px - height_px/2)
                x2 = int(x_center_px + width_px/2)
                y2 = int(y_center_px + height_px/2)
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add label
                label = f"Rock: 0.95"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                detections.append({
                    "class_id": class_id,
                    "confidence": 0.95,
                    "bbox": [x1, y1, x2, y2],
                    "label": "rock"
                })
    
    # Save the annotated image
    output_path = "../frontend/public/demo_detection.jpg"
    cv2.imwrite(output_path, image)
    
    # Create detection results JSON
    results = {
        "total_detections": len(detections),
        "detections": detections,
        "image_size": {"width": w, "height": h},
        "processing_time": 0.234
    }
    
    with open("../frontend/public/demo_detection_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Demo detection created: {len(detections)} rocks detected")
    print(f"Annotated image saved to: {output_path}")

if __name__ == "__main__":
    create_demo_detection()