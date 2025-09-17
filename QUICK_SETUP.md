# 🚀 Quick Setup Guide for Web Developers

This guide helps web developers quickly integrate with the Rockfall Detection and Prediction System.

## ⚡ 5-Minute Setup

### 1. **Environment Setup**
```bash
# Clone repository
git clone <repository-url>
cd rockfall_detection

# Create virtual environment  
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Test the System**
```bash
# Test prediction models (ready to use)
python src/prediction/test_models.py

# Test detection model (already trained)
python main.py --mode detect --source data/rockfall_training_data/test/images/

# Launch dashboard
python src/dashboard/app.py
```

## 🔌 API Integration

### **Prediction API**
```python
from src.prediction.test_models import load_prediction_models, predict_rockfall_risk
import numpy as np

# Load once at startup
models, scalers, features, performance = load_prediction_models()

# Make predictions
def get_risk_score(environmental_data):
    """
    environmental_data: dict with 19 features
    Returns: float (0-1 risk score)
    """
    input_array = np.array([[environmental_data[f] for f in features]])
    predictions = predict_rockfall_risk(models, scalers, features, input_array)
    return predictions['ensemble']
```

### **Detection API**
```python
from ultralytics import YOLO

# Load model once
model = YOLO('outputs/experiment_20250916_210441/weights/best.pt')

def detect_rocks(image_path):
    """
    Returns: list of detection results
    """
    results = model(image_path)
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                detections.append({
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                })
    return detections
```

## 🌐 Web Framework Examples

### **Flask API Server**
```python
from flask import Flask, request, jsonify, render_template
from src.prediction.test_models import load_prediction_models, predict_rockfall_risk
import numpy as np

app = Flask(__name__)

# Load models at startup
models, scalers, features, performance = load_prediction_models()

@app.route('/api/risk', methods=['POST'])
def predict_risk():
    data = request.json
    input_array = np.array([[data[f] for f in features]])
    predictions = predict_rockfall_risk(models, scalers, features, input_array)
    
    return jsonify({
        'risk_score': predictions['ensemble'],
        'risk_level': 'HIGH' if predictions['ensemble'] > 0.7 else 'MEDIUM' if predictions['ensemble'] > 0.3 else 'LOW'
    })

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### **Express.js API Server**
```javascript
const express = require('express');
const { spawn } = require('child_process');
const app = express();

app.use(express.json());

// API endpoint for risk prediction
app.post('/api/risk', (req, res) => {
    const python = spawn('python', ['api_wrapper.py', JSON.stringify(req.body)]);
    
    python.stdout.on('data', (data) => {
        res.json(JSON.parse(data.toString()));
    });
});

app.listen(3000, () => {
    console.log('API server running on port 3000');
});
```

### **React Component**
```jsx
import React, { useState, useEffect } from 'react';

const RockfallMonitor = () => {
    const [riskData, setRiskData] = useState(null);
    const [loading, setLoading] = useState(false);

    const fetchRiskAssessment = async (environmentalData) => {
        setLoading(true);
        try {
            const response = await fetch('/api/risk', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(environmentalData)
            });
            const data = await response.json();
            setRiskData(data);
        } catch (error) {
            console.error('Risk assessment failed:', error);
        }
        setLoading(false);
    };

    return (
        <div className="rockfall-monitor">
            <h2>🏔️ Rockfall Risk Monitor</h2>
            
            {riskData && (
                <div className={`risk-card ${riskData.risk_level.toLowerCase()}`}>
                    <div className="risk-score">
                        Risk Score: {(riskData.risk_score * 100).toFixed(1)}%
                    </div>
                    <div className="risk-level">
                        Level: {riskData.risk_level}
                    </div>
                </div>
            )}
            
            <button 
                onClick={() => fetchRiskAssessment(sampleData)}
                disabled={loading}
            >
                {loading ? '⏳ Analyzing...' : '🔍 Assess Risk'}
            </button>
        </div>
    );
};

export default RockfallMonitor;
```

## 📊 Required Input Features

### **Environmental Data Schema**
```javascript
const environmentalData = {
    "slope": 45.0,              // Terrain slope in degrees (0-90)
    "elevation": 1500.0,        // Elevation in meters
    "fracture_density": 3.5,    // Fractures per m² (0-10)
    "roughness": 0.7,          // Surface roughness (0-1)
    "instability_index": 0.8,   // Geological instability (0-1)
    "rainfall": 50.0,          // mm per day
    "temperature": 5.0,        // Celsius
    "freeze_thaw_cycles": 5,   // Number of cycles
    "seismic_activity": 2.0,   // Magnitude (0-7)
    "wind_speed": 30.0,        // km/h
    // ... 9 more features (see full API docs)
};
```

## 🎨 CSS Styling Examples

```css
.risk-card {
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    transition: all 0.3s ease;
}

.risk-card.low {
    background: linear-gradient(135deg, #4CAF50, #81C784);
    color: white;
}

.risk-card.medium {
    background: linear-gradient(135deg, #FF9800, #FFB74D);
    color: white;
}

.risk-card.high {
    background: linear-gradient(135deg, #F44336, #EF5350);
    color: white;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
```

## 🔧 Development Tools

### **Python API Wrapper** (`api_wrapper.py`)
```python
#!/usr/bin/env python3
import sys
import json
from src.prediction.test_models import load_prediction_models, predict_rockfall_risk
import numpy as np

def main():
    # Load models
    models, scalers, features, performance = load_prediction_models()
    
    # Get input data
    input_data = json.loads(sys.argv[1])
    
    # Make prediction
    input_array = np.array([[input_data[f] for f in features]])
    predictions = predict_rockfall_risk(models, scalers, features, input_array)
    
    # Return result
    result = {
        'risk_score': predictions['ensemble'],
        'risk_level': 'HIGH' if predictions['ensemble'] > 0.7 else 'MEDIUM' if predictions['ensemble'] > 0.3 else 'LOW',
        'model_breakdown': predictions
    }
    
    print(json.dumps(result))

if __name__ == '__main__':
    main()
```

## 📱 Mobile Integration

### **React Native Example**
```jsx
const RockfallAPI = {
    baseURL: 'http://your-server.com',
    
    async assessRisk(environmentalData) {
        const response = await fetch(`${this.baseURL}/api/risk`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(environmentalData)
        });
        return response.json();
    },
    
    async detectRocks(imageUri) {
        const formData = new FormData();
        formData.append('image', {
            uri: imageUri,
            type: 'image/jpeg',
            name: 'rockfall_image.jpg'
        });
        
        const response = await fetch(`${this.baseURL}/api/detect`, {
            method: 'POST',
            body: formData,
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return response.json();
    }
};
```

## 🐳 Docker Deployment

### **Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
```

### **docker-compose.yml**
```yaml
version: '3.8'
services:
  rockfall-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./outputs:/app/outputs
    environment:
      - FLASK_ENV=production
```

## 🚀 Deployment Commands

```bash
# Development server
python src/dashboard/app.py

# Production with gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Docker deployment
docker-compose up -d
```

## 🔍 Testing & Validation

```bash
# Test API endpoints
curl -X POST http://localhost:5000/api/risk \
  -H "Content-Type: application/json" \
  -d '{"slope": 45, "elevation": 1500, ...}'

# Validate system
python validate_system.py

# Load testing
pip install locust
locust -f load_test.py
```

## 📚 Additional Resources

- **Full API Documentation**: See main README.md
- **Model Performance**: `outputs/models/model_metadata.joblib`
- **System Logs**: `outputs/logs/` and `logs/`
- **Sample Data**: `sample_data/` directory

## 🆘 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Models not found | Run `python src/prediction/train_models.py` |
| Import errors | Check Python path and virtual environment |
| Memory issues | Use smaller batch sizes or CPU-only mode |
| Slow predictions | Cache models, use ensemble selectively |

---

**🎯 Ready to integrate? Start with the Flask API example and customize for your stack!**