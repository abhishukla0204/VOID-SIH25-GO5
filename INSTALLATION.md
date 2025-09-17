# Rockfall Detection System - Installation Guide

## Quick Start for Web Developers

This guide will help you set up and run the complete rockfall detection system with React frontend and FastAPI backend.

## Prerequisites

- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher)
- **Git**

## Backend Setup (FastAPI)

1. **Navigate to the backend directory:**
```bash
cd backend
```

2. **Create a Python virtual environment:**
```bash
python -m venv venv
```

3. **Activate the virtual environment:**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. **Install Python dependencies:**
```bash
pip install fastapi uvicorn python-multipart numpy opencv-python ultralytics joblib scikit-learn pandas websockets
```

5. **Start the FastAPI server:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`

## Frontend Setup (React)

1. **Navigate to the frontend directory:**
```bash
cd frontend
```

2. **Install Node.js dependencies:**
```bash
npm install
```

3. **Start the development server:**
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## System Architecture

```
┌─────────────────────────────────────────────┐
│                Frontend (React)              │
│  ┌─────────────────────────────────────────┐ │
│  │          User Interface                 │ │
│  │  • Dashboard (Real-time monitoring)    │ │
│  │  • Detection (Image upload & analysis) │ │
│  │  • Risk Assessment (Environmental data)│ │
│  │  • Settings (System configuration)     │ │
│  └─────────────────────────────────────────┘ │
│                     │                       │
│              HTTP Requests &                │
│              WebSocket Connection           │
│                     │                       │
└─────────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────┐
│              Backend (FastAPI)              │
│  ┌────────────────────────────────────────┐ │
│  │              API Endpoints             │ │
│  │  • /api/detect-rocks (Image analysis)  │ │
│  │  • /api/predict-risk (Risk calculation)│ │
│  │  • /api/status (System status)         │ │
│  │  • /ws (WebSocket for real-time)       │ │
│  └────────────────────────────────────────┘ │
│                     │                       │
│            ML Model Integration             │
│                     │                       │
│  ┌────────────────────────────────────────┐ │
│  │            ML Models                   │ │
│  │  • YOLOv8 (Rock detection)             │ │
│  │  • Risk Prediction Model               │ │
│  │  • Image Preprocessing                 │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## API Endpoints

### Rock Detection
- **POST** `/api/detect-rocks`
- Upload image file for rock detection
- Returns: Detection results with bounding boxes and confidence scores

### Risk Prediction
- **POST** `/api/predict-risk`
- Send environmental parameters
- Returns: Risk level, probability, and contributing factors

### System Status
- **GET** `/api/status`
- Returns: System health, model status, and performance metrics

### WebSocket Real-time Updates
- **WS** `/ws`
- Real-time notifications and system updates

## Environment Variables

Create a `.env` file in the backend directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Model Paths
YOLO_MODEL_PATH=../outputs/models/best.pt
RISK_MODEL_PATH=../outputs/models/risk_prediction_model.joblib

# WebSocket
WS_MAX_CONNECTIONS=100

# CORS Origins
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

## Features

### 🎯 **Dashboard**
- Real-time system monitoring
- Detection statistics and trends
- Model status indicators
- Recent activity logs

### 🔍 **Rock Detection**
- Drag-and-drop image upload
- AI-powered rock detection using YOLOv8
- Confidence scoring and bounding boxes
- Detection history with preview gallery

### ⚠️ **Risk Assessment**
- Environmental parameter input forms
- AI-based risk level prediction
- Contributing factor analysis
- Assessment history tracking

### ⚙️ **Settings**
- System configuration
- Notification preferences
- Model management
- Performance tuning

## Development

### Frontend Development
```bash
cd frontend
npm run dev        # Start development server
npm run build      # Build for production
npm run preview    # Preview production build
```

### Backend Development
```bash
cd backend
uvicorn main:app --reload  # Start with auto-reload
python -m pytest          # Run tests
```

## Production Deployment

### Docker Deployment (Recommended)

1. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

### Manual Deployment

1. **Backend (FastAPI):**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

2. **Frontend (React):**
```bash
cd frontend
npm run build
# Serve the 'dist' folder with your preferred web server
```

## Troubleshooting

### Common Issues

1. **CORS Errors:**
   - Ensure the frontend URL is added to CORS_ORIGINS in backend
   - Check that the backend is running on the correct port

2. **Model Loading Errors:**
   - Verify model files exist in the specified paths
   - Check Python dependencies are installed

3. **WebSocket Connection Issues:**
   - Ensure no firewall blocking WebSocket connections
   - Check the WebSocket URL in frontend configuration

### Performance Optimization

1. **Backend:**
   - Enable GPU acceleration if available
   - Adjust batch processing settings
   - Monitor memory usage

2. **Frontend:**
   - Enable image compression for uploads
   - Implement virtual scrolling for large lists
   - Use React.memo for expensive components

## Support

For issues and questions:
1. Check the logs in both frontend and backend
2. Review the API documentation at `http://localhost:8000/docs`
3. Ensure all dependencies are properly installed

## System Requirements

### Minimum Requirements
- **RAM:** 8GB
- **Storage:** 10GB free space
- **GPU:** Optional (CUDA-compatible for acceleration)

### Recommended Requirements
- **RAM:** 16GB or higher
- **Storage:** 50GB free space
- **GPU:** NVIDIA GPU with 4GB+ VRAM

## Next Steps

1. **Test the complete system:** Upload test images and verify detection works
2. **Configure notifications:** Set up email alerts and thresholds
3. **Monitor performance:** Check system metrics and optimize as needed
4. **Scale deployment:** Add load balancing and clustering for production

The system is now ready for use! Visit `http://localhost:5173` to access the web interface.