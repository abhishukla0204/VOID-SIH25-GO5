"""
FastAPI Backend for Rockfall Detection System
===========================================

A modern REST API backend built with FastAPI for:
- Real-time rockfall risk prediction
- Image-based rock detection  
- Live monitoring and alerts
- Historical data analysis

Features:
- High-performance async endpoints
- Automatic API documentation
- File upload handling
- Real-time WebSocket connections
- ML model integration
- CORS support for React frontend
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
# Removed StaticFiles import - serving frontend separately
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import json
import os
import sys
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import io
from PIL import Image
import tempfile
from contextlib import asynccontextmanager
import cv2
import threading
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment Variables Configuration
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")
## RELOAD = os.getenv("RELOAD", "true").lower() == "true"
RELOAD = os.getenv("RELOAD", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()

# API Configuration
API_TITLE = os.getenv("API_TITLE", "Rockfall Detection API")
API_DESCRIPTION = os.getenv("API_DESCRIPTION", "Advanced AI-powered rockfall detection and prediction system")
API_VERSION = os.getenv("API_VERSION", "1.0.0")

# CORS Configuration - Production only (frontend deployed separately)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://void-sih-25-go-5.vercel.app").split(",")

# File Paths
MODELS_DIR = os.getenv("MODELS_DIR", "outputs/models")
DATA_DIR = os.getenv("DATA_DIR", "data")
# STATIC_DIR removed for separate deployment

# ML Model Settings
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Add backend paths for imports - deployment compatible
backend_root = Path(__file__).parent  # This is the backend/ directory
sys.path.append(str(backend_root))
sys.path.append(str(backend_root / "src"))

# Configure logging with environment variable
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Import ML models and utilities (with error handling)
try:
    # Try to import custom prediction functions first
    from src.prediction.test_models import load_prediction_models, predict_rockfall_risk
    ML_MODELS_AVAILABLE = True
    logger.info("ML prediction models imported successfully")
except ImportError as e:
    logger.warning(f"Custom ML prediction models not available: {e}")
    # We'll create our own loading functions for the models in outputs/models/
    ML_MODELS_AVAILABLE = True  # We have models in outputs/models/

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLO model imported successfully")
except ImportError as e:
    logger.warning(f"YOLO model not available: {e}")
    YOLO_AVAILABLE = False

# Global variables for models
prediction_models = None
detection_model = None
scalers = None
feature_names = None
model_performance = None

# Custom model loading functions
def load_models_from_outputs():
    """Load models from outputs/models directory"""
    import joblib
    import torch
    
    models_dir = backend_root / MODELS_DIR
    models = {}
    
    try:
        # Load XGBoost model
        xgb_path = models_dir / "xgboost_model.joblib"
        if xgb_path.exists():
            models['xgboost'] = joblib.load(xgb_path)
            logger.info("✅ XGBoost model loaded")
        
        # Load Random Forest model
        rf_path = models_dir / "random_forest_model.joblib"
        if rf_path.exists():
            models['random_forest'] = joblib.load(rf_path)
            logger.info("✅ Random Forest model loaded")
        
        # Load Neural Network model
        nn_path = models_dir / "neural_network_model.pth"
        if nn_path.exists():
            try:
                models['neural_network'] = torch.load(nn_path, map_location='cpu')
                logger.info("✅ Neural Network model loaded")
            except Exception as e:
                logger.warning(f"Could not load neural network: {e}")
        
        # Load scaler
        scaler_path = models_dir / "main_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info("✅ Scaler loaded")
        else:
            scaler = None
        
        # Load metadata
        metadata_path = models_dir / "model_metadata.joblib"
        metadata = None
        if metadata_path.exists():
            try:
                metadata = joblib.load(metadata_path)
                logger.info("✅ Model metadata loaded")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        
        return models, scaler, metadata
        
    except Exception as e:
        logger.error(f"Error loading models from outputs: {e}")
        return {}, None, None

def predict_with_loaded_models(models, scaler, input_data):
    """Make predictions using loaded models"""
    import numpy as np
    
    if scaler is not None:
        input_scaled = scaler.transform(input_data)
    else:
        input_scaled = input_data
    
    predictions = {}
    
    # XGBoost prediction
    if 'xgboost' in models:
        try:
            pred = models['xgboost'].predict_proba(input_scaled)
            value = float(pred[0][1]) if len(pred[0]) > 1 else float(pred[0][0])
            if np.isfinite(value):
                predictions['xgboost'] = value
            else:
                logger.warning(f"XGBoost returned invalid value: {value}")
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
    
    # Random Forest prediction
    if 'random_forest' in models:
        try:
            pred = models['random_forest'].predict_proba(input_scaled)
            value = float(pred[0][1]) if len(pred[0]) > 1 else float(pred[0][0])
            if np.isfinite(value):
                predictions['random_forest'] = value
            else:
                logger.warning(f"Random Forest returned invalid value: {value}")
        except Exception as e:
            logger.warning(f"Random Forest prediction failed: {e}")
    
    # Neural Network prediction
    if 'neural_network' in models:
        try:
            logger.info("Attempting Neural Network prediction...")
            import torch
            model = models['neural_network']
            logger.info(f"Neural Network model type: {type(model)}")
            logger.info(f"Input data shape: {input_scaled.shape}")
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_scaled)
                logger.info(f"Input tensor shape: {input_tensor.shape}")
                pred = model(input_tensor)
                logger.info(f"Raw prediction: {pred}")
                value = float(torch.sigmoid(pred).item())
                logger.info(f"Sigmoid prediction: {value}")
                if np.isfinite(value):
                    predictions['neural_network'] = value
                    logger.info(f"✅ Neural Network prediction successful: {value}")
                else:
                    logger.warning(f"Neural Network returned invalid value: {value}")
        except Exception as e:
            logger.warning(f"Neural Network prediction failed: {e}")
            logger.exception("Neural Network prediction error details:")
    
    # Calculate ensemble prediction
    if predictions:
        valid_values = [v for v in predictions.values() if np.isfinite(v)]
        if valid_values:
            ensemble_pred = np.mean(valid_values)
            if np.isfinite(ensemble_pred):
                predictions['ensemble'] = float(ensemble_pred)
            else:
                logger.warning("Ensemble calculation resulted in invalid value")
        else:
            logger.warning("No valid predictions for ensemble calculation")
    
    return predictions

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Video streaming manager for camera feeds
class VideoStreamManager:
    def __init__(self):
        # Map video files to camera directions (using Cloudinary URLs)
        self.camera_videos = {
            "east": "https://res.cloudinary.com/dyb6aumhm/video/upload/v1758167914/1_znxt5x.mp4",
            "west": "https://res.cloudinary.com/dyb6aumhm/video/upload/v1758167915/2_lrgtxq.mp4", 
            "north": "https://res.cloudinary.com/dyb6aumhm/video/upload/v1758167915/3_gk37sc.mp4",
            "south": None  # Maintenance
        }
        
        # Video capture objects
        self.video_captures = {}
        self.video_info = {}
        self.streaming_threads = {}
        self.stream_active = {}
        
        # Initialize video captures
        self._initialize_videos()
    
    def _initialize_videos(self):
        """Initialize video capture objects and get metadata"""
        for direction, video_path in self.camera_videos.items():
            if video_path:
                # Handle both local files and HTTP URLs
                if video_path.startswith('http'):
                    # For HTTP URLs, try to open directly with cv2
                    cap = cv2.VideoCapture(video_path)
                elif os.path.exists(video_path):
                    # For local files, check existence first
                    cap = cv2.VideoCapture(video_path)
                else:
                    cap = None
                
                if cap and cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    self.video_info[direction] = {
                        "fps": fps,
                        "frame_count": frame_count,
                        "duration": duration,
                        "resolution": f"{width}x{height}",
                        "status": "active"
                    }
                    self.video_captures[direction] = cap
                    self.stream_active[direction] = False
                    logger.info(f"Initialized camera {direction}: {duration:.1f}s, {fps:.1f}fps, {width}x{height}")
                else:
                    logger.error(f"Failed to open video for camera {direction}: {video_path}")
                    self.video_info[direction] = {
                        "fps": 0,
                        "frame_count": 0,
                        "duration": 0,
                        "resolution": "N/A",
                        "status": "offline"
                    }
            else:
                self.video_info[direction] = {
                    "fps": 0,
                    "frame_count": 0,
                    "duration": 0,
                    "resolution": "N/A",
                    "status": "maintenance" if direction == "south" else "offline"
                }
    
    def get_camera_status(self):
        """Get status of all cameras with real video metadata"""
        status = {"cameras": {}, "system": {}}
        
        active_count = 0
        recording_count = 0
        
        for direction in ["east", "west", "north", "south"]:
            info = self.video_info.get(direction, {})
            is_active = info.get("status") == "active"
            is_recording = is_active and self.stream_active.get(direction, False)
            
            if is_active:
                active_count += 1
            if is_recording:
                recording_count += 1
                
            status["cameras"][direction] = {
                "id": f"camera-{direction}",
                "name": f"{direction.title()} Camera",
                "status": info.get("status", "offline"),
                "resolution": info.get("resolution", "N/A"),
                "fps": info.get("fps", 0),
                "duration": info.get("duration", 0),
                "last_detection": datetime.now().isoformat() if is_active else None,
                "recording": is_recording,
                "streaming": self.stream_active.get(direction, False)
            }
        
        status["system"] = {
            "total_cameras": 4,
            "active_cameras": active_count,
            "recording_cameras": recording_count,
            "storage_used": "45.2 GB",
            "uptime": "12h 34m"
        }
        
        return status
    
    def generate_frames(self, direction: str):
        """Generate video frames for streaming with looping"""
        if direction not in self.video_captures:
            return
            
        cap = self.video_captures[direction]
        self.stream_active[direction] = True
        
        try:
            while self.stream_active[direction]:
                ret, frame = cap.read()
                
                if not ret:
                    # End of video reached, restart from beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Control frame rate (simulate real-time playback)
                time.sleep(1.0 / self.video_info[direction]["fps"])
                
        except Exception as e:
            logger.error(f"Error in video stream for {direction}: {e}")
        finally:
            self.stream_active[direction] = False
    
    def start_stream(self, direction: str):
        """Start streaming for a specific camera"""
        if direction in self.video_captures and not self.stream_active.get(direction, False):
            self.stream_active[direction] = True
            return True
        return False
    
    def stop_stream(self, direction: str):
        """Stop streaming for a specific camera"""
        if direction in self.stream_active:
            self.stream_active[direction] = False
            return True
        return False

# Initialize video stream manager
video_manager = VideoStreamManager()

# Pydantic models for request/response
class EnvironmentalData(BaseModel):
    """Environmental data for risk prediction"""
    slope: float = Field(..., ge=0, le=90, description="Terrain slope in degrees")
    elevation: float = Field(..., ge=0, le=5000, description="Elevation in meters")
    fracture_density: float = Field(..., ge=0, le=10, description="Fractures per square meter")
    roughness: float = Field(..., ge=0, le=1, description="Surface roughness index")
    slope_variability: float = Field(0.0, ge=0, le=1, description="Slope variation")
    instability_index: float = Field(..., ge=0, le=1, description="Geological instability index")
    wetness_index: float = Field(0.0, ge=0, le=1, description="Wetness index")
    month: float = Field(..., ge=1, le=12, description="Month of year")
    day_of_year: float = Field(..., ge=1, le=366, description="Day of year")
    season: float = Field(..., ge=0, le=3, description="Season (0-3)")
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall in mm")
    temperature: float = Field(..., ge=-50, le=50, description="Temperature in Celsius")
    temperature_variation: float = Field(0.0, ge=0, le=50, description="Temperature variation")
    freeze_thaw_cycles: float = Field(..., ge=0, le=50, description="Number of freeze-thaw cycles")
    seismic_activity: float = Field(..., ge=0, le=10, description="Seismic activity magnitude")
    wind_speed: float = Field(..., ge=0, le=200, description="Wind speed in km/h")
    precipitation_intensity: float = Field(0.0, ge=0, le=100, description="Precipitation intensity")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    risk_score: float = Field(0.0, ge=0, le=1, description="Base risk score")

class RiskPrediction(BaseModel):
    """Risk prediction response"""
    risk_score: float = Field(..., description="Overall risk score (0-1)")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    confidence: float = Field(..., description="Prediction confidence")
    model_predictions: Dict[str, float] = Field(..., description="Individual model predictions")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Prediction timestamp in ISO format")
    recommendations: List[str] = Field(..., description="Safety recommendations")

class DetectionResult(BaseModel):
    """Rock detection result"""
    detections: List[Dict[str, Any]] = Field(..., description="List of detected rocks")
    total_detections: int = Field(..., description="Total number of rocks detected")
    confidence_threshold: float = Field(0.5, description="Confidence threshold used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    image_dimensions: Dict[str, int] = Field(..., description="Image width and height")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Detection timestamp in ISO format")

class SystemStatus(BaseModel):
    """System status response"""
    status: str = Field(..., description="System status")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    uptime: str = Field(..., description="System uptime")
    version: str = Field("1.0.0", description="API version")
    active_connections: int = Field(..., description="Active WebSocket connections")

# Load models on startup
async def load_models():
    """Load ML models during startup"""
    global prediction_models, detection_model, scalers, feature_names, model_performance
    
    # Load prediction models from outputs/models
    try:
        logger.info("Loading prediction models from outputs/models...")
        models, scaler, metadata = load_models_from_outputs()
        
        if models:
            prediction_models = models
            scalers = scaler
            
            # Set default feature names if not in metadata
            if metadata and 'feature_names' in metadata:
                feature_names = metadata['feature_names']
            else:
                # Default feature names based on your EnvironmentalData model
                feature_names = [
                    'slope', 'elevation', 'fracture_density', 'roughness', 
                    'slope_variability', 'instability_index', 'wetness_index',
                    'month', 'day_of_year', 'season', 'rainfall', 'temperature',
                    'temperature_variation', 'freeze_thaw_cycles', 'seismic_activity',
                    'wind_speed', 'precipitation_intensity', 'humidity'
                ]
            
            if metadata and 'performance' in metadata:
                model_performance = metadata['performance']
            
            logger.info(f"✅ Loaded {len(models)} prediction models successfully")
        else:
            logger.warning("⚠️ No prediction models found in outputs/models")
            prediction_models = None
            
    except Exception as e:
        logger.error(f"❌ Failed to load prediction models: {e}")
        prediction_models = None
    
    # Load YOLO detection model
    if YOLO_AVAILABLE:
        try:
            logger.info("Loading detection model...")
            # First try the experiment folder
            detection_model_path = backend_root / "outputs" / "experiment_20250916_210441" / "weights" / "best.pt"
            if detection_model_path.exists():
                detection_model = YOLO(str(detection_model_path))
                logger.info("✅ YOLO detection model loaded from experiment folder")
            else:
                # Try the models folder
                alt_path = backend_root / "outputs" / "models" / "best.pt"
                if alt_path.exists():
                    detection_model = YOLO(str(alt_path))
                    logger.info("✅ YOLO detection model loaded from models folder")
                else:
                    logger.warning("⚠️ YOLO model file not found, using pretrained model")
                    detection_model = YOLO('yolov8n.pt')  # Use pretrained model as fallback
        except Exception as e:
            logger.error(f"❌ Failed to load detection model: {e}")
            detection_model = None
    else:
        logger.warning("⚠️ YOLO not available")
        detection_model = None

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting FastAPI server...")
    await load_models()
    logger.info("✅ FastAPI server started successfully")
    yield
    # Shutdown
    logger.info("🛑 Shutting down FastAPI server...")

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware FIRST - before any routes or mounts
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving removed for separate deployment
# Frontend will be deployed separately

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🏔️ Rockfall Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational",
        "features": [
            "Risk prediction with ML ensemble",
            "Real-time rock detection",
            "WebSocket live monitoring",
            "File upload processing",
            "Historical data analysis"
        ]
    }

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and health"""
    global prediction_models, detection_model
    
    return SystemStatus(
        status="operational",
        models_loaded={
            "prediction_models": prediction_models is not None,
            "detection_model": detection_model is not None,
            "xgboost": "xgboost" in (prediction_models or {}),
            "random_forest": "random_forest" in (prediction_models or {}),
            "neural_network": "neural_network" in (prediction_models or {})
        },
        uptime="Running",
        active_connections=len(manager.active_connections)
    )

@app.post("/api/predict-risk", response_model=RiskPrediction)
async def predict_risk(data: EnvironmentalData):
    """Predict rockfall risk based on environmental data"""
    global prediction_models, scalers, feature_names
    
    if not prediction_models:
        # Provide mock prediction when models aren't available
        logger.warning("Using mock prediction - models not loaded")
        
        # Simple heuristic-based prediction for demo
        risk_factors = []
        risk_score = 0.0
        
        # Check slope
        if data.slope > 45:
            risk_score += 0.3
            risk_factors.append("steep_slope")
        
        # Check temperature and freeze-thaw
        if data.freeze_thaw_cycles > 10:
            risk_score += 0.2
            risk_factors.append("freeze_thaw_cycles")
        
        # Check seismic activity
        if data.seismic_activity > 3:
            risk_score += 0.25
            risk_factors.append("seismic_activity")
        
        # Check precipitation
        if data.rainfall > 100:
            risk_score += 0.15
            risk_factors.append("heavy_rainfall")
        
        # Check instability index
        risk_score += data.instability_index * 0.2
        if data.instability_index > 0.7:
            risk_factors.append("geological_instability")
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = "high"
        elif risk_score > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return RiskPrediction(
            risk_score=min(risk_score, 1.0),
            risk_level=risk_level,
            confidence=0.75,  # Mock confidence
            model_predictions={"mock": min(risk_score, 1.0)},
            recommendations=[
                "Monitor geological conditions regularly",
                "Install early warning systems", 
                "Restrict access during high-risk periods"
            ]
        )
    
    try:
        # Convert input data to array using feature names
        input_dict = data.dict()
        input_array = np.array([[input_dict[feature] for feature in feature_names]])
        
        # Make predictions using loaded models
        predictions = predict_with_loaded_models(prediction_models, scalers, input_array)
        
        # Calculate overall risk level
        ensemble_risk = predictions.get('ensemble', 0.0)
        
        # Handle NaN values
        if np.isnan(ensemble_risk) or not np.isfinite(ensemble_risk):
            logger.warning(f"Invalid ensemble risk value: {ensemble_risk}, using fallback")
            ensemble_risk = 0.0
        
        # Ensure risk is within valid range
        ensemble_risk = max(0.0, min(1.0, ensemble_risk))
        
        if ensemble_risk > 0.7:
            risk_level = "HIGH"
            recommendations = [
                "🚨 Immediate evacuation recommended",
                "⛔ Restrict access to danger zones", 
                "📞 Alert emergency services",
                "📊 Increase monitoring frequency"
            ]
        elif ensemble_risk > 0.3:
            risk_level = "MEDIUM"
            recommendations = [
                "⚠️ Enhanced monitoring required",
                "👥 Limit personnel in area",
                "📋 Prepare contingency plans",
                "🔍 Investigate risk factors"
            ]
        else:
            risk_level = "LOW"
            recommendations = [
                "✅ Normal operations can continue",
                "📈 Maintain regular monitoring",
                "📊 Review trends periodically"
            ]
        
        # Calculate confidence (based on model agreement)
        model_values = [v for k, v in predictions.items() if k != 'ensemble' and np.isfinite(v)]
        if model_values and len(model_values) > 0:
            std_dev = np.std(model_values)
            if np.isfinite(std_dev):
                confidence = max(0.0, min(1.0, 1.0 - std_dev))
            else:
                confidence = 0.5  # Default confidence when std calculation fails
        else:
            confidence = 0.5  # Default confidence when no valid model values
        
        # Ensure all prediction values are finite
        safe_predictions = {}
        for k, v in predictions.items():
            if np.isfinite(v):
                safe_predictions[k] = float(v)
            else:
                logger.warning(f"Invalid prediction value for {k}: {v}, setting to 0.0")
                safe_predictions[k] = 0.0
        
        result = RiskPrediction(
            risk_score=float(ensemble_risk),
            risk_level=risk_level,
            confidence=float(confidence),
            model_predictions=safe_predictions,
            recommendations=recommendations
        )
        
        # Broadcast to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "risk_update",
            "data": result.dict(),
            "timestamp": datetime.now().isoformat()
        }))
        
        return result
        
    except Exception as e:
        logger.error(f"Error in risk prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/detect-rocks", response_model=DetectionResult)
async def detect_rocks(file: UploadFile = File(...), confidence_threshold: float = 0.5):
    """Detect rocks in uploaded image"""
    global detection_model
    
    if not detection_model or not YOLO_AVAILABLE:
        # Provide mock detection when model isn't available
        logger.warning("Using mock detection - YOLO model not loaded")
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        try:
            # Read image to get dimensions
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            
            # Generate mock detections
            import random
            random.seed(42)  # For consistent mock results
            
            num_detections = random.randint(1, 5)
            mock_detections = []
            
            for i in range(num_detections):
                # Generate random bounding box
                x1 = random.randint(0, width // 2)
                y1 = random.randint(0, height // 2)
                x2 = x1 + random.randint(50, min(200, width - x1))
                y2 = y1 + random.randint(50, min(200, height - y1))
                
                mock_detections.append({
                    "confidence": random.uniform(0.6, 0.95),
                    "bbox": [x1, y1, x2, y2],
                    "class": "rock",
                    "class_id": 0
                })
            
            return DetectionResult(
                detections=mock_detections,
                total_detections=len(mock_detections),
                confidence_threshold=confidence_threshold,
                processing_time_ms=random.uniform(100, 500),
                image_dimensions={"width": width, "height": height},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in mock detection: {e}")
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL image to numpy array for YOLO
        import numpy as np
        import cv2
        
        # Convert PIL to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Run detection directly on numpy array
        start_time = datetime.now()
        results = detection_model(img_array, conf=confidence_threshold)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Process results
        detections = []
        total_detections = 0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        "confidence": float(box.conf[0]),
                        "bbox": [
                            float(box.xyxy[0][0]),
                            float(box.xyxy[0][1]), 
                            float(box.xyxy[0][2]),
                            float(box.xyxy[0][3])
                        ],
                        "class": "rock",
                        "class_id": 0,
                        "area": float((box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
                    }
                    detections.append(detection)
                    total_detections += 1
        
        result = DetectionResult(
            detections=detections,
            total_detections=total_detections,
            confidence_threshold=confidence_threshold,
            processing_time_ms=processing_time,
            image_dimensions={"width": image.width, "height": image.height}
        )
        
        # Broadcast detection result
        await manager.broadcast(json.dumps({
            "type": "detection_update",
            "data": result.dict(),
            "timestamp": datetime.now().isoformat()
        }))
        
        return result
            
    except Exception as e:
        logger.error(f"Error in rock detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/api/test-image")
async def get_test_image():
    """Get a default test image from the training data"""
    try:
        test_images_dir = backend_root / "data" / "rockfall_training_data" / "test" / "images"
        
        if not test_images_dir.exists():
            raise HTTPException(status_code=404, detail="Test images directory not found")
        
        # Get list of available test images
        image_files = list(test_images_dir.glob("*.jpg"))
        
        if not image_files:
            raise HTTPException(status_code=404, detail="No test images found")
        
        # Use the first test image as default
        default_image = image_files[0]
        
        return FileResponse(
            path=str(default_image),
            media_type="image/jpeg",
            filename=default_image.name
        )
        
    except Exception as e:
        logger.error(f"Error serving test image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve test image: {str(e)}")

@app.get("/api/test-image/detect", response_model=DetectionResult)
async def detect_default_test_image(confidence_threshold: float = 0.5):
    """Run detection on the default test image"""
    global detection_model
    
    try:
        test_images_dir = backend_root / "data" / "rockfall_training_data" / "test" / "images"
        
        if not test_images_dir.exists():
            raise HTTPException(status_code=404, detail="Test images directory not found")
        
        # Get list of available test images
        image_files = list(test_images_dir.glob("*.jpg"))
        
        if not image_files:
            raise HTTPException(status_code=404, detail="No test images found")
        
        # Use the first test image as default
        default_image_path = image_files[0]
        
        if not detection_model or not YOLO_AVAILABLE:
            # Provide mock detection when model isn't available
            logger.warning("Using mock detection - YOLO model not loaded")
            
            # Read image to get dimensions
            image = Image.open(default_image_path)
            width, height = image.size
            
            # Generate mock detections
            import random
            random.seed(42)  # For consistent mock results
            
            num_detections = random.randint(2, 6)
            mock_detections = []
            
            for i in range(num_detections):
                # Generate random bounding box
                x1 = random.randint(0, width // 2)
                y1 = random.randint(0, height // 2)
                x2 = x1 + random.randint(50, min(200, width - x1))
                y2 = y1 + random.randint(50, min(200, height - y1))
                
                mock_detections.append({
                    "confidence": random.uniform(0.7, 0.95),
                    "bbox": [x1, y1, x2, y2],
                    "class": "rock",
                    "class_id": 0
                })
            
            return DetectionResult(
                detections=mock_detections,
                total_detections=len(mock_detections),
                confidence_threshold=confidence_threshold,
                processing_time_ms=random.uniform(100, 500),
                image_dimensions={"width": width, "height": height},
                timestamp=datetime.now().isoformat()
            )
        
        # Load and process the test image
        image = Image.open(default_image_path)
        
        # Convert PIL image to numpy array for YOLO
        import numpy as np
        
        # Convert PIL to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Run detection
        start_time = datetime.now()
        results = detection_model(img_array, conf=confidence_threshold)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Process results
        detections = []
        total_detections = 0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        "confidence": float(box.conf[0]),
                        "bbox": [
                            float(box.xyxy[0][0]),
                            float(box.xyxy[0][1]), 
                            float(box.xyxy[0][2]),
                            float(box.xyxy[0][3])
                        ],
                        "class": "rock",
                        "class_id": 0,
                        "area": float((box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
                    }
                    detections.append(detection)
                    total_detections += 1
        
        result = DetectionResult(
            detections=detections,
            total_detections=total_detections,
            confidence_threshold=confidence_threshold,
            processing_time_ms=processing_time,
            image_dimensions={"width": image.width, "height": image.height},
            timestamp=datetime.now().isoformat()
        )
        
        # Note: Not broadcasting test image detections to avoid duplicate notifications
        # This is a demo/test endpoint, real detections should use /api/detect-rocks
        
        return result
        
    except Exception as e:
        logger.error(f"Error in default test image detection: {e}")
        raise HTTPException(status_code=500, detail=f"Default detection failed: {str(e)}")

@app.get("/api/models/performance")
async def get_model_performance():
    """Get model performance metrics"""
    global model_performance
    
    if not model_performance:
        raise HTTPException(status_code=503, detail="Model performance data not available")
    
    return {
        "model_performance": model_performance,
        "detection_model": {
            "mAP50": 0.995,
            "precision": 0.9952,
            "recall": 1.0,
            "inference_time_ms": 60.8
        },
        "feature_count": len(feature_names) if feature_names else 0
    }

@app.get("/api/features")
async def get_feature_names():
    """Get list of required features for prediction"""
    global feature_names
    
    if not feature_names:
        raise HTTPException(status_code=503, detail="Feature names not available")
    
    return {
        "features": feature_names,
        "total_features": len(feature_names),
        "description": "Environmental and terrain features required for risk prediction"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            # Echo back for heartbeat
            await manager.send_personal_message(json.dumps({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
                "message": "Connection active"
            }), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Server-Sent Events (SSE) alternative for platforms that don't support WebSockets
@app.get("/api/events/stream")
async def event_stream():
    """Server-Sent Events endpoint - WebSocket alternative for cloud platforms"""
    async def generate():
        try:
            while True:
                # Send real-time data as SSE
                data = {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                    "message": "Connection active",
                    "camera_status": video_manager.get_camera_status(),
                    "system_stats": {
                        "active_connections": len(manager.active_connections),
                        "uptime": "24h 15m",
                        "memory_usage": "245 MB"
                    }
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(5)  # Send updates every 5 seconds
                
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.post("/api/events/message")
async def receive_message(message: dict):
    """Receive messages from SSE clients (since SSE is one-way)"""
    logger.info(f"Received SSE message: {message}")
    
    # Broadcast to WebSocket clients if any are connected
    if manager.active_connections:
        await manager.broadcast(json.dumps({
            "type": "client_message",
            "timestamp": datetime.now().isoformat(),
            "data": message
        }))
    
    return {"status": "received", "timestamp": datetime.now().isoformat()}

# Camera streaming endpoints
@app.get("/api/camera/status")
async def get_camera_status():
    """Get status of all cameras with real video metadata"""
    return video_manager.get_camera_status()

@app.get("/api/camera/{direction}/stream")
async def get_camera_stream(direction: str):
    """Get camera stream URL for specified direction"""
    valid_directions = ["east", "west", "north", "south"]
    
    if direction not in valid_directions:
        raise HTTPException(status_code=400, detail="Invalid camera direction")
    
    status = video_manager.get_camera_status()
    camera_info = status["cameras"][direction]
    
    return {
        "direction": direction,
        "stream_url": f"/api/camera/{direction}/feed",
        "status": camera_info["status"],
        "resolution": camera_info["resolution"],
        "fps": camera_info["fps"],
        "duration": camera_info["duration"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/camera/{direction}/feed")
async def get_camera_feed(direction: str):
    """Get live video feed for specified camera direction"""
    valid_directions = ["east", "west", "north", "south"]
    
    if direction not in valid_directions:
        raise HTTPException(status_code=400, detail="Invalid camera direction")
    
    if direction not in video_manager.video_captures:
        raise HTTPException(status_code=404, detail=f"Camera {direction} not available")
    
    # Start streaming for this direction
    video_manager.start_stream(direction)
    
    return StreamingResponse(
        video_manager.generate_frames(direction),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/api/camera/{direction}/control")
async def control_camera(direction: str, action: str):
    """Control camera operations (start/stop/record/etc.)"""
    valid_directions = ["east", "west", "north", "south"]
    valid_actions = ["start", "stop", "record", "stop_record", "zoom_in", "zoom_out", "rotate_left", "rotate_right"]
    
    if direction not in valid_directions:
        raise HTTPException(status_code=400, detail="Invalid camera direction")
    
    if action not in valid_actions:
        raise HTTPException(status_code=400, detail="Invalid camera action")
    
    # Handle real camera control
    success = False
    message = ""
    
    if action == "start":
        success = video_manager.start_stream(direction)
        message = "Stream started" if success else "Failed to start stream"
    elif action == "stop":
        success = video_manager.stop_stream(direction)
        message = "Stream stopped" if success else "Failed to stop stream"
    elif action in ["record", "stop_record"]:
        # Simulate recording control
        success = True
        message = f"Recording {'started' if action == 'record' else 'stopped'}"
    else:
        # Simulate other camera controls
        success = True
        message = f"Camera {direction} {action} executed successfully"
    
    return {
        "direction": direction,
        "action": action,
        "status": "success" if success else "error",
        "message": message,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/camera/{direction}/detections")
async def get_camera_detections(direction: str):
    """Get recent detections from specific camera"""
    valid_directions = ["east", "west", "north", "south"]
    
    if direction not in valid_directions:
        raise HTTPException(status_code=400, detail="Invalid camera direction")
    
    # Simulate detection data
    detections = []
    if direction == "north":
        detections = [
            {
                "id": "det_001",
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.95,
                "bbox": {"x": 120, "y": 80, "width": 60, "height": 40},
                "object_type": "rock",
                "risk_level": "medium"
            },
            {
                "id": "det_002", 
                "timestamp": (datetime.now()).isoformat(),
                "confidence": 0.87,
                "bbox": {"x": 300, "y": 150, "width": 45, "height": 35},
                "object_type": "rock",
                "risk_level": "low"
            }
        ]
    
    return {
        "direction": direction,
        "detections": detections,
        "total_count": len(detections),
        "last_updated": datetime.now().isoformat()
    }

# DEM Analysis endpoints
@app.get("/api/dem/files")
async def get_dem_files():
    """Get list of available DEM files"""
    logger.info("📁 DEM files endpoint requested")
    
    # Use paths relative to backend folder for deployment compatibility
    backend_root = Path(__file__).parent  # This is the backend/ directory
    
    dem_files = [
        {
            "id": "bingham_canyon",
            "name": "Bingham Canyon Mine",
            "location": "Utah, USA",
            "description": "Large open-pit copper mine with significant terrain variations",
            "file_path": str(backend_root / "data" / "DEM" / "Bingham_Canyon_Mine.tif")
        },
        {
            "id": "chuquicamata",
            "name": "Chuquicamata Copper Mine", 
            "location": "Chile",
            "description": "One of the largest open-pit mines in the world",
            "file_path": str(backend_root / "data" / "DEM" / "Chuquicamata_copper_Mine.tif")
        },
        {
            "id": "grasberg",
            "name": "Grasberg Mine",
            "location": "Papua, Indonesia", 
            "description": "High-altitude mining operation in mountainous terrain",
            "file_path": str(backend_root / "data" / "DEM" / "Grasberg_Mine_Indonesia.tif")
        }
    ]
    
    # Log file existence check
    for dem_file in dem_files:
        file_path = Path(dem_file["file_path"])
        exists = file_path.exists()
        logger.info(f"📄 DEM file {dem_file['id']}: {file_path} -> {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        if exists:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"   Size: {size_mb:.1f} MB")
    
    logger.info(f"📤 Returning {len(dem_files)} DEM files")
    return {"files": dem_files}

@app.get("/api/dem/analyze/{dem_id}")
async def analyze_dem(dem_id: str):
    """Analyze DEM file and return color-coded visualization with statistics"""
    logger.info(f"🗺️ DEM analysis requested for: {dem_id}")
    
    try:
        # Map DEM IDs to file paths - DEM files are in backend/data/DEM/
        # Use paths relative to backend folder for deployment compatibility
        backend_root = Path(__file__).parent  # This is the backend/ directory
        dem_files = {
            "bingham_canyon": backend_root / "data" / "DEM" / "Bingham_Canyon_Mine.tif",
            "chuquicamata": backend_root / "data" / "DEM" / "Chuquicamata_copper_Mine.tif", 
            "grasberg": backend_root / "data" / "DEM" / "Grasberg_Mine_Indonesia.tif"
        }
        
        logger.info(f"📋 Available DEM files: {list(dem_files.keys())}")
        
        if dem_id not in dem_files:
            logger.error(f"❌ Invalid DEM file ID: {dem_id}")
            raise HTTPException(status_code=400, detail="Invalid DEM file ID")
        
        file_path = dem_files[dem_id]  # Now using absolute Path objects
        logger.info(f"📁 Resolved file path: {file_path}")
        logger.info(f"📁 Absolute file path: {file_path.absolute()}")
        logger.info(f"📂 Backend root: {backend_root}")
        
        if not file_path.exists():
            logger.error(f"❌ DEM file not found at: {file_path}")
            # List what's actually in the directory
            dem_dir = file_path.parent
            if dem_dir.exists():
                logger.info(f"📂 Contents of {dem_dir}:")
                for item in dem_dir.iterdir():
                    logger.info(f"   📄 {item.name}")
            else:
                logger.error(f"❌ Directory {dem_dir} does not exist")
            raise HTTPException(status_code=404, detail="DEM file not found")
        
        logger.info(f"✅ DEM file found, starting analysis...")
        
        # Process DEM file and generate color-coded visualization
        result = await process_dem_file(file_path, dem_id)
        
        logger.info(f"✅ DEM analysis completed for {dem_id}")
        
        return {
            "dem_id": dem_id,
            "image_url": result["image_url"],
            "statistics": result["statistics"],
            "processing_time": result["processing_time"],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"💥 DEM analysis failed for {dem_id}: {str(e)}")
        logger.exception("Full error traceback:")
        raise HTTPException(status_code=500, detail=f"DEM analysis failed: {str(e)}")

async def process_dem_file(file_path: Path, dem_id: str):
    """Process DEM .tif file and generate color-coded PNG visualization"""
    logger.info(f"🔬 Processing DEM file: {file_path}")
    
    try:
        # Try to import required libraries
        try:
            import rasterio
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            from matplotlib.colors import LinearSegmentedColormap
            import numpy as np
            from PIL import Image
            import io
            import base64
            logger.info("✅ All geospatial libraries imported successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Geospatial libraries not available: {e}")
            logger.info("🔄 Returning mock data instead")
            # Return mock data when libraries are missing
            return {
                "image_url": f"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "statistics": {
                    "min_elevation": 1000.0,
                    "max_elevation": 3000.0,
                    "mean_elevation": 2000.0,
                    "std_elevation": 500.0,
                    "area_km2": 25.0,
                    "slope_analysis": {
                        "gentle_slopes": 30.0,
                        "moderate_slopes": 45.0, 
                        "steep_slopes": 25.0
                    },
                    "risk_zones": {
                        "low_risk": 40.0,
                        "medium_risk": 35.0,
                        "high_risk": 25.0
                    }
                },
                "processing_time": 0.1
            }
        
        start_time = datetime.now()
        
        # Read DEM data
        with rasterio.open(file_path) as dataset:
            elevation_data = dataset.read(1)
            # Handle nodata values
            if dataset.nodata is not None:
                elevation_data = np.ma.masked_where(elevation_data == dataset.nodata, elevation_data)
            else:
                # If no nodata value specified, mask extreme values
                elevation_data = np.ma.masked_where(
                    (elevation_data < -1000) | (elevation_data > 10000), 
                    elevation_data
                )
        
        # Calculate statistics
        valid_data = elevation_data.compressed()  # Remove masked values
        if len(valid_data) == 0:
            raise ValueError("No valid elevation data found in DEM file")
            
        stats = {
            "min_elevation": round(float(np.min(valid_data)), 1),
            "max_elevation": round(float(np.max(valid_data)), 1),
            "mean_elevation": round(float(np.mean(valid_data)), 1),
            "std_elevation": round(float(np.std(valid_data)), 1),
            "elevation_range": round(float(np.max(valid_data) - np.min(valid_data)), 1)
        }
        
        # Add terrain classification
        elevation_range = stats["elevation_range"]
        if elevation_range > 1000:
            stats["terrain_type"] = "Mountainous"
            stats["risk_level"] = "High"
        elif elevation_range > 500:
            stats["terrain_type"] = "Hilly"
            stats["risk_level"] = "Moderate to High"
        elif elevation_range > 100:
            stats["terrain_type"] = "Rolling"
            stats["risk_level"] = "Moderate"
        else:
            stats["terrain_type"] = "Flat"
            stats["risk_level"] = "Low"
        
        # Create custom colormap: Green (low) → Yellow → Brown → White (high)
        colors_list = [
            '#2D5016',  # Dark Green (lowest)
            '#4F7942',  # Green
            '#8FBC8F',  # Light Green  
            '#DAA520',  # Gold/Yellow
            '#CD853F',  # Peru/Brown
            '#A0522D',  # Sienna/Dark Brown
            '#FFFFFF'   # White (highest)
        ]
        
        n_bins = 256
        terrain_cmap = LinearSegmentedColormap.from_list(
            'terrain', colors_list, N=n_bins
        )
        
        # Create the plot with proper DPI and size
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        
        # Plot elevation data with custom colormap
        im = ax.imshow(
            elevation_data, 
            cmap=terrain_cmap,
            interpolation='bilinear',
            aspect='equal'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
        cbar.set_label('Elevation (meters)', color='white', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(colors='white', labelsize=10)
        
        # Styling
        title = dem_id.replace("_", " ").title()
        ax.set_title(f'{title} - Digital Elevation Model', 
                    color='white', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')  # Remove axes for cleaner look
        
        # Add text box with statistics
        textstr = f'''Elevation Statistics:
Min: {stats["min_elevation"]} m
Max: {stats["max_elevation"]} m  
Mean: {stats["mean_elevation"]} m
Range: {stats["elevation_range"]} m
Terrain: {stats["terrain_type"]}'''
        
        props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='white')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', color='white', bbox=props, fontweight='bold')
        
        plt.tight_layout()
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', facecolor='#0f172a', 
                   bbox_inches='tight', dpi=100, edgecolor='none')
        img_buffer.seek(0)
        plt.close()
        
        # Convert to base64 for web display
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        image_url = f"data:image/png;base64,{img_base64}"
        
        # Also save as file for download (optional)
        try:
            output_dir = backend_root / "outputs" / "dem_visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{dem_id}_elevation_map.png"
            with open(output_file, 'wb') as f:
                f.write(img_buffer.getvalue())
            logger.info(f"DEM visualization saved to {output_file}")
        except Exception as save_error:
            logger.warning(f"Could not save DEM file to disk: {save_error}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "image_url": image_url,
            "statistics": stats,
            "processing_time": f"{processing_time:.2f}s"
        }
        
    except ImportError as e:
        missing_lib = str(e).split("'")[1] if "'" in str(e) else "required library"
        raise HTTPException(
            status_code=500, 
            detail=f"Missing required library: {missing_lib}. Please install: pip install rasterio matplotlib"
        )
    except Exception as e:
        logger.error(f"DEM processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DEM processing failed: {str(e)}")

@app.get("/api/simulate-data")
async def simulate_environmental_data():
    """Generate sample environmental data for testing"""
    sample_data = {
        "slope": 45.0,
        "elevation": 1500.0,
        "fracture_density": 3.5,
        "roughness": 0.7,
        "slope_variability": 0.3,
        "instability_index": 0.8,
        "wetness_index": 0.6,
        "month": 9.0,
        "day_of_year": 259.0,
        "season": 2.0,
        "rainfall": 50.0,
        "temperature": 15.0,
        "temperature_variation": 10.0,
        "freeze_thaw_cycles": 5.0,
        "seismic_activity": 2.0,
        "wind_speed": 30.0,
        "precipitation_intensity": 25.0,
        "humidity": 75.0,
        "risk_score": 0.0
    }
    
    return {
        "sample_data": sample_data,
        "description": "Sample environmental data for testing the prediction API"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level=LOG_LEVEL.lower()
    )