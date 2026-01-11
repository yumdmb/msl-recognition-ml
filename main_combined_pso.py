"""
MSL Recognition API (PSO-Optimized Combined Model)
FastAPI service for real-time Malaysian Sign Language recognition.
Uses PSO-optimized classifier for best accuracy.
Supports all 44 classes: Alphabet (A-Z), Numbers (0-10), Words (7).
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64

from realtime_predict_combined_pso import MSLCombinedRecognizerPSO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global recognizer instance
recognizer: Optional[MSLCombinedRecognizerPSO] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app - loads models on startup"""
    global recognizer
    logger.info("Loading MSL PSO-optimized combined recognition models...")
    try:
        recognizer = MSLCombinedRecognizerPSO()
        logger.info("Models loaded successfully!")
        logger.info(f"Total classes: {len(recognizer.labels)}")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    yield
    # Cleanup on shutdown
    logger.info("Shutting down MSL Recognition API (PSO Combined)...")


# FastAPI app with metadata
app = FastAPI(
    title="MSL Recognition API (PSO Combined)",
    description="Real-time Malaysian Sign Language recognition using PSO-optimized combined model. Best accuracy. Supports Alphabet (A-Z), Numbers (0-10), and Words.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": str(type(exc).__name__)}
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MSL Recognition API (PSO Combined)",
        "version": "1.0.0",
        "model": "combined_pso (44 classes)",
        "optimization": "PSO (Particle Swarm Optimization)",
        "categories": {
            "alphabet": "A-Z (26 classes)",
            "number": "0-10 (11 classes)",
            "word": "7 common words"
        },
        "status": "running",
        "endpoints": {
            "health": "/health",
            "classes": "/classes",
            "predict": "/predict-image/",
            "websocket": "/ws/recognize",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring"""
    return {
        "status": "healthy",
        "model_loaded": recognizer is not None,
        "model_type": "combined_pso",
        "total_classes": len(recognizer.labels) if recognizer else 0,
        "service": "msl-recognition-api-combined-pso"
    }


@app.get("/classes")
async def list_classes():
    """List all supported sign classes"""
    if not recognizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    classes = list(recognizer.labels.keys())
    
    # Categorize classes
    alphabet = sorted([c for c in classes if len(c) == 1 and c.isalpha()])
    numbers = sorted([c for c in classes if c.startswith('NUM_')])
    words = sorted([c for c in classes if len(c) > 1 and not c.startswith('NUM_')])
    
    return {
        "total_classes": len(classes),
        "categories": {
            "alphabet": {"count": len(alphabet), "classes": alphabet},
            "number": {"count": len(numbers), "classes": [c[4:] for c in numbers]},
            "word": {"count": len(words), "classes": words}
        }
    }


@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    """
    POST endpoint for single image prediction using PSO-optimized model.
    Auto-detects whether the sign is an alphabet, number, or word.
    
    - **file**: Image file (JPEG, PNG, or WebP)
    - **Returns**: Predicted sign, confidence, and sign type
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 10MB")
    
    try:
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        label, confidence, sign_type = recognizer.predict_from_image(frame)
        
        return {
            "success": True,
            "label": label,
            "confidence": confidence,
            "sign_type": sign_type,
            "model": "pso_optimized",
            "message": f"Detected: {label} ({sign_type})" if label else "No hand detected"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.websocket("/ws/recognize")
async def recognize_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time MSL recognition using PSO model.
    Accepts base64-encoded JPEG frames, returns prediction with sign type.
    
    Connection: ws://host/ws/recognize
    Send: base64 encoded JPEG image
    Receive: {"label": "A", "confidence": 0.95, "sign_type": "ALPHABET", "success": true}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")
    
    frames_processed = 0
    try:
        while True:
            data = await websocket.receive_text()
            
            # Handle ping/pong for connection keep-alive
            if data == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({
                        "success": False,
                        "error": "Invalid frame data"
                    })
                    continue

                label, confidence, sign_type = recognizer.predict_from_image(frame)
                frames_processed += 1
                
                await websocket.send_json({
                    "success": True,
                    "label": label,
                    "confidence": confidence,
                    "sign_type": sign_type,
                    "model": "pso_optimized",
                    "frame_count": frames_processed
                })
                
            except base64.binascii.Error:
                await websocket.send_json({
                    "success": False,
                    "error": "Invalid base64 encoding"
                })
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                await websocket.send_json({
                    "success": False,
                    "error": "Processing failed"
                })
                
    except Exception as e:
        logger.info(f"WebSocket disconnected: {websocket.client} - {e}")
    finally:
        logger.info(f"Total frames processed: {frames_processed}")
        try:
            await websocket.close()
        except:
            pass
