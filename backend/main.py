from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import base64
import torch
import json
from typing import List, Dict

# ==============================
# INITIALIZE FASTAPI APP
# ==============================
app = FastAPI(title="Object Detection API", version="1.0")

# ==============================
# CORS CONFIGURATION (Fix Frontend Connection)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domain in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# LOAD MODEL ONCE (Performance)
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {device.upper()}")
print("📦 Loading YOLO model...")
model = YOLO("yolov8m.pt")
model.to(device)
print("✅ Model loaded successfully!")

# ==============================
# CONFIDENCE THRESHOLDS
# ==============================
CONFIDENCE_THRESHOLD = 0.5

# ==============================
# HELPER FUNCTIONS
# ==============================
def prepare_detection_response(results, img_shape=None):
    """
    Parse YOLO results and return structured data
    """
    detections = []
    
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            
            # Only include detections above threshold
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                
                detections.append({
                    "class": label,
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "width": round(x2 - x1, 2),
                        "height": round(y2 - y1, 2)
                    }
                })
    
    return detections

# ==============================
# HEALTH CHECK ENDPOINT
# ==============================
@app.get("/")
async def root():
    return {"status": "✅ Backend API Running", "device": device}

# ==============================
# IMAGE DETECTION API
# ==============================
@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    """
    Detect objects in uploaded image
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image file", "detections": []}
        
        # Run detection
        results = model(img, conf=0.25, device=device)
        detections = prepare_detection_response(results)
        
        return {
            "status": "success",
            "detections": detections,
            "objects": list(set([d["class"] for d in detections])),
            "image_shape": list(img.shape)
        }
    
    except Exception as e:
        return {"error": str(e), "detections": []}

# ==============================
# VIDEO DETECTION API
# ==============================
@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    """
    Detect objects in uploaded video (sample frames)
    """
    try:
        contents = await file.read()
        temp_path = "temp_video.mp4"
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        cap = cv2.VideoCapture(temp_path)
        all_detections = []
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for speed
            if frame_count % 5 == 0:
                results = model(frame, conf=0.25, device=device)
                detections = prepare_detection_response(results)
                all_detections.extend(detections)
                processed_frames += 1
            
            frame_count += 1
        
        cap.release()
        
        # Remove duplicate detections
        unique_objects = list(set([d["class"] for d in all_detections]))
        
        return {
            "status": "success",
            "detections": all_detections,
            "objects": unique_objects,
            "total_frames": frame_count,
            "processed_frames": processed_frames
        }
    
    except Exception as e:
        return {"error": str(e), "detections": []}

# ==============================
# WEBSOCKET FOR LIVE CAMERA
# ==============================
@app.websocket("/ws/detect-camera")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for live camera detection
    Frontend sends frame data, backend returns detections
    """
    await websocket.accept()
    print("✅ Client connected to live camera stream")
    
    try:
        while True:
            # Receive frame data from frontend (base64 encoded image)
            data = await websocket.receive_text()
            
            try:
                # Decode base64 image
                image_data = base64.b64decode(data.split(",")[1])
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Run detection
                results = model(frame, conf=0.25, device=device)
                detections = prepare_detection_response(results)
                
                # Send detections back
                response = {
                    "detections": detections,
                    "objects": list(set([d["class"] for d in detections])),
                    "frame_shape": list(frame.shape)
                }
                
                await websocket.send_json(response)
            
            except Exception as e:
                print(f"⚠️  Detection error: {e}")
                await websocket.send_json({"error": str(e), "detections": []})
    
    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.close(code=1000)
