import cv2
import torch
from ultralytics import YOLO
import open_clip
from PIL import Image
import numpy as np
import time

# ==============================
# CONFIGURATION
# ==============================
YOLO_CONFIDENCE = 0.5          # High confidence threshold for YOLO
CLIP_THRESHOLD = 0.3           # Low confidence - use CLIP for refinement
ENABLE_CLIP = True             # Toggle CLIP on/off

# ==============================
# DEVICE SETUP
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {device.upper()}")

# ==============================
# LOAD MODELS
# ==============================

# YOLO Model - Change to yolov8s.pt for better accuracy
print("📦 Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")  # Use yolov8s.pt after fine-tuning on your data
yolo_model.to(device)

# CLIP Model - For refining low-confidence detections
print("📦 Loading CLIP model...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model.to(device)
clip_model.eval()  # Set to evaluation mode for inference

# ==============================
# YOUR CUSTOM LABELS
# ==============================
# These should match your dataset.yaml classes
CUSTOM_LABELS = ["box", "card", "key", "lau", "mouse", "pen", "wallet"]

# ==============================
# CLIP REFINEMENT FUNCTION
# ==============================
@torch.no_grad()
def refine_with_clip(frame_crop, yolo_label, yolo_conf):
    """
    Use CLIP to refine low-confidence YOLO detections
    
    Args:
        frame_crop: Cropped image from bounding box
        yolo_label: YOLO's predicted label
        yolo_conf: YOLO's confidence score
    
    Returns:
        refined_label: Most confident CLIP prediction from custom labels
        clip_conf: CLIP confidence score
    """
    try:
        # Convert to PIL and preprocess
        if frame_crop.size == 0:
            return yolo_label, yolo_conf
        
        image = Image.fromarray(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB))
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Tokenize labels
        text_tokens = open_clip.tokenize(CUSTOM_LABELS).to(device)
        
        # Get embeddings
        image_features = clip_model.encode_image(image_tensor)
        text_features = clip_model.encode_text(text_tokens)
        
        # Normalize and compute similarities
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarities = (image_features @ text_features.T).softmax(dim=-1)
        clip_conf, best_idx = similarities.max(dim=-1)
        
        refined_label = CUSTOM_LABELS[best_idx.item()]
        
        return refined_label, float(clip_conf[0])
    
    except Exception as e:
        print(f"⚠️  CLIP error: {e}")
        return yolo_label, yolo_conf

# ==============================
# DINO FUNCTION - REMOVED (Too Slow)
# ==============================

# ==============================
# MAIN DETECTION LOOP
# ==============================
print("\n🚀 Starting detection (Press 'q' to quit)")
print(f"✅ YOLO Confidence Threshold: {YOLO_CONFIDENCE}")
print(f"✅ CLIP Refinement Active: {ENABLE_CLIP}")
print(f"✅ CLIP Threshold: {CLIP_THRESHOLD}")

cap = cv2.VideoCapture(0)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    h, w = frame.shape[:2]
    
    # Run YOLO detection
    results = yolo_model(frame, conf=0.25, device=device)
    
    detections_info = []  # Track all detections for display
    
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Clip boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            yolo_label = yolo_model.names[int(box.cls[0])]
            
            # ================================================
            # DECISION: HIGH CONFIDENCE → Trust YOLO
            # ================================================
            if conf >= YOLO_CONFIDENCE:
                final_label = yolo_label
                final_conf = conf
                color = (0, 255, 0)      # Green - High confidence
                model_source = "YOLO"
            
            # ================================================
            # DECISION: LOW CONFIDENCE → Refine with CLIP
            # ================================================
            elif conf >= CLIP_THRESHOLD and ENABLE_CLIP:
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    refined_label, clip_conf = refine_with_clip(crop, yolo_label, conf)
                    final_label = refined_label
                    final_conf = clip_conf
                else:
                    final_label = yolo_label
                    final_conf = conf
                
                color = (255, 165, 0)    # Orange - CLIP refinement
                model_source = "CLIP"
            
            # ================================================
            # DECISION: VERY LOW CONFIDENCE → Ignore
            # ================================================
            else:
                continue
            
            # Store detection info
            detections_info.append({
                'bbox': (x1, y1, x2, y2),
                'label': final_label,
                'conf': final_conf,
                'color': color,
                'source': model_source,
                'yolo_conf': conf
            })
    
    # ==============================
    # DRAW RESULTS ON FRAME
    # ==============================
    for det in detections_info:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        conf = det['conf']
        color = det['color']
        source = det['source']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare display text
        if source == "YOLO":
            display_text = f"{label} {conf:.2f}"
        else:  # CLIP
            display_text = f"{label} (CLIP: {conf:.2f})"
        
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
        
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                     (x1 + text_size[0] + 5, y1), color, -1)
        cv2.putText(frame, display_text, (x1 + 2, y1 - 5), 
                   font, font_scale, (0, 0, 0), thickness)
    
    # ==============================
    # DISPLAY STATS
    # ==============================
    elapsed = time.time() - start_time
    current_fps = frame_count / elapsed if elapsed > 0 else 0
    
    stats_text = f"FPS: {current_fps:.1f} | Detections: {len(detections_info)}"
    cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 255, 0), 2)
    
    cv2.putText(frame, "Press 'q' to quit", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (200, 200, 200), 1)
    
    # Show frame
    cv2.imshow("YOLO + CLIP Hybrid Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Detection completed!")
print(f"Total frames processed: {frame_count}")
print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")