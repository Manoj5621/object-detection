from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model

# Train the model with custom hyperparameters
results = model.train(
    data='custom_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    lrf=0.001,
    iou=0.6,  # IoU threshold for training
    conf=0.001,  # confidence threshold for training
    patience=20,  # early stopping patience
    device=0  # GPU device
)

# Run inference with custom parameters
results = model.predict(
    source='test_images/',
    conf=0.35,  # confidence threshold
    iou=0.5,  # NMS IoU threshold
    max_det=300  # maximum detections per image
)