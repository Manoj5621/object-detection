from ultralytics import YOLO
import os

# Verify the dataset paths
base_path = r"C:\Users\defaultuser0\Documents\object detection"
train_path = os.path.join(base_path, 'Images', 'train')
val_path = os.path.join(base_path, 'Images', 'val')

# Ensure paths exist
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)

# Create the model
model = YOLO("yolov8n.yaml")

# Train the model
results = model.train(
    data='conf.yaml',  # Use the YAML configuration file
    epochs=10,
    imgsz=1254,
    batch=50,
    device='cpu'  # Use CPU if you don't have a compatible GPU
)

# Save the trained model
model.save('yolov8_trained.pt')