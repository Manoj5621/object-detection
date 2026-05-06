from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("best.pt")

# Try to open camera
camera_index = 0
cap = cv2.VideoCapture(camera_index)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create display window
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

# Main detection loop
while True:
    # Read frame
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    # Run detection and get annotated frame
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # Display result
    cv2.imshow("YOLO Detection", annotated_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()