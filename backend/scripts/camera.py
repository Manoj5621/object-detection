import cv2
import os
import time
from datetime import datetime

# Create directory if it doesn't exist
path = 'data/Images/'
os.makedirs(path, exist_ok=True)

# Try to open webcam with different indices if first attempt fails
webcam_index = 0
max_attempts = 3
capture = None

while webcam_index < max_attempts and (capture is None or not capture.isOpened()):
    capture = cv2.VideoCapture(webcam_index)
    if not capture.isOpened():
        print(f"Failed to open webcam at index {webcam_index}")
        webcam_index += 1
        time.sleep(1)

if not capture.isOpened():
    print("Error: Could not open any webcam. Please check your hardware.")
    exit()
else:
    print(f"Successfully opened webcam at index {webcam_index}")

# Get webcam properties
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv2.CAP_PROP_FPS)
print(f"Webcam resolution: {frame_width}x{frame_height}, FPS: {fps}")

counter = 1
timer_active = False
timer_duration = 3  # seconds
timer_start = 0

print("\nControls:")
print("- Press 's' to save an image manually")
print("- Press 't' to activate a 3-second timer before saving")
print("- Press '+' or '-' to adjust brightness")
print("- Press 'Esc' to exit")

# Initialize brightness adjustment
brightness = 0

while True:
    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break
    
    # Apply brightness adjustment if needed
    if brightness != 0:
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness)
    
    # Show countdown if timer is active
    current_time = time.time()
    if timer_active:
        seconds_left = max(0, int(timer_duration - (current_time - timer_start)))
        cv2.putText(frame, f"Taking photo in: {seconds_left}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if current_time - timer_start >= timer_duration:
            # Timer complete, save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = os.path.join(path, f'image_{counter}_{timestamp}.jpg')
            success = cv2.imwrite(image_name, frame)
            
            if success:
                print(f'image_{counter}_{timestamp}.jpg saved to {path}')
                counter += 1
            else:
                print(f'Failed to save image_{counter}_{timestamp}.jpg')
                
            timer_active = False
    
    # Display the current frame with counter
    display_text = f'Collecting Images ({counter-1} saved)'
    if brightness != 0:
        display_text += f' | Brightness: {brightness}'
    cv2.putText(frame, display_text, (20, frame_height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Webcam Image Collector', frame)

    # Wait for key press
    k = cv2.waitKey(1)
    
    # Exit on ESC
    if k % 256 == 27:
        print("Exiting program")
        break
    
    # Save image on 's' key
    elif k % 256 == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = os.path.join(path, f'image_{counter}_{timestamp}.jpg')
        
        success = cv2.imwrite(image_name, frame)
        
        if success:
            print(f'image_{counter}_{timestamp}.jpg saved to {path}')
            counter += 1
        else:
            print(f'Failed to save image_{counter}_{timestamp}.jpg')
    
    # Activate timer on 't' key
    elif k % 256 == ord('t'):
        timer_active = True
        timer_start = time.time()
        print("Timer activated - photo will be taken in 3 seconds")
    
    # Adjust brightness
    elif k % 256 == ord('+') or k % 256 == ord('='):
        brightness += 5
        print(f"Brightness increased to {brightness}")
    elif k % 256 == ord('-') or k % 256 == ord('_'):
        brightness -= 5
        print(f"Brightness decreased to {brightness}")

# Release resources
capture.release()
cv2.destroyAllWindows()
print(f"Session complete. {counter-1} images saved to {path}")