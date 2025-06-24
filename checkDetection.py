from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model (replace with your actual path)
model = YOLO(r"C:\Users\prati\OneDrive\Desktop\runs\detect\train\weights\best.pt")


# Your IP webcam video stream URL (change to your IP webcam URL)
ip_webcam_url = "http://172.20.10.3:8080/video"  # Example URL

# Open video capture with IP webcam URL
cap = cv2.VideoCapture(ip_webcam_url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection on the frame
    results = model(frame)

    # Draw detections on the frame
    annotated_frame = results[0].plot()

    # Show the frame with detections
    cv2.imshow("YOLOv8 Parking Slot Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()