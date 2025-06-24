from ultralytics import YOLO
import cv2

# Load YOLO model once
model = YOLO(r"C:\Users\prati\OneDrive\Desktop\runs\detect\train\weights\best.pt")


# Your IP webcam stream
ip_webcam_url = "http://172.20.10.3:8080/video"  # Replace with your actual URL


def detect_parking_slots():
    cap = cv2.VideoCapture(ip_webcam_url)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return [False] * 6  # return 6 slots default False

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to grab frame")
        return [False] * 6  # return 6 slots default False

    results = model(frame)

    num_slots = 6  # changed to 6 slots
    detections = results[0].boxes
    occupied = [False] * num_slots

    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf > 0.5:
            for i in range(len(occupied)):
                if not occupied[i]:
                    occupied[i] = True
                    break

    available = [not occ for occ in occupied]
    return available
