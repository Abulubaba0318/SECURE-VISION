from flask import Flask, Response, render_template, request, redirect
import cv2
import time
import pygame
import os
from ultralytics import YOLO
import torch
from datetime import datetime

# Flask  
app = Flask(__name__)

# Pygame   
pygame.mixer.init()

# GPU/CPU  
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt")
print("Model class names:", model.names)

# Custom class names used in training
classNames = ["KNIFE", "SMOKE", "GUN"]
threatening_objects = ["KNIFE", "SMOKE", "GUN"]

cap = None
camera_running = False  

last_alert_time = 0
alert_cooldown = 3  

@app.route('/')
def index():
    return render_template('index.html')

def detect_threats_from_frame(img):
    global last_alert_time
    img = cv2.resize(img, (640, 640))
    results = model(img)
    threatening_object_detected = False
    detected_objects = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            if cls >= len(classNames):
                continue

            class_name = classNames[cls]
            confidence = round(float(box.conf[0]) * 100, 2)

            if confidence < 70:
                continue

            if class_name in threatening_objects:
                threatening_object_detected = True
                detected_objects.append(class_name)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, f"{class_name} ({confidence}%)", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    current_time = time.time()
    if threatening_object_detected and (current_time - last_alert_time > alert_cooldown):
        last_alert_time = current_time
        try:
            pygame.mixer.music.load("beep.wav")
            pygame.mixer.music.play()
        except pygame.error:
            print("Alert sound file 'beep.wav' not found!")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join("captures", f"threat_capture_{timestamp}.jpg")
        cv2.imwrite(image_path, img)
        print(f"Threat detected: {', '.join(detected_objects)} | Frame saved at {image_path}")

    return img

def generate_frames():
    global cap, camera_running
    if not camera_running:
        cap = cv2.VideoCapture(0)
        camera_running = True

    while camera_running:
        success, img = cap.read()
        if not success:
            break

        img = detect_threats_from_frame(img)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if cap is not None:
        cap.release()
        camera_running = False

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file:
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)

        # Get FPS of uploaded video
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_time = int(1000 / fps) if fps > 0 else 33  # Default 30 FPS ~33ms

        frame_count = 0
        skip_frames = 0  

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Frame skipping logic
            if skip_frames == 0 or frame_count % (skip_frames + 1) == 0:
                processed_frame = detect_threats_from_frame(frame)
            else:
                processed_frame = frame

            cv2.imshow("Processed Video", processed_frame)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    return redirect('/')

@app.route('/stop_camera')
def stop_camera():
    global cap, camera_running
    if cap is not None:
        cap.release()
        cap = None
    camera_running = False
    return "Camera stopped"

@app.teardown_appcontext
def cleanup(e=None):
    global cap, camera_running
    if cap is not None:
        cap.release()
        cap = None
    camera_running = False

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    if not os.path.exists("captures"):
        os.makedirs("captures")
    app.run(debug=True)
