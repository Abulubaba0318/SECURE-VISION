# SecureVision: Prohibited Items Detection System 🛡️

SecureVision is an advanced, AI-powered security monitoring system designed to detect prohibited items in real-time. Leveraging state-of-the-art Deep Learning models (**YOLOv11**), the system identifies potential threats such as **Guns**, **Knives**, and **Smoke/Fire** to enhance public safety.

## 🚀 Features

- **Real-time Detection:** Process live video feed from webcams or security cameras with high accuracy.
- **Threat Alerts:** Instant audible alerts (beep) when a prohibited item is detected.
- **Evidence Capture:** Automatically saves frames containing detected threats for later review.
- **Video Upload:** Support for analyzing pre-recorded video files.
- **Interactive UI:** A modern, web-based dashboard for monitoring and system control.
- **GPU Acceleration:** Optimized to run on CUDA-enabled GPUs for maximum performance.

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Deep Learning:** YOLOv11 (Ultralytics)
- **Computer Vision:** OpenCV
- **Audio Processing:** Pygame
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)

## 📂 Project Structure

```text
FYP/
├── model/                  # Model training and dataset
│   ├── FYP.ipynb           # Training notebook
│   ├── data.yaml           # YOLO dataset configuration
│   └── yolo11m-seg.pt      # Pre-trained/Base model
├── web/                    # Web Application
│   ├── app.py              # Main Flask server
│   ├── best.pt             # Optimized detection model
│   ├── beep.wav            # Alert sound file
│   ├── static/             # Static assets (if any)
│   └── templates/          # HTML Templates (index.html)
├── captures/               # Saved threat detection frames
├── uploads/                # Directory for uploaded videos
```

## ⚙️ Installation & Setup

### 1. Prerequisite
Ensure you have **Python 3.10+** installed. A GPU with CUDA support is recommended for real-time performance.

### 2. Clone the Repository
```bash
git clone <repository-url>
cd FYP
```

### 3. Install Dependencies
```bash
pip install flask opencv-python ultralytics torch pygame
```

### 4. Run the Application
Navigate to the `web` directory and start the Flask server:
```bash
cd web
python app.py
```

### 5. Access the Dashboard
Open your web browser and go to:
`http://127.0.0.1:5000`

## 🖥️ Usage

1. **Live Monitoring:** Click the **"Live Cam"** button to start real-time detection via your webcam.
2. **Video Analysis:** Use the **"Upload Video"** button to process a recorded video file.
3. **Alerts:** If a Weapon or Smoke is detected with confidence > 70%, the system will:
   - Play an alert sound.
   - Highlight the object with a red bounding box.
   - Save the frame in the `web/captures/` folder.

## 🎯 Detection Classes
The model is specifically trained to detect:
- `GUN`
- `KNIFE`
- `SMOKE`

---
**Developed for Final Year Project (FYP)**


