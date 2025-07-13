# Object Tracking with YOLOv8 and OpenCV

This project demonstrates object tracking on video streams using the **YOLOv8** model from the `ultralytics` library combined with **OpenCV**, implemented in Google Colab.

## 🔧 Features

- Object detection using YOLOv8 (`ultralytics`)
- Object tracking using built-in YOLO tracking or ByteTrack
- Simple Re-Identification (ReID) using centroids and bounding box sizes
- Frame-by-frame processing and result video generation
- Google Drive integration for input/output

## 📁 Files

- `Object_trackking.ipynb`: Main notebook for detection and tracking
- `15sec_input_720p.mp4`: Input video stored in Google Drive
- `best.pt`: Custom YOLO model trained for specific object classes
- `output_l.mp4`: Output annotated video saved in Colab environment

## 🚀 Getting Started

### 1. Environment Setup

Ensure you're using Google Colab. Install required dependencies:

```python
!pip install ultralytics
2. Mount Google Drive
python
Copy
Edit
from google.colab import drive
drive.mount('/content/drive')
3. Load the Model and Video
python
Copy
Edit
from ultralytics import YOLO
model = YOLO('/content/drive/MyDrive/best.pt')
4. Run Object Tracking
Two modes:

A. On Individual Frames (Visual Output)
python
Copy
Edit
cap = cv2.VideoCapture('/content/drive/MyDrive/15sec_input_720p.mp4')
while True:
    ret, frame = cap.read()
    if not ret: break
    result = model.track(frame, persist=True)
    annotated_frame = result[0].plot()
    cv2_imshow(annotated_frame)
cap.release()
B. Save to Output Video
python
Copy
Edit
results = model.track(source='/content/drive/MyDrive/15sec_input_720p.mp4', show=True, save=True)
🔍 Re-Identification Logic (Optional)
A simple ReID module is implemented based on:

Bounding box centroid

Bounding box width and height

Euclidean distance for re-association

📦 Output
Annotated video with object IDs saved as output_l.mp4

Optional metadata can be stored as JSON

📚 Requirements
Python 3.7+

OpenCV

ultralytics >= 8.0

Google Colab

📌 Notes
ByteTrack tracker config (bytetrack.yaml) can be passed to improve long-term tracking

Limit frame processing using max_frames if working with long videos

📸 Sample Output
Output image screenshot  with bounding boxes and tracked object IDs overlayed frame.
![image](Football_analysis-
/Screenshot 2025-07-07 153833.png)

