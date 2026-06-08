# Object Counter Web App

This small Flask app lets you upload an image and runs the existing YOLOv8 detector to annotate and count detected objects.

Quick start:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the web app:

```bash
python app.py
```

3. Open http://127.0.0.1:5000/ in your browser and upload an image.

Notes:
- The YOLO model file `yolov8n.pt` must be present in the project root (already included).
- Uploaded and annotated images are stored under `static/uploads`.
