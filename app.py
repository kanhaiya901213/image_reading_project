from flask import Flask, render_template, request, redirect, url_for
import os
from time import perf_counter
from werkzeug.utils import secure_filename
from collections import defaultdict
import cv2
from ultralytics import YOLO

# Load YOLO model once at startup
model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'avif'}
UPLOAD_HISTORY_LIMIT = 5
upload_history = []


def detect_image(image_path, save_path=None, people_only=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(image)

    counts = defaultdict(int)
    annotated = image.copy()
    names = {}
    try:
        names = model.names
    except Exception:
        try:
            names = results[0].names
        except Exception:
            names = {}

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if people_only and cls != 0:
                continue
            name = names.get(cls, str(cls))
            counts[name] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated,
                        f"{counts[name]} {name}",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

    total = sum(counts.values())
    cv2.putText(annotated,
                f"Total: {total}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        try:
            annotated_resized = cv2.resize(annotated, (900, 600))
        except Exception:
            annotated_resized = annotated
        cv2.imwrite(save_path, annotated_resized)

    return dict(counts), total, annotated

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        supported_formats=sorted(ALLOWED_EXT),
        recent_uploads=list(reversed(upload_history)),
    )


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        out_name = 'annotated_' + filename
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        people_only = request.form.get('people_only') == 'on'
        start_time = perf_counter()
        counts, total, _ = detect_image(path, save_path=out_path, people_only=people_only)
        elapsed_ms = int((perf_counter() - start_time) * 1000)
        image_url = '/' + out_path.replace('\\', '/')

        upload_history.append({
            'filename': filename,
            'image_url': image_url,
            'total': total,
        })
        if len(upload_history) > UPLOAD_HISTORY_LIMIT:
            upload_history.pop(0)

        return render_template(
            'result.html',
            image_url=image_url,
            counts=counts,
            total=total,
            filename=filename,
            people_only=people_only,
            elapsed_ms=elapsed_ms,
        )
    return redirect(url_for('index'))



app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 5000))
)
