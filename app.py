from flask import Flask, render_template, request, redirect, url_for
import os
from time import perf_counter
from werkzeug.utils import secure_filename

from main import detect_image

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'avif'}
UPLOAD_HISTORY_LIMIT = 5
upload_history = []

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
