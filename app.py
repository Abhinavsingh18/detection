from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import whisper
import cv2
import numpy as np
from fer import FER
import tempfile
import os
import subprocess

app = Flask(__name__)
CORS(app)

# Load models once
whisper_model = whisper.load_model("base")
emotion_detector = FER(mtcnn=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    top_emotion = emotion_detector.top_emotion(img)
    if top_emotion:
        emotion, score = top_emotion
        return jsonify({'emotion': emotion, 'score': score})
    else:
        return jsonify({'emotion': None, 'score': None})

@app.route('/detect_language', methods=['POST'])
def detect_language():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio uploaded'}), 400
    file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_in:
        file.save(tmp_in.name)
        tmp_in_path = tmp_in.name

    # Convert to wav using ffmpeg
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp_out_path = tmp_out.name
    tmp_out.close()
    subprocess.run([
        'ffmpeg', '-y', '-i', tmp_in_path, '-ar', '16000', '-ac', '1', tmp_out_path
    ], check=True)

    result = whisper_model.transcribe(tmp_out_path)
    os.unlink(tmp_in_path)
    os.unlink(tmp_out_path)
    return jsonify({'language': result['language'], 'transcript': result['text']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
