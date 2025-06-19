from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import glob

app = Flask(__name__)

# Load model with a reliable relative path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

# Ensure image log folder exists
os.makedirs("Emotion_Log", exist_ok=True)

# Optional: clean up files to avoid storage issues
def cleanup_images():
    files = glob.glob("Emotion_Log/*")
    for f in files:
        os.remove(f)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    filename = file.filename
    filepath = os.path.join("Emotion_Log", filename)
    file.save(filepath)

    cleanup_images()

    # Predict with lighter config
    results = model(filepath, imgsz=320, conf=0.25)
    result = results[0]

    if result.boxes and len(result.boxes.cls) > 0:
        cls_id = int(result.boxes.cls[0])
        label = model.names[cls_id]
        return jsonify({"emotion": label})
    else:
        return jsonify({"emotion": "none"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
