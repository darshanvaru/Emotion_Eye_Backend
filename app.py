# app.py
from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load the YOLO model once
model = YOLO("best.pt")

# Create a directory for saving images if it doesn't exist
os.makedirs("Emotion_Log", exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    
    # Save the file securely
    filename = file.filename
    filepath = os.path.join("Emotion_Log", filename)
    file.save(filepath)

    # Perform prediction
    results = model(filepath)
    result = results[0]

    # Analyze results
    if result.boxes and len(result.boxes.cls) > 0:
        cls_id = int(result.boxes.cls[0])
        label = model.names[cls_id]
        return jsonify({"emotion": label})
    else:
        return jsonify({"emotion": "none"})

if __name__ == "__main__":
    # Run on all network interfaces
    app.run(host="0.0.0.0", port=8000, debug=True)
