"""
RMU Face Recognition Attendance System
Backend: Flask + OpenCV LBPH Face Recognizer
"""

import os
import cv2
import json
import base64
import numpy as np
from datetime import datetime, date
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(__file__)
FACES_DIR  = os.path.join(BASE, "data", "faces")
MODELS_DIR = os.path.join(BASE, "data", "models")
LOGS_DIR   = os.path.join(BASE, "data", "logs")
STUDENTS_F = os.path.join(BASE, "data", "students.json")
MODEL_FILE = os.path.join(MODELS_DIR, "recognizer.yml")
LABELS_FILE= os.path.join(MODELS_DIR, "labels.pkl")

for d in [FACES_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_students():
    if os.path.exists(STUDENTS_F):
        with open(STUDENTS_F) as f:
            return json.load(f)
    return {}

def save_students(data):
    with open(STUDENTS_F, "w") as f:
        json.dump(data, f, indent=2)

def load_log(log_date: str):
    path = os.path.join(LOGS_DIR, f"{log_date}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_log(log_date: str, data: dict):
    path = os.path.join(LOGS_DIR, f"{log_date}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def decode_image(b64_string: str) -> np.ndarray:
    """Decode base64 image → BGR numpy array."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def detect_face(img_bgr: np.ndarray):
    """Returns (face_gray, x, y, w, h) or None."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])  # largest face
    face_gray = gray[y:y+h, x:x+w]
    face_gray = cv2.resize(face_gray, (200, 200))
    return face_gray, x, y, w, h

def build_recognizer():
    """Train LBPH recognizer from stored face images."""
    students = load_students()
    if not students:
        return None, {}

    label_map = {}   # numeric_id → student_id
    faces, labels = [], []

    for sid, info in students.items():
        face_folder = os.path.join(FACES_DIR, sid)
        if not os.path.isdir(face_folder):
            continue
        imgs = [f for f in os.listdir(face_folder) if f.endswith(".jpg")]
        if not imgs:
            continue
        numeric_id = info.get("label_id")
        label_map[numeric_id] = sid
        for img_name in imgs:
            img = cv2.imread(os.path.join(face_folder, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                labels.append(numeric_id)

    if not faces:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)
    return recognizer, label_map


def load_recognizer():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        return None, {}
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(MODEL_FILE)
    with open(LABELS_FILE, "rb") as f:
        label_map = pickle.load(f)
    return rec, label_map


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/students", methods=["GET"])
def get_students():
    return jsonify(load_students())


@app.route("/api/students", methods=["POST"])
def add_student():
    body = request.json
    sid       = body.get("student_id", "").strip().upper()
    name      = body.get("name", "").strip()
    dept      = body.get("department", "").strip()
    programme = body.get("programme", "").strip()
    year      = body.get("year", "").strip()

    if not sid or not name:
        return jsonify({"error": "student_id and name are required"}), 400

    students = load_students()
    if sid in students:
        return jsonify({"error": "Student ID already exists"}), 409

    # Assign a unique numeric label
    existing_labels = {v.get("label_id", 0) for v in students.values()}
    label_id = max(existing_labels, default=0) + 1

    students[sid] = {
        "name": name,
        "department": dept,
        "programme": programme,
        "year": year,
        "label_id": label_id,
        "registered_at": datetime.now().isoformat(),
        "face_samples": 0,
    }
    save_students(students)
    os.makedirs(os.path.join(FACES_DIR, sid), exist_ok=True)
    return jsonify({"success": True, "student": students[sid]})


@app.route("/api/students/<sid>", methods=["DELETE"])
def delete_student(sid):
    students = load_students()
    if sid not in students:
        return jsonify({"error": "Not found"}), 404
    del students[sid]
    save_students(students)
    import shutil
    face_dir = os.path.join(FACES_DIR, sid)
    if os.path.isdir(face_dir):
        shutil.rmtree(face_dir)
    # Retrain
    build_recognizer()
    return jsonify({"success": True})


@app.route("/api/enroll", methods=["POST"])
def enroll_face():
    """Save face sample for a student."""
    body   = request.json
    sid    = body.get("student_id", "").strip().upper()
    image  = body.get("image")

    if not sid or not image:
        return jsonify({"error": "student_id and image required"}), 400

    students = load_students()
    if sid not in students:
        return jsonify({"error": "Student not found"}), 404

    img = decode_image(image)
    result = detect_face(img)
    if result is None:
        return jsonify({"error": "No face detected. Ensure good lighting and face the camera."}), 422

    face_gray = result[0]
    face_folder = os.path.join(FACES_DIR, sid)
    os.makedirs(face_folder, exist_ok=True)
    count = len([f for f in os.listdir(face_folder) if f.endswith(".jpg")])
    cv2.imwrite(os.path.join(face_folder, f"{count+1:04d}.jpg"), face_gray)

    students[sid]["face_samples"] = count + 1
    save_students(students)

    # Retrain after every 5 samples (or at 1st)
    if (count + 1) % 5 == 0 or count == 0:
        build_recognizer()

    return jsonify({"success": True, "samples": count + 1})


@app.route("/api/train", methods=["POST"])
def train_model():
    rec, label_map = build_recognizer()
    if rec is None:
        return jsonify({"error": "No training data available"}), 400
    return jsonify({"success": True, "students_trained": len(label_map)})


@app.route("/api/recognize", methods=["POST"])
def recognize():
    """Recognize a face and optionally mark attendance."""
    body      = request.json
    image     = body.get("image")
    course    = body.get("course", "General")
    mark_att  = body.get("mark_attendance", False)
    log_date  = body.get("date", str(date.today()))

    if not image:
        return jsonify({"error": "image required"}), 400

    rec, label_map = load_recognizer()
    if rec is None:
        return jsonify({"error": "Model not trained yet. Please enroll students first."}), 422

    img = decode_image(image)
    result = detect_face(img)
    if result is None:
        return jsonify({"recognized": False, "error": "No face detected"}), 200

    face_gray, x, y, w, h = result
    label_id, confidence = rec.predict(face_gray)

    # LBPH: lower confidence = better match. Threshold ~80 works well.
    THRESHOLD = 80
    if confidence > THRESHOLD:
        return jsonify({
            "recognized": False,
            "confidence": round(float(confidence), 2),
            "message": "Face not recognized",
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        })

    students  = load_students()
    sid       = label_map.get(label_id)
    student   = students.get(sid, {})

    response = {
        "recognized": True,
        "student_id": sid,
        "name": student.get("name"),
        "department": student.get("department"),
        "programme": student.get("programme"),
        "year": student.get("year"),
        "confidence": round(float(confidence), 2),
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
    }

    if mark_att:
        log = load_log(log_date)
        already_marked = sid in log.get(course, {})
        if not already_marked:
            if course not in log:
                log[course] = {}
            log[course][sid] = {
                "name": student.get("name"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "confidence": round(float(confidence), 2),
            }
            save_log(log_date, log)
        response["attendance_marked"] = not already_marked
        response["already_marked"]    = already_marked

    return jsonify(response)


@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    log_date = request.args.get("date", str(date.today()))
    course   = request.args.get("course")
    log      = load_log(log_date)
    if course:
        return jsonify({course: log.get(course, {})})
    return jsonify(log)


@app.route("/api/attendance/summary", methods=["GET"])
def attendance_summary():
    """List all log dates and counts."""
    files = [f.replace(".json", "") for f in os.listdir(LOGS_DIR) if f.endswith(".json")]
    files.sort(reverse=True)
    summary = []
    for d in files:
        log = load_log(d)
        total = sum(len(v) for v in log.values())
        summary.append({"date": d, "total": total, "courses": list(log.keys())})
    return jsonify(summary)


@app.route("/api/attendance/export", methods=["GET"])
def export_csv():
    """Export attendance as CSV text."""
    log_date = request.args.get("date", str(date.today()))
    log = load_log(log_date)
    rows = ["Date,Course,Student ID,Name,Time,Confidence"]
    for course, entries in log.items():
        for sid, info in entries.items():
            rows.append(f"{log_date},{course},{sid},{info['name']},{info['time']},{info['confidence']}")
    from flask import Response
    return Response("\n".join(rows), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment;filename=attendance_{log_date}.csv"})


if __name__ == "__main__":
    print("🎓 RMU Attendance System starting on http://localhost:5000")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)