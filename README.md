# RMU Face Recognition Attendance System

A full-stack web application for automated attendance using OpenCV LBPH face recognition.

## Tech Stack
- **Backend**: Python · Flask · OpenCV (LBPH Face Recognizer)
- **Frontend**: HTML / CSS / Vanilla JS · No frameworks needed
- **Storage**: JSON files (no database required)

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note**: Make sure `opencv-contrib-python` is installed (not just `opencv-python`),
> as the LBPH face recognizer lives in the `cv2.face` module.

### 2. Run the server

```bash
python app.py
```

The server starts at **http://localhost:5000**

### 3. Open the web app

Open your browser to **http://localhost:5000** — the frontend is served directly by Flask.

---

## Usage Workflow

### Step 1 — Enroll Students
1. Go to the **Enroll Student** tab
2. Fill in Student ID, Name, Department, Year → click **Register Student**
3. Click **Start Camera** and **Capture** 20 face photos (vary angles slightly)
4. Click **Train / Update Model** to finalize

### Step 2 — Take Attendance
1. Go to the **Take Attendance** tab
2. Enter the course name and date
3. Click **Start Camera** → **Scan Face**
4. Use **Auto-Scan** mode for continuous recognition every 3 seconds
5. Attendance is automatically logged with timestamp

### Step 3 — View & Export Records
1. Go to the **Records** tab
2. Pick a date and optionally filter by course
3. Click **Export CSV** to download the attendance sheet

---

## Project Structure

```
rmu_attendance/
├── app.py              # Flask backend + all API routes
├── index.html          # Frontend (single page app)
├── requirements.txt
└── data/
    ├── students.json   # Student registry
    ├── faces/          # Stored face samples (per student)
    │   └── RMU001/
    │       ├── 0001.jpg
    │       └── ...
    ├── models/
    │   ├── recognizer.yml   # Trained LBPH model
    │   └── labels.pkl       # Label → student_id mapping
    └── logs/
        └── 2024-11-15.json  # Attendance logs by date
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/students` | List all students |
| POST | `/api/students` | Register a new student |
| DELETE | `/api/students/<id>` | Remove a student |
| POST | `/api/enroll` | Upload a face sample |
| POST | `/api/train` | Retrain the model |
| POST | `/api/recognize` | Recognize a face + mark attendance |
| GET  | `/api/attendance` | Get attendance log for a date |
| GET  | `/api/attendance/summary` | List all session dates |
| GET  | `/api/attendance/export` | Download CSV |

---

## Tips for Best Accuracy

- Capture **at least 15–20 samples** per student
- Ensure **good, even lighting** during enrollment and scanning
- The LBPH recognizer threshold is set to **80** (lower = stricter)
- Retrain the model after enrolling new students
- For large cohorts (100+ students), consider upgrading to `face_recognition` (dlib-based)