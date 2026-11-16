# 😷 Mask Detection using CNN  Real-time face mask detection built with Python, OpenCV, and a CNN. Detects faces from a live webcam feed and classifies them as Mask or No Mask. Includes a simple CRUD interface for data management and WEKA-based preprocessing.  **Run:** `mask.py]`  **Author:** Simran Vaswani
## screenshot
<img width="1920" height="1080" alt="login Page" src="https://github.com/user-attachments/assets/b3eecf20-4d21-4d77-a52a-358fb2768f1d" />
<img width="1920" height="1051" alt="loading page" src="https://github.com/user-attachments/assets/96b0656b-d8f2-47e1-9f09-da1b6f8fbe9c" />
<img width="1916" height="1080" alt="detection page" src="https://github.com/user-attachments/assets/c235339f-87eb-46b3-a7b8-31ac1c2ff2c8" />
<img width="1920" height="1060" alt="detection1 page" src="https://github.com/user-attachments/assets/9be001a7-7e47-4539-9098-7fa7ff58a852" />

import argparse
import json
import os
import sqlite3
import sys
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_IMPORT_ERROR = None
except ModuleNotFoundError as e:
    keras = None
    layers = None
    TF_IMPORT_ERROR = e

IMG_SIZE = 128
ID_IMG_SIZE = 64
ID_TOP_RATIO = 0.6
NO_MASK_THRESHOLD = 0.50
PREDICTION_SMOOTHING_WINDOW = 5
CALIBRATION_MAX_SAMPLES_PER_CLASS = 80
FACE_SWITCH_RESET_DISTANCE = 100
MASK_REGION_TOP_RATIO = 0.35
LABEL_HYSTERESIS_MARGIN = 0.03
NO_MASK_STRICT_MARGIN = 0.05
MASK_STATE_THRESHOLD_BOOST = 0.07
NO_MASK_STATE_THRESHOLD_DROP = 0.07
SKIN_BLEND_WEIGHT = 0.35
MIN_FACE_AREA_RATIO = 0.08
CENTER_TOLERANCE_X = 0.22
CENTER_TOLERANCE_Y = 0.25
LOW_LIGHT_MEAN_THRESHOLD = 60.0
# Set to True to convert a mirrored webcam feed to normal orientation.
UNMIRROR_CAMERA_FEED = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PERSON_DATASET_DIR = os.path.join(DATASET_DIR, "persons")
CLASS_NAMES = ["mask", "no_mask"]
MODEL_PATH = os.path.join(BASE_DIR, "mask_detector_cnn.keras")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
USERS_DB_PATH = os.path.join(BASE_DIR, "users.db")

def ensure_dirs():
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(DATASET_DIR, class_name), exist_ok=True)
    os.makedirs(PERSON_DATASET_DIR, exist_ok=True)
    os.makedirs(PROFILES_DIR, exist_ok=True)

def get_profile_path(name):
    safe_name = name.strip().replace(" ", "_")
    if not safe_name:
        return None
    return os.path.join(PROFILES_DIR, f"{safe_name}.json")

def load_user_profile(name):
    path = get_profile_path(name)
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None

def save_user_profile(name, threshold, invert, mask_mean, no_mask_mean):
    path = get_profile_path(name)
    if path is None:
        return
    os.makedirs(PROFILES_DIR, exist_ok=True)
    payload = {
        "threshold": float(threshold),
        "invert_labels": bool(invert),
        "mask_mean": float(mask_mean),
        "no_mask_mean": float(no_mask_mean),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def init_users_db():
    conn = sqlite3.connect(USERS_DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

def save_user_name(name):
    clean_name = name.strip()
    if not clean_name:
        return
    conn = sqlite3.connect(USERS_DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_entries (name) VALUES (?)",
            (clean_name,),
        )
        conn.commit()
    finally:
        conn.close()

def get_recent_user_names(limit=20):
    if not os.path.exists(USERS_DB_PATH):
        return []
    conn = sqlite3.connect(USERS_DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name FROM (
                SELECT name, MAX(created_at) AS last_seen
                FROM user_entries
                GROUP BY name
                ORDER BY last_seen DESC
                LIMIT ?
            )
            """,
            (int(limit),),
        )
        return [str(row[0]) for row in cursor.fetchall()]
    finally:
        conn.close()

def create_face_detectors():
    detector_paths = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
    ]
    detectors = []
    for path in detector_paths:
        if os.path.exists(path):
            det = cv2.CascadeClassifier(path)
            if not det.empty():
                detectors.append(det)
    if not detectors:
        raise RuntimeError("No Haar face cascade could be loaded from OpenCV data path.")
    return detectors

def detect_faces_robust(gray, detectors):
    def compute_iou(box_a, box_b):
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area) / float(union)

    all_faces = []
    # Try multiple parameter sets because mask occlusion can reduce detector confidence.
    param_sets = [(1.1, 4), (1.15, 4), (1.2, 5)]
    for detector in detectors:
        for scale_factor, min_neighbors in param_sets:
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(80, 80),
            )
            if len(faces) > 0:
                all_faces.extend(faces.tolist())
    if not all_faces:
        return []

    all_faces = [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h in all_faces
    ]
    all_faces.sort(key=lambda f: f[2] * f[3], reverse=True)

    deduped = []
    for x, y, w, h in all_faces:
        is_duplicate = False
        for dx, dy, dw, dh in deduped:
            iou = compute_iou((x, y, w, h), (dx, dy, dw, dh))
            cdist = ((x + (w // 2) - (dx + (dw // 2))) ** 2 + (y + (h // 2) - (dy + (dh // 2))) ** 2) ** 0.5
            min_diag = max(1.0, min((w * w + h * h) ** 0.5, (dw * dw + dh * dh) ** 0.5))
            size_ratio = min(float(w * h), float(dw * dh)) / max(1.0, max(float(w * h), float(dw * dh)))

            # Merge near-identical detections produced by multiple cascades/params.
            if iou >= 0.30 or (cdist <= 0.18 * min_diag and size_ratio >= 0.60):
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append((int(x), int(y), int(w), int(h)))
    return deduped

def normalize_camera_frame(frame):
    if UNMIRROR_CAMERA_FEED:
        return cv2.flip(frame, 1)
    return frame

def open_camera(camera_index=0):
    video = cv2.VideoCapture(camera_index)
    if not video.isOpened():
        return video
    # Request clearer feed: HD + MJPG + low buffering.
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video.set(cv2.CAP_PROP_FPS, 30)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if hasattr(cv2, "CAP_PROP_FOURCC"):
        video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return video

def get_face_feature(face_bgr):
    # Use only upper-face region (eyes/forehead) so mask coverage affects less.
    top_h = max(1, int(face_bgr.shape[0] * ID_TOP_RATIO))
    upper_face = face_bgr[:top_h, :, :]
    gray = cv2.cvtColor(upper_face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (ID_IMG_SIZE, ID_IMG_SIZE))
    gray = cv2.equalizeHist(gray)
    feature = gray.astype(np.float32).reshape(-1) / 255.0
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm
    return feature

def collect_images(label, num_samples=200):
    if label not in CLASS_NAMES:
        raise ValueError(f"Label must be one of: {CLASS_NAMES}")

    ensure_dirs()
    save_dir = os.path.join(DATASET_DIR, label)
    detectors = create_face_detectors()
    video = open_camera(0)
    if not video.isOpened():
        raise RuntimeError("Could not open webcam (camera index 0).")

    count = 0
    frame_idx = 0
    print(f"Collecting {num_samples} images for class: {label}")
    print("Press q to stop early.")
    cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Data Collection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    failed_reads = 0
    while True:
        ret, frame = video.read()
        if not ret:
            failed_reads += 1
            if failed_reads >= 30:
                raise RuntimeError("Unable to read frames from webcam.")
            continue
        failed_reads = 0
        frame = normalize_camera_frame(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_robust(gray, detectors)

        for (x, y, w, h) in faces:
            face = frame[y : y + h, x : x + w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            if frame_idx % 3 == 0 and count < num_samples:
                file_path = os.path.join(save_dir, f"{label}_{count:04d}.jpg")
                cv2.imwrite(file_path, face)
                count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 255), 2)
            cv2.putText(
                frame,
                f"{label}: {count}/{num_samples}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 220, 255),
                2,
            )
            break

        frame_idx += 1
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or count >= num_samples:
            break

    video.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images to {save_dir}")

def collect_person_images(name, num_samples=100):
    ensure_dirs()
    safe_name = name.strip().replace(" ", "_")
    if not safe_name:
        raise ValueError("Name cannot be empty.")

    save_dir = os.path.join(PERSON_DATASET_DIR, safe_name)
    os.makedirs(save_dir, exist_ok=True)

    detectors = create_face_detectors()
    video = open_camera(0)
    if not video.isOpened():
        raise RuntimeError("Could not open webcam (camera index 0).")

    count = 0
    frame_idx = 0
    print(f"Collecting {num_samples} face images for person: {safe_name}")
    print("Press q to stop early.")
    cv2.namedWindow("Person Data Collection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Person Data Collection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    failed_reads = 0
    while True:
        ret, frame = video.read()
        if not ret:
            failed_reads += 1
            if failed_reads >= 30:
                raise RuntimeError("Unable to read frames from webcam.")
            continue
        failed_reads = 0
        frame = normalize_camera_frame(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_robust(gray, detectors)

        for (x, y, w, h) in faces:
            face = frame[y : y + h, x : x + w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            if frame_idx % 3 == 0 and count < num_samples:
                file_path = os.path.join(save_dir, f"{safe_name}_{count:04d}.jpg")
                cv2.imwrite(file_path, face)
                count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            cv2.putText(
                frame,
                f"{safe_name}: {count}/{num_samples}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 200, 0),
                2,
            )
            break

        frame_idx += 1
        cv2.imshow("Person Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or count >= num_samples:
            break

    video.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images to {save_dir}")

def load_identity_database():
    if not os.path.isdir(PERSON_DATASET_DIR):
        return None, None

    person_to_features = {}
    for person_name in os.listdir(PERSON_DATASET_DIR):
        person_dir = os.path.join(PERSON_DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        person_to_features.setdefault(person_name, [])
        for file_name in os.listdir(person_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(person_dir, file_name)
            img = cv2.imread(path)
            if img is None:
                continue
            person_to_features[person_name].append(get_face_feature(img))

    names = []
    features = []
    for person_name, feats in person_to_features.items():
        if not feats:
            continue
        mean_feat = np.mean(np.array(feats, dtype=np.float32), axis=0)
        norm = np.linalg.norm(mean_feat)
        if norm > 0:
            mean_feat = mean_feat / norm
        names.append(person_name)
        features.append(mean_feat)

    if not names:
        return None, None
    return names, np.array(features, dtype=np.float32)

def predict_identity(face_bgr, names, features, threshold=0.85, unknown_name="Unknown"):
    if names is None or features is None:
        return unknown_name
    query = get_face_feature(face_bgr)
    dists = np.linalg.norm(features - query, axis=1)
    best_idx = int(np.argmin(dists))
    if float(dists[best_idx]) > threshold:
        return unknown_name
    return names[best_idx]

def save_person_face(name, face_bgr):
    safe_name = name.strip().replace(" ", "_")
    if not safe_name or safe_name.lower() == "unknown":
        return
    person_dir = os.path.join(PERSON_DATASET_DIR, safe_name)
    os.makedirs(person_dir, exist_ok=True)
    existing = [
        f
        for f in os.listdir(person_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    file_path = os.path.join(person_dir, f"{safe_name}_{len(existing):04d}.jpg")
    cv2.imwrite(file_path, cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE)))

def match_live_track(cx, cy, tracks, max_dist=90):
    best_idx = -1
    best_dist = 1e9
    for i, track in enumerate(tracks):
        tx, ty = track["center"]
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        if dist < max_dist and dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx

def load_dataset():
    images = []
    labels = []

    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(class_dir, file_name)
            img = cv2.imread(path)
            if img is None:
                continue
            images.append(preprocess_face_for_model(img))
            labels.append(idx)

    if not images:
        raise RuntimeError(
            "No training images found. Collect data first in dataset/mask and dataset/no_mask."
        )

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y

def build_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_model(epochs=10, batch_size=32):
    X, y = load_dataset()
    model = build_model()
    class_counts = np.bincount(y, minlength=len(CLASS_NAMES))
    total = float(np.sum(class_counts))
    class_weight = {
        idx: (total / (len(CLASS_NAMES) * max(1.0, float(count))))
        for idx, count in enumerate(class_counts)
    }
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=True,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def preprocess_face_for_model(face_bgr):
    h = face_bgr.shape[0]
    top = int(h * MASK_REGION_TOP_RATIO)
    top = min(max(0, top), h - 1)
    mask_region = face_bgr[top:, :, :]
    resized = cv2.resize(mask_region, (IMG_SIZE, IMG_SIZE))
    return resized.astype(np.float32) / 255.0

def augment_face_for_inference(face_bgr):
    base = (preprocess_face_for_model(face_bgr) * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    brighter = cv2.convertScaleAbs(base, alpha=1.08, beta=10)
    return [base, equalized, brighter]

def predict_no_mask_probability(model, face_bgr):
    variants = augment_face_for_inference(face_bgr)
    batch = np.array([preprocess_face_for_model(v) for v in variants], dtype=np.float32)
    preds = model.predict(batch, verbose=0).reshape(-1)
    return float(np.mean(preds))

def estimate_no_mask_skin_score(face_bgr):
    h = face_bgr.shape[0]
    top = int(h * MASK_REGION_TOP_RATIO)
    top = min(max(0, top), h - 1)
    lower = face_bgr[top:, :, :]
    lower = cv2.resize(lower, (IMG_SIZE, IMG_SIZE))

    ycrcb = cv2.cvtColor(lower, cv2.COLOR_BGR2YCrCb)
    hls = cv2.cvtColor(lower, cv2.COLOR_BGR2HLS)
    skin_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin_hls = cv2.inRange(hls, (0, 40, 0), (25, 255, 255))
    skin_mask = cv2.bitwise_and(skin_ycrcb, skin_hls)
    skin_ratio = float(np.count_nonzero(skin_mask)) / float(skin_mask.size)
    # Map rough skin ratio to [0,1] no-mask score.
    return float(np.clip((skin_ratio - 0.08) / 0.22, 0.0, 1.0))

def calibrate_mask_classifier(model, max_samples_per_class=CALIBRATION_MAX_SAMPLES_PER_CLASS):
    class_scores = {}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_images = []
        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(class_dir, file_name))
            if img is None:
                continue
            class_images.append(preprocess_face_for_model(img))
            if len(class_images) >= max_samples_per_class:
                break

        if class_images:
            batch = np.array(class_images, dtype=np.float32)
            preds = model.predict(batch, verbose=0).reshape(-1)
            class_scores[class_name] = preds.astype(np.float32)

    if "mask" not in class_scores or "no_mask" not in class_scores:
        return NO_MASK_THRESHOLD, False, 0.0, 0.35, 0.65

    mask_scores = class_scores["mask"]
    no_mask_scores = class_scores["no_mask"]
    best_threshold = NO_MASK_THRESHOLD
    best_invert = False
    best_bal_acc = -1.0

    best_mask_eval = None
    best_no_mask_eval = None
    for invert_candidate in (False, True):
        if invert_candidate:
            mask_eval = 1.0 - mask_scores
            no_mask_eval = 1.0 - no_mask_scores
        else:
            mask_eval = mask_scores
            no_mask_eval = no_mask_scores

        for threshold in np.linspace(0.25, 0.75, 101):
            tpr = float(np.mean(no_mask_eval >= threshold))
            tnr = float(np.mean(mask_eval < threshold))
            bal_acc = 0.5 * (tpr + tnr)
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_threshold = float(threshold)
                best_invert = invert_candidate
                best_mask_eval = mask_eval
                best_no_mask_eval = no_mask_eval

    mean_mask = float(np.mean(best_mask_eval)) if best_mask_eval is not None else 0.35
    mean_no_mask = float(np.mean(best_no_mask_eval)) if best_no_mask_eval is not None else 0.65
    return best_threshold, best_invert, best_bal_acc, mean_mask, mean_no_mask

def ask_name_frontend():
    entered_name = {"value": None}
    init_users_db()

    def start_detection():
        name = name_var.get().strip()
        if not name:
            messagebox.showerror("Name Required", "Please enter your name.")
            return
        save_user_name(name)
        entered_name["value"] = name
        root.destroy()

    def close_app():
        root.destroy()

    def on_enter_primary(_event):
        primary_btn.configure(bg="#1d4ed8")

    def on_leave_primary(_event):
        primary_btn.configure(bg="#2563eb")

    def on_enter_secondary(_event):
        secondary_btn.configure(bg="#cbd5e1")

    def on_leave_secondary(_event):
        secondary_btn.configure(bg="#e2e8f0")

    def apply_known_name(_event=None):
        if known_list.curselection():
            selected = known_list.get(known_list.curselection()[0])
            if selected != "No known users":
                name_var.set(selected.replace("_", " "))
                entry.focus_set()

    def refresh_time():
        import datetime
        t = datetime.datetime.now().strftime("%A, %d %b %Y  |  %I:%M:%S %p")
        clock_label.configure(text=t)
        root.after(1000, refresh_time)

    known_users = []
    if os.path.isdir(PERSON_DATASET_DIR):
        known_users.extend(
            [
                n
                for n in os.listdir(PERSON_DATASET_DIR)
                if os.path.isdir(os.path.join(PERSON_DATASET_DIR, n))
            ]
        )
    known_users.extend(get_recent_user_names(limit=20))
    # Keep order while removing duplicates (case-insensitive).
    deduped_users = []
    seen = set()
    for n in known_users:
        key = n.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped_users.append(n)
    known_users = deduped_users[:20]

    root = tk.Tk()
    root.title("Mask Detection Suite")
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda _event: root.destroy())
    root.configure(bg="#0b1220")

    header = tk.Frame(root, bg="#111827", height=82)
    header.pack(fill="x", side="top")
    header.pack_propagate(False)

    tk.Label(
        header,
        text="MASK DETECTION SUITE",
        font=("Segoe UI Semibold", 16),
        bg="#111827",
        fg="#e5e7eb",
    ).pack(side="left", padx=28, pady=20)

    tk.Label(
        header,
        text="Secure Screening Console",
        font=("Segoe UI", 11),
        bg="#111827",
        fg="#9ca3af",
    ).pack(side="right", padx=28, pady=24)

    clock_label = tk.Label(
        root,
        text="",
        font=("Segoe UI", 10),
        bg="#0b1220",
        fg="#94a3b8",
    )
    clock_label.place(relx=0.5, rely=0.11, anchor="center")
    refresh_time()

    wrapper = tk.Frame(root, bg="#0b1220")
    wrapper.place(relx=0.5, rely=0.54, anchor="center")

    info_card = tk.Frame(
        wrapper,
        bg="#1f2937",
        bd=0,
        highlightthickness=1,
        highlightbackground="#334155",
        padx=32,
        pady=34,
    )
    info_card.pack(side="left", padx=(0, 18), fill="y")

    tk.Label(
        info_card,
        text="Session Overview",
        font=("Segoe UI Semibold", 22),
        bg="#1f2937",
        fg="#f8fafc",
    ).pack(anchor="w")

    tk.Label(
        info_card,
        text="Use this console to run live mask compliance checks.",
        font=("Segoe UI", 11),
        bg="#1f2937",
        fg="#cbd5e1",
        wraplength=360,
        justify="left",
    ).pack(anchor="w", pady=(8, 18))

    for item in [
        "Single-face detection with confidence badge",
        "Per-user calibration profile loading",
        "On-screen controls for threshold and mapping",
    ]:
        tk.Label(
            info_card,
            text=f"- {item}",
            font=("Segoe UI", 11),
            bg="#1f2937",
            fg="#e2e8f0",
            justify="left",
        ).pack(anchor="w", pady=3)

    tk.Label(
        info_card,
        text="Tip: Press Esc anytime to close the app.",
        font=("Segoe UI", 10),
        bg="#1f2937",
        fg="#93c5fd",
    ).pack(anchor="w", pady=(22, 0))

    card = tk.Frame(
        wrapper,
        bg="#f8fafc",
        bd=0,
        highlightthickness=2,
        highlightbackground="#cbd5e1",
        padx=34,
        pady=30,
    )
    card.pack(side="left")

    content = tk.Frame(card, bg="#f8fafc")
    content.pack(fill="both", expand=True)

    left_col = tk.Frame(content, bg="#f8fafc")
    left_col.pack(side="left", fill="both", expand=True, padx=(0, 22))

    right_col = tk.Frame(
        content,
        bg="#eef2ff",
        bd=0,
        highlightthickness=1,
        highlightbackground="#c7d2fe",
        padx=14,
        pady=12,
    )
    right_col.pack(side="left", fill="y")

    tk.Label(
        left_col,
        text="Operator Login",
        font=("Segoe UI Semibold", 22),
        bg="#f8fafc",
        fg="#0f172a",
    ).pack(anchor="w")

    tk.Label(
        left_col,
        text="Enter your name to begin live face mask detection.",
        font=("Segoe UI", 12),
        bg="#f8fafc",
        fg="#475569",
    ).pack(anchor="w", pady=(6, 18))

    tk.Label(
        left_col,
        text="Full Name",
        font=("Segoe UI Semibold", 11),
        bg="#f8fafc",
        fg="#334155",
    ).pack(anchor="w", pady=(0, 8))

    name_var = tk.StringVar()
    entry = tk.Entry(
        left_col,
        textvariable=name_var,
        font=("Segoe UI", 15),
        width=30,
        relief=tk.FLAT,
        bd=0,
        highlightthickness=2,
        highlightbackground="#cbd5e1",
        highlightcolor="#2563eb",
    )
    entry.pack(pady=(0, 16), ipady=10, fill="x")
    entry.focus_set()
    entry.bind("<Return>", lambda _event: start_detection())

    actions = tk.Frame(left_col, bg="#f8fafc")
    actions.pack(fill="x")

    primary_btn = tk.Button(
        actions,
        text="Open Face Detection",
        font=("Segoe UI Semibold", 12),
        bg="#2563eb",
        fg="white",
        activebackground="#1d4ed8",
        activeforeground="white",
        relief=tk.FLAT,
        padx=18,
        pady=11,
        command=start_detection,
        cursor="hand2",
    )
    primary_btn.pack(side="left")
    primary_btn.bind("<Enter>", on_enter_primary)
    primary_btn.bind("<Leave>", on_leave_primary)

    secondary_btn = tk.Button(
        actions,
        text="Exit",
        font=("Segoe UI Semibold", 12),
        bg="#e2e8f0",
        fg="#1e293b",
        activebackground="#cbd5e1",
        activeforeground="#0f172a",
        relief=tk.FLAT,
        padx=18,
        pady=11,
        command=close_app,
        cursor="hand2",
    )
    secondary_btn.pack(side="left", padx=(12, 0))
    secondary_btn.bind("<Enter>", on_enter_secondary)
    secondary_btn.bind("<Leave>", on_leave_secondary)

    tk.Label(
        right_col,
        text="Known / Used Users",
        font=("Segoe UI Semibold", 11),
        bg="#eef2ff",
        fg="#1e1b4b",
    ).pack(anchor="w", pady=(0, 8))

    known_list = tk.Listbox(
        right_col,
        width=22,
        height=9,
        font=("Segoe UI", 10),
        bd=0,
        highlightthickness=0,
        activestyle="none",
        bg="#ffffff",
        fg="#1f2937",
        selectbackground="#2563eb",
        selectforeground="white",
    )
    known_list.pack(fill="both", expand=True)
    if known_users:
        for user in known_users:
            known_list.insert(tk.END, user)
    else:
        known_list.insert(tk.END, "No known users")
    known_list.bind("<Double-Button-1>", apply_known_name)

    tk.Label(
        right_col,
        text="Double-click to autofill",
        font=("Segoe UI", 9),
        bg="#eef2ff",
        fg="#475569",
    ).pack(anchor="w", pady=(8, 0))

    tk.Label(
        root,
        text="Press Enter to continue | Esc to close",
        font=("Segoe UI", 11),
        bg="#0b1220",
        fg="#94a3b8",
    ).place(relx=0.5, rely=0.97, anchor="center")

    root.mainloop()
    return entered_name["value"]

def show_startup_status(window_name, lines, duration_ms=1800):
    panel = np.zeros((360, 820, 3), dtype=np.uint8)
    panel[:] = (17, 24, 39)
    cv2.rectangle(panel, (20, 20), (800, 340), (71, 85, 105), 2)
    cv2.putText(
        panel,
        "Initializing Detection Pipeline",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (226, 232, 240),
        2,
    )
    y = 135
    for line in lines:
        cv2.putText(
            panel,
            f"[OK] {line}",
            (50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (74, 222, 128),
            2,
        )
        y += 50
    cv2.putText(
        panel,
        "Preparing live camera stream...",
        (50, 305),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (148, 163, 184),
        1,
    )
    cv2.imshow(window_name, panel)
    cv2.waitKey(duration_ms)

def get_control_buttons():
    return [
        {"id": "invert", "label": "Toggle Mapping", "x": 14, "y": 14, "w": 152, "h": 30},
        {"id": "thr_down", "label": "Threshold -", "x": 14, "y": 50, "w": 152, "h": 30},
        {"id": "thr_up", "label": "Threshold +", "x": 14, "y": 86, "w": 152, "h": 30},
        {"id": "sample_mask", "label": "Sample Mask", "x": 14, "y": 122, "w": 152, "h": 30},
        {"id": "sample_no_mask", "label": "Sample No-Mask", "x": 14, "y": 158, "w": 152, "h": 30},
    ]

def draw_control_buttons(frame, buttons):
    for button in buttons:
        x, y, w, h = button["x"], button["y"], button["w"], button["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 41, 59), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (148, 163, 184), 1)
        cv2.putText(
            frame,
            button["label"],
            (x + 8, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (226, 232, 240),
            1,
        )

def handle_button_action(
    action_id,
    invert_labels_live,
    threshold_live,
    last_raw_prob_no_mask,
    live_mask_samples_raw,
    live_no_mask_samples_raw,
    apply_live_calibration,
):
    if action_id == "invert":
        invert_labels_live = not invert_labels_live
        print(f"invert_labels={'yes' if invert_labels_live else 'no'}")
    elif action_id == "thr_down":
        threshold_live = max(0.30, threshold_live - 0.02)
        print(f"threshold={threshold_live:.2f}")
    elif action_id == "thr_up":
        threshold_live = min(0.70, threshold_live + 0.02)
        print(f"threshold={threshold_live:.2f}")
    elif action_id == "sample_mask" and last_raw_prob_no_mask is not None:
        live_mask_samples_raw.append(last_raw_prob_no_mask)
        apply_live_calibration()
        print(
            f"mask_samples={len(live_mask_samples_raw)}, no_mask_samples={len(live_no_mask_samples_raw)}, "
            f"threshold={threshold_live:.2f}, invert_labels={'yes' if invert_labels_live else 'no'}"
        )
    elif action_id == "sample_no_mask" and last_raw_prob_no_mask is not None:
        live_no_mask_samples_raw.append(last_raw_prob_no_mask)
        apply_live_calibration()
        print(
            f"mask_samples={len(live_mask_samples_raw)}, no_mask_samples={len(live_no_mask_samples_raw)}, "
            f"threshold={threshold_live:.2f}, invert_labels={'yes' if invert_labels_live else 'no'}"
        )
    return invert_labels_live, threshold_live

def detect_mask(forced_name=None):
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Train the model first, then run detect again:")
        print(f"  python {os.path.basename(__file__)} train --epochs 10 --batch-size 32")
        return

    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e
    detectors = create_face_detectors()
    (
        calibrated_threshold,
        invert_labels,
        calibration_bal_acc,
        mask_mean_score,
        no_mask_mean_score,
    ) = calibrate_mask_classifier(model)
    person_names, person_features = load_identity_database()
    if person_names is None:
        print("No person dataset found. Person name will show as Unknown.")
        print("Collect person data with:")
        print(f"  python {os.path.basename(__file__)} collect-person --name YourName --samples 100")
    video = open_camera(0)
    if not video.isOpened():
        raise RuntimeError("Could not open webcam (camera index 0).")
    cv2.namedWindow("Mask Detection (CNN)", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Mask Detection (CNN)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )
    show_startup_status(
        "Mask Detection (CNN)",
        ["Model Loaded", "Camera Connected", "Calibration Prepared"],
    )

    invert_labels_live = invert_labels
    threshold_live = calibrated_threshold
    mask_mean_live = mask_mean_score
    no_mask_mean_live = no_mask_mean_score
    profile_dirty = False
    live_mask_samples_raw = []
    live_no_mask_samples_raw = []
    last_raw_prob_no_mask = None
    if forced_name:
        profile = load_user_profile(forced_name)
        if profile is not None:
            threshold_live = float(profile.get("threshold", threshold_live))
            invert_labels_live = bool(profile.get("invert_labels", invert_labels_live))
            mask_mean_live = float(profile.get("mask_mean", mask_mean_live))
            no_mask_mean_live = float(profile.get("no_mask_mean", no_mask_mean_live))
            print(f"Loaded profile for {forced_name}")

    def apply_live_calibration():
        nonlocal invert_labels_live, threshold_live, mask_mean_live, no_mask_mean_live, profile_dirty
        if len(live_mask_samples_raw) < 3 or len(live_no_mask_samples_raw) < 3:
            return
        mean_mask_raw = float(np.mean(live_mask_samples_raw))
        mean_no_mask_raw = float(np.mean(live_no_mask_samples_raw))
        invert_labels_live = mean_no_mask_raw < mean_mask_raw

        if invert_labels_live:
            mask_scores = [1.0 - s for s in live_mask_samples_raw]
            no_mask_scores = [1.0 - s for s in live_no_mask_samples_raw]
        else:
            mask_scores = live_mask_samples_raw
            no_mask_scores = live_no_mask_samples_raw

        mask_mean_live = float(np.mean(mask_scores))
        no_mask_mean_live = float(np.mean(no_mask_scores))
        threshold_live = float(
            np.clip((mask_mean_live + no_mask_mean_live) / 2.0, 0.25, 0.80)
        )
        profile_dirty = True

    print("Starting live detection. Press q to exit.")
    print(
        f"Calibration: threshold={calibrated_threshold:.2f}, "
        f"invert_labels={'yes' if invert_labels else 'no'}, "
        f"balanced_acc={calibration_bal_acc:.2f}, "
        f"mask_mean={mask_mean_live:.2f}, no_mask_mean={no_mask_mean_live:.2f}"
    )
    print("Controls: i=toggle label mapping, [=lower threshold, ]=raise threshold")
    print("Live calibration: m=sample MASK frame, n=sample NO-MASK frame")

    failed_reads = 0
    prob_history = []
    last_face_center = None
    last_mask_label = None
    no_mask_transition_streak = 0
    buttons = get_control_buttons()
    click_state = {"point": None}

    def on_mouse(event, x, y, _flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param["point"] = (x, y)

    cv2.setMouseCallback("Mask Detection (CNN)", on_mouse, click_state)

    while True:
        ret, frame = video.read()
        if not ret:
            failed_reads += 1
            if failed_reads >= 30:
                raise RuntimeError("Unable to read frames from webcam.")
            continue
        failed_reads = 0
        frame = normalize_camera_frame(frame)
        guidance_message = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_robust(gray, detectors)
        if not faces:
            eq_gray = cv2.equalizeHist(gray)
            faces = detect_faces_robust(eq_gray, detectors)
        if not faces:
            prob_history.clear()
            last_face_center = None
            last_mask_label = None
            no_mask_transition_streak = 0
            guidance_message = "No face detected. Look at camera."
            if float(np.mean(gray)) < LOW_LIGHT_MEAN_THRESHOLD:
                guidance_message = "Low light. Improve lighting."

        if len(faces) > 1:
            prob_history.clear()
            last_face_center = None
            last_mask_label = None
            no_mask_transition_streak = 0
            cv2.putText(
                frame,
                "Only one face allowed",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Mask Detection (CNN)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("i"):
                key_action = "invert"
            elif key == ord("["):
                key_action = "thr_down"
            elif key == ord("]"):
                key_action = "thr_up"
            elif key == ord("m"):
                key_action = "sample_mask"
            elif key == ord("n"):
                key_action = "sample_no_mask"
            else:
                key_action = None
            if key_action is not None:
                invert_labels_live, threshold_live = handle_button_action(
                    key_action,
                    invert_labels_live,
                    threshold_live,
                    last_raw_prob_no_mask,
                    live_mask_samples_raw,
                    live_no_mask_samples_raw,
                    apply_live_calibration,
                )
                profile_dirty = True
            continue

        for (x, y, w, h) in faces[:1]:
            cx = x + (w // 2)
            cy = y + (h // 2)
            if last_face_center is not None:
                dx = cx - last_face_center[0]
                dy = cy - last_face_center[1]
                if ((dx * dx) + (dy * dy)) ** 0.5 > FACE_SWITCH_RESET_DISTANCE:
                    prob_history.clear()
                    last_mask_label = None
                    no_mask_transition_streak = 0
            last_face_center = (cx, cy)

            face_raw = frame[y : y + h, x : x + w]
            person_name = (
                forced_name
                if forced_name
                else predict_identity(face_raw, person_names, person_features)
            )
            prob_no_mask = predict_no_mask_probability(model, face_raw)
            last_raw_prob_no_mask = prob_no_mask
            if invert_labels_live:
                prob_no_mask = 1.0 - prob_no_mask
            prob_history.append(prob_no_mask)
            if len(prob_history) > PREDICTION_SMOOTHING_WINDOW:
                prob_history.pop(0)
            smooth_prob_no_mask = float(np.mean(prob_history))
            skin_no_mask_score = estimate_no_mask_skin_score(face_raw)
            # Keep model score as baseline; use skin evidence only to boost no-mask confidence.
            blended_no_mask_score = (
                (1.0 - SKIN_BLEND_WEIGHT) * smooth_prob_no_mask
                + SKIN_BLEND_WEIGHT * skin_no_mask_score
            )
            combined_no_mask_score = max(smooth_prob_no_mask, blended_no_mask_score)
            mask_score = 1.0 - combined_no_mask_score

            decision_thr = min(0.95, threshold_live + NO_MASK_STRICT_MARGIN)
            # Adaptive thresholding by current state:
            # - while mask is detected, raise threshold (harder to switch to no-mask)
            # - while no-mask is detected, lower threshold (easier to stay no-mask)
            mask_state_thr = min(0.95, decision_thr + MASK_STATE_THRESHOLD_BOOST)
            no_mask_state_thr = max(0.05, decision_thr - NO_MASK_STATE_THRESHOLD_DROP)
            upper_thr = min(0.95, mask_state_thr + LABEL_HYSTERESIS_MARGIN)
            lower_thr = max(0.05, no_mask_state_thr - LABEL_HYSTERESIS_MARGIN)

            if last_mask_label == "No Mask Detected":
                no_mask_transition_streak = 0
                is_no_mask = combined_no_mask_score >= lower_thr
            elif last_mask_label == "Mask Detected":
                if prob_no_mask >= decision_thr:
                    no_mask_transition_streak += 1
                else:
                    no_mask_transition_streak = 0
                fast_no_mask_switch = no_mask_transition_streak >= 2
                is_no_mask = fast_no_mask_switch or (combined_no_mask_score >= upper_thr)
            else:
                no_mask_transition_streak = 0
                is_no_mask = combined_no_mask_score >= decision_thr

            if is_no_mask:
                label = "No Mask Detected"
                color = (0, 0, 255)
                conf = combined_no_mask_score
            else:
                label = "Mask Detected"
                color = (0, 255, 0)
                conf = 1.0 - combined_no_mask_score
            display_score = mask_score
            if is_no_mask:
                no_mask_transition_streak = 0
            last_mask_label = label

            frame_h, frame_w = frame.shape[:2]
            face_area_ratio = float(w * h) / float(frame_w * frame_h)
            face_cx = x + (w // 2)
            face_cy = y + (h // 2)
            frame_cx = frame_w // 2
            frame_cy = frame_h // 2
            if face_area_ratio < MIN_FACE_AREA_RATIO:
                guidance_message = "Move closer to the camera."
            elif abs(face_cx - frame_cx) > int(frame_w * CENTER_TOLERANCE_X) or abs(face_cy - frame_cy) > int(frame_h * CENTER_TOLERANCE_Y):
                guidance_message = "Center your face in the frame."
            elif float(np.mean(gray)) < LOW_LIGHT_MEAN_THRESHOLD:
                guidance_message = "Improve lighting for better detection."

            if conf >= 0.80:
                confidence_level = "High"
                badge_color = (22, 163, 74)
            elif conf >= 0.65:
                confidence_level = "Medium"
                badge_color = (245, 158, 11)
            else:
                confidence_level = "Low"
                badge_color = (239, 68, 68)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{person_name} | {label} {display_score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            cv2.rectangle(frame, (x, y + h + 8), (x + 200, y + h + 36), badge_color, -1)
            cv2.putText(
                frame,
                f"Confidence: {confidence_level}",
                (x + 8, y + h + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"mask_score={mask_score:.2f} model={smooth_prob_no_mask:.2f} thr={threshold_live:.2f}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"m(mask):{len(live_mask_samples_raw)} n(no-mask):{len(live_no_mask_samples_raw)}",
                (20, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        draw_control_buttons(frame, buttons)
        cv2.putText(
            frame,
            f"invert={'yes' if invert_labels_live else 'no'}",
            (18, 206),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (226, 232, 240),
            1,
        )
        cv2.putText(
            frame,
            f"threshold={threshold_live:.2f}",
            (18, 224),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (226, 232, 240),
            1,
        )
        if guidance_message:
            box_w = 520
            box_h = 36
            bx = max(10, (frame.shape[1] - box_w) // 2)
            by = 12
            cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (30, 41, 59), -1)
            cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (245, 158, 11), 1)
            cv2.putText(
                frame,
                guidance_message,
                (bx + 12, by + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (251, 191, 36),
                2,
            )
        if click_state["point"] is not None:
            px, py = click_state["point"]
            click_state["point"] = None
            for button in buttons:
                x, y, w, h = button["x"], button["y"], button["w"], button["h"]
                if x <= px <= x + w and y <= py <= y + h:
                    invert_labels_live, threshold_live = handle_button_action(
                        button["id"],
                        invert_labels_live,
                        threshold_live,
                        last_raw_prob_no_mask,
                        live_mask_samples_raw,
                        live_no_mask_samples_raw,
                        apply_live_calibration,
                    )
                    profile_dirty = True
                    break

        cv2.imshow("Mask Detection (CNN)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("i"):
            key_action = "invert"
        elif key == ord("["):
            key_action = "thr_down"
        elif key == ord("]"):
            key_action = "thr_up"
        elif key == ord("m"):
            key_action = "sample_mask"
        elif key == ord("n"):
            key_action = "sample_no_mask"
        else:
            key_action = None
        if key_action is not None:
            invert_labels_live, threshold_live = handle_button_action(
                key_action,
                invert_labels_live,
                threshold_live,
                last_raw_prob_no_mask,
                live_mask_samples_raw,
                live_no_mask_samples_raw,
                apply_live_calibration,
            )
            profile_dirty = True

    if forced_name and profile_dirty:
        save_user_profile(
            forced_name,
            threshold_live,
            invert_labels_live,
            mask_mean_live,
            no_mask_mean_live,
        )
        print(f"Saved profile for {forced_name}")

    video.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Face mask detection using CNN")
    parser.add_argument(
        "mode",
        choices=["collect", "collect-person", "train", "detect", "app"],
        nargs="?",
        default="app",
        help="collect data, train CNN, run live detection, or launch frontend app",
    )
    parser.add_argument(
        "--label",
        choices=CLASS_NAMES,
        help="class label for collect mode",
    )
    parser.add_argument("--name", type=str, help="person name for collect-person mode")
    parser.add_argument("--samples", type=int, default=200, help="images to collect")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="training batch size")

    args = parser.parse_args()

    if args.mode in ("train", "detect", "app") and TF_IMPORT_ERROR is not None:
        print("TensorFlow is not installed for this Python interpreter.")
        print(f"Current interpreter: {sys.executable}")
        print(f"Script location: {BASE_DIR}")
        print("Use the project virtual environment:")
        print(r"  .\.venv\Scripts\Activate.ps1")
        print("Then run:")
        print(f"  python {os.path.basename(__file__)} {args.mode}")
        return

    if args.mode == "collect":
        if not args.label:
            raise ValueError("collect mode requires --label mask or --label no_mask")
        try:
            collect_images(args.label, args.samples)
        except RuntimeError as e:
            print(str(e))
    elif args.mode == "collect-person":
        person_name = args.name if args.name else input("Enter person name: ").strip()
        try:
            collect_person_images(person_name, args.samples)
        except (RuntimeError, ValueError) as e:
            print(str(e))
    elif args.mode == "train":
        try:
            train_model(args.epochs, args.batch_size)
        except RuntimeError as e:
            print(str(e))
            print("Collect data first:")
            print(f"  python {os.path.basename(__file__)} collect --label mask --samples 200")
            print(f"  python {os.path.basename(__file__)} collect --label no_mask --samples 200")
    elif args.mode in ("detect", "app"):
        person_name = ask_name_frontend()
        if not person_name:
            print("No name entered. Exiting.")
            return
        try:
            detect_mask(forced_name=person_name)
        except RuntimeError as e:
            print(str(e))

if __name__ == "__main__":
    main()


