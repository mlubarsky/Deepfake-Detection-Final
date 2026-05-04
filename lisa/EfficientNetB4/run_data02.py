"""
Deepfake Detection on data02 Videos
Trains a quick version of ReDeepFake on data01, then runs inference on data02 videos.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA01_DIR  = os.path.join(BASE_DIR, 'data01')
DATA02_DIR  = os.path.join(BASE_DIR, 'data02')
MODEL_PATH  = os.path.join(BASE_DIR, 'redeepfake_model.h5')
FACES_DIR   = os.path.join(DATA01_DIR, 'faces_224')
META_CSV    = os.path.join(DATA01_DIR, 'metadata.csv')

VIDEO_DIR    = os.path.join(DATA02_DIR, 'video')
DEEPFAKE_DIR = os.path.join(DATA02_DIR, 'deepfake')

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

SAMPLE_PER_CLASS = 2000   # images per class for quick training
EPOCHS           = 5      # enough to get meaningful weights fast


# ── Build model (same architecture as notebook) ────────────────────────────────
def build_model():
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    base_model = EfficientNetB4(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False  # freeze for quick training

    inputs  = tf.keras.layers.Input(shape=(224, 224, 3))
    x       = data_augmentation(inputs)
    x       = base_model(x, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model   = tf.keras.Model(inputs, outputs)

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        metrics=["accuracy"]
    )
    return model


# ── Train on data01 ────────────────────────────────────────────────────────────
def train_model():
    print("="*60)
    print("  STEP 1: Training model on data01")
    print("="*60)

    meta = pd.read_csv(META_CSV)
    real_df = meta[meta['label'] == 'REAL'].sample(SAMPLE_PER_CLASS, random_state=42)
    fake_df = meta[meta['label'] == 'FAKE'].sample(SAMPLE_PER_CLASS, random_state=42)
    sample  = pd.concat([real_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Using {SAMPLE_PER_CLASS} REAL + {SAMPLE_PER_CLASS} FAKE images ({EPOCHS} epochs)")

    preprocess = tf.keras.applications.efficientnet.preprocess_input

    def load_image(row):
        path = os.path.join(FACES_DIR, row['videoname'][:-4] + '.jpg')
        img  = cv2.imread(path)
        if img is None:
            return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        return img, 1 if row['label'] == 'FAKE' else 0

    print("  Loading images...")
    images, labels = [], []
    for _, row in sample.iterrows():
        img, lbl = load_image(row)
        if img is not None:
            images.append(img)
            labels.append(lbl)

    X = np.array(images)
    y = np.array(labels)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .map(lambda x, y: (preprocess(x), y))
                .shuffle(500).batch(16).prefetch(1))
    val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                .map(lambda x, y: (preprocess(x), y))
                .batch(16))

    model = build_model()
    print(f"  Training for {EPOCHS} epochs...\n")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    model.save(MODEL_PATH)
    print(f"\n  Model saved to {MODEL_PATH}")
    return model


# ── Load or train ──────────────────────────────────────────────────────────────
def get_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading saved model from {MODEL_PATH} ...")
        return tf.keras.models.load_model(MODEL_PATH)
    return train_model()


# ── Face extraction from video ─────────────────────────────────────────────────
def extract_faces_from_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0, 0, 0

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    dur    = total / fps if fps > 0 else 0
    idxs   = np.linspace(0, max(total - 1, 0), min(max_frames, total), dtype=int)

    faces = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets  = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, w, h) in dets:
            pad = int(0.1 * max(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            crop = cv2.resize(frame[y1:y2, x1:x2], (224, 224))
            faces.append((int(idx), crop))
    cap.release()
    return faces, total, fps, dur


# ── Predict ────────────────────────────────────────────────────────────────────
def predict_faces(model, faces):
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    results = []
    for frame_idx, face in faces:
        img  = preprocess(face.astype(np.float32))
        prob = float(model.predict(np.expand_dims(img, 0), verbose=0)[0][0])
        results.append((frame_idx, prob, "FAKE" if prob > 0.5 else "REAL"))
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    model = get_model()
    all_results = []

    for label_name, folder, ground_truth in [
        ("REAL videos",     VIDEO_DIR,    "REAL"),
        ("DEEPFAKE videos", DEEPFAKE_DIR, "FAKE"),
    ]:
        print(f"\n{'='*60}")
        print(f"  {label_name}  (ground truth: {ground_truth})")
        print(f"{'='*60}")

        video_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
        ])

        for vf in video_files:
            vpath = os.path.join(folder, vf)
            print(f"\n[{vf}]")
            faces, total, fps, dur = extract_faces_from_video(vpath)
            print(f"  Duration: {dur:.1f}s | Frames: {total} | FPS: {fps:.1f}")
            print(f"  Faces detected: {len(faces)}")

            if not faces:
                print("  No faces detected — skipping.")
                continue

            preds     = predict_faces(model, faces)
            probs     = [r[1] for r in preds]
            avg_prob  = np.mean(probs)
            verdict   = "FAKE" if avg_prob > 0.5 else "REAL"
            correct   = verdict == ground_truth

            print(f"  Avg fake prob : {avg_prob:.3f}")
            print(f"  Fake votes    : {sum(1 for r in preds if r[2]=='FAKE')}/{len(preds)}")
            print(f"  Real votes    : {sum(1 for r in preds if r[2]=='REAL')}/{len(preds)}")
            print(f"  Verdict       : {verdict}  (ground truth: {ground_truth})  {'✓ CORRECT' if correct else '✗ WRONG'}")

            all_results.append({
                'video': vf, 'ground_truth': ground_truth,
                'verdict': verdict, 'avg_fake_prob': avg_prob, 'correct': correct
            })

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Video':<25} {'GT':<8} {'Predicted':<10} {'Avg P(fake)':<14} Result")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*14} ------")
    for r in all_results:
        tick = "✓" if r['correct'] else "✗"
        print(f"  {r['video']:<25} {r['ground_truth']:<8} {r['verdict']:<10} {r['avg_fake_prob']:<14.3f} {tick}")

    if all_results:
        acc = sum(r['correct'] for r in all_results) / len(all_results)
        print(f"\n  Accuracy on data02: {acc:.1%}  ({sum(r['correct'] for r in all_results)}/{len(all_results)} correct)")


if __name__ == '__main__':
    main()
