import cv2
import pickle
import numpy as np
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace
from deep_sort_realtime.deepsort_tracker import DeepSort

# Arquivos
EMBEDDINGS_FILE = "embeddings.pkl"
LOG_FILE = "log_identificacoes.csv"

# YOLO
model = YOLO("yolov8n.pt")

# DeepSORT
tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3)

# Carregar embeddings
with open(EMBEDDINGS_FILE, "rb") as f:
    gallery_embeddings = pickle.load(f)

print("Embeddings carregados para:", list(gallery_embeddings.keys()))

# Criar log se não existir
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "track_id", "identity", "distance"])

# Identidades já atribuídas a track_ids
track_identities = {}

# Funções
def get_embedding(face_img):
    try:
        rep = DeepFace.represent(
            img_path=face_img,
            model_name="ArcFace",
            enforce_detection=False
        )[0]["embedding"]
        return np.array(rep)
    except:
        return None

def identify_face(emb, gallery, threshold=0.7):
    best_name = "Desconhecido"
    best_dist = float("inf")
    for name, emb_list in gallery.items():
        dists = [np.linalg.norm(emb - np.array(ref)) for ref in emb_list]
        mean_dist = np.mean(dists)
        if mean_dist < best_dist:
            best_name = name
            best_dist = mean_dist
    return (best_name, best_dist) if best_dist < threshold else ("Desconhecido", best_dist)

def salvar_log(track_id, identity, dist):
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), track_id, identity, round(dist, 4)])

# Vídeo (0 para webcam)
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detecta
    results = model(frame, verbose=False)
    detections = []

    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = float(r.conf[0])
        cls = int(r.cls[0])
        if cls == 0 and conf > 0.5:  # apenas pessoas
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # DeepSORT rastreia
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Se já identificado antes
        if track_id in track_identities:
            identity, dist = track_identities[track_id]
        else:
            # Recorta rosto e identifica
            face_crop = frame[y1:y2, x1:x2]
            emb = get_embedding(face_crop)
            if emb is not None:
                identity, dist = identify_face(emb, gallery_embeddings, threshold=0.7)
                track_identities[track_id] = (identity, dist)
                salvar_log(track_id, identity, dist)
            else:
                identity, dist = "Desconhecido", -1

        # Cor e texto
        color = (0, 255, 0) if identity != "Desconhecido" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}: {identity}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO + DeepSORT + DeepFace", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
