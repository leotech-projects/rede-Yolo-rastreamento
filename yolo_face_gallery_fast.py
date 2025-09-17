import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

# Arquivo com embeddings
EMBEDDINGS_FILE = "embeddings.pkl"

# Carregar YOLO
model = YOLO("yolov8n.pt")

# Carregar embeddings da galeria
with open(EMBEDDINGS_FILE, "rb") as f:
    gallery_embeddings = pickle.load(f)

print("Embeddings carregados para:", list(gallery_embeddings.keys()))

# Função para calcular embedding de uma face
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

# Função para comparar embeddings (distância euclidiana)
def compare_embeddings(emb1, emb2, threshold=0.7):
    dist = np.linalg.norm(emb1 - emb2)
    return dist < threshold

# Abrir vídeo (ou 0 para webcam)
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = float(r.conf[0])
        cls = int(r.cls[0])

        if cls == 0 and conf > 0.5:  # apenas pessoas
            face_crop = frame[y1:y2, x1:x2]

            identity = "Desconhecido"
            color = (0, 0, 255)

            emb = get_embedding(face_crop)
            if emb is not None:
                for name, emb_list in gallery_embeddings.items():
                    for ref_emb in emb_list:
                        if compare_embeddings(emb, np.array(ref_emb)):
                            identity = name
                            color = (0, 255, 0)
                            break
                    if identity != "Desconhecido":
                        break

            # Desenhar caixa e nome
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, identity, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO + DeepFace (Rápido com Embeddings)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
