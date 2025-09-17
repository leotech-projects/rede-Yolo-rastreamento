import cv2
from ultralytics import YOLO
from deepface import DeepFace

# Foto da pessoa conhecida (já cadastrada)
known_image_path = "pessoa_referencia.jpg"

# Carregar modelo YOLO (detecção de pessoas)
model = YOLO("yolov8n.pt")

# Abrir vídeo (substitua por "0" para webcam)
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecta pessoas no frame
    results = model(frame)
    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = float(r.conf[0])
        cls = int(r.cls[0])

        # Filtrar apenas pessoas (classe 0 no COCO)
        if cls == 0 and conf > 0.5:
            face_crop = frame[y1:y2, x1:x2]

            try:
                # Comparar com a foto da pessoa conhecida
                result = DeepFace.verify(
                    face_crop,
                    known_image_path,
                    model_name="Facenet",   # Pode trocar por VGG-Face, ArcFace etc.
                    enforce_detection=False
                )

                if result["verified"]:
                    label = "Pessoa conhecida"
                    color = (0, 255, 0)  # Verde
                else:
                    label = "Desconhecido"
                    color = (0, 0, 255)  # Vermelho

            except Exception as e:
                label = "Erro"
                color = (255, 0, 0)

            # Desenhar caixa e texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO + DeepFace Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
