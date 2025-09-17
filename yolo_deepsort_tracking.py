import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Carregar modelo YOLO pré-treinado (pode trocar por 'yolov8n.pt' para ser mais leve)
model = YOLO("yolov8s.pt")

# Inicializar o DeepSORT
tracker = DeepSort(max_age=30)

# Abrir o vídeo (ou 0 para webcam)
cap = cv2.VideoCapture("video.mp4")  # substitua por "0" para webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Executa detecção YOLO
    results = model(frame)

    detections = []
    for r in results[0].boxes:
        x1, y1, x2, y2 = r.xyxy[0]
        conf = float(r.conf[0])
        cls = int(r.cls[0])

        # Filtrar apenas pessoas (classe 0 no COCO)
        if cls == 0 and conf > 0.5:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # Passa as detecções para o DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    # Desenha os rastros na tela
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO + DeepSORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
