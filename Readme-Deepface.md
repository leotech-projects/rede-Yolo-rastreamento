Perfeito 🚀
Comparar **a cada frame** com todas as fotos é lento. O jeito certo é **pré-calcular embeddings** das imagens da galeria, e depois só comparar com os embeddings das pessoas detectadas no vídeo. Isso reduz o custo e acelera bastante o reconhecimento.

Vou separar em **duas etapas**:

---

## 📂 Estrutura Final

```
yolo-deepface-multi/
│── cadastro.py                  # Cadastro de várias fotos por pessoa
│── build_embeddings.py          # Pré-calcula os embeddings das fotos
│── yolo_face_gallery_fast.py    # Rastreamento otimizado com embeddings
│── embeddings.pkl               # Banco de embeddings gerado
│── gallery/                     
│    ├── joao_1.jpg
│    ├── joao_2.jpg
│    ├── maria_1.jpg
│    └── pedro_1.jpg
│── video.mp4
│── README.md
```

---

## 📝 Código: **build\_embeddings.py**

```python
import os
import pickle
from deepface import DeepFace

GALLERY_PATH = "gallery/"
EMBEDDINGS_FILE = "embeddings.pkl"

# Dicionário para guardar os embeddings
embeddings = {}

print("Gerando embeddings das imagens da galeria...")

for f in os.listdir(GALLERY_PATH):
    if f.lower().endswith((".jpg", ".png", ".jpeg")):
        name = f.split("_")[0]
        file_path = os.path.join(GALLERY_PATH, f)

        try:
            embedding = DeepFace.represent(
                img_path=file_path,
                model_name="ArcFace",
                enforce_detection=False
            )[0]["embedding"]

            if name not in embeddings:
                embeddings[name] = []
            embeddings[name].append(embedding)

            print(f"Embedding gerado para {f}")

        except Exception as e:
            print(f"Erro ao processar {f}: {e}")

# Salvar em arquivo
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(embeddings, f)

print(f"Embeddings salvos em {EMBEDDINGS_FILE}")
```

---

## 📝 Código: **yolo\_face\_gallery\_fast.py**

```python
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
```

---

## ▶️ Como Usar

1. **Cadastrar fotos** com `cadastro.py`.
2. **Gerar embeddings** uma única vez:

```bash
python build_embeddings.py
```

Isso cria `embeddings.pkl`.
3\. **Rodar rastreamento otimizado**:

```bash
python yolo_face_gallery_fast.py
```

---

✅ Agora o reconhecimento fica muito mais rápido, pois cada face detectada é comparada só no **espaço vetorial** e não mais com imagens inteiras.

---

Quer que eu adicione também um **limiar de confiança por pessoa** (ex.: “se a média de distâncias para as fotos de João for menor que X, classifica como João”), para aumentar a precisão?
