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
