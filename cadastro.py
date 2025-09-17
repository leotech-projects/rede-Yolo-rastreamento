import cv2
import os

# Pasta onde as fotos serão salvas
GALLERY_PATH = "gallery/"

# Criar pasta se não existir
if not os.path.exists(GALLERY_PATH):
    os.makedirs(GALLERY_PATH)

# Nome da pessoa
name = input("Digite o nome da pessoa: ").strip().lower()

# Quantidade de fotos a capturar
num_photos = 5
count = 0

# Abrir webcam
cap = cv2.VideoCapture(0)

print("Pressione ESPAÇO para capturar cada foto, ESC para cancelar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break

    # Mostrar a câmera
    cv2.imshow("Cadastro de Pessoa", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC → cancelar
        print("Cadastro cancelado.")
        break
    elif key == 32:  # ESPAÇO → captura uma foto
        count += 1
        file_path = os.path.join(GALLERY_PATH, f"{name}_{count}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"[{count}/{num_photos}] Foto salva em {file_path}")

        if count >= num_photos:
            print("Cadastro concluído com sucesso!")
            break

cap.release()
cv2.destroyAllWindows()
