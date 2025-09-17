A **YOLO** (You Only Look Once) é uma família de redes neurais voltadas para **detecção de objetos em tempo real**. Apesar de ser conhecida principalmente pela detecção, também pode ser adaptada para **rastreamento de objetos** (object tracking), combinando técnicas adicionais.

### 1. Detecção com YOLO

* A YOLO divide a imagem em uma grade e, de forma **única e direta**, prevê:

  * **Classes dos objetos** (ex.: pessoa, carro, cachorro);
  * **Caixas delimitadoras (bounding boxes)**;
  * **Confiança (score)** de cada detecção.
* É extremamente rápida, o que a torna ideal para aplicações em **tempo real**, como câmeras de segurança, drones e veículos autônomos.

### 2. YOLO em Rastreamento

YOLO, por si só, não faz rastreamento. Ele apenas detecta quadro a quadro.
Para transformá-la em um sistema de **tracking**, ela costuma ser combinada com algoritmos de associação de identidade, como:

* **SORT (Simple Online and Realtime Tracking)**: usa detecção do YOLO + filtro de Kalman + associação via Hungarian algorithm.
* **DeepSORT**: versão aprimorada, que além das posições usa **descritores visuais (features)** para manter a identidade do objeto mesmo em casos de oclusão ou cruzamento.

Fluxo simplificado:

1. YOLO detecta objetos no frame atual.
2. O rastreador (ex.: DeepSORT) recebe as caixas detectadas.
3. O algoritmo associa as detecções às identidades já existentes.
4. Cada objeto mantém um **ID único** ao longo dos frames, caracterizando o rastreamento.

### 3. Vantagens

* **Alta velocidade**: YOLO é rápido o suficiente para rodar em tempo real.
* **Boa precisão**: especialmente nas versões mais recentes (YOLOv7, YOLOv8, YOLO-NAS, YOLO-World).
* **Escalabilidade**: pode rodar em câmeras de vigilância, veículos autônomos, drones, etc.

### 4. Exemplos de uso prático

* Monitoramento de pessoas em um ambiente (segurança institucional).
* Rastreamento de veículos em rodovias.
* Sistemas de análise de comportamento em lojas.
* Contagem de objetos móveis em tempo real.

👉 Em resumo:
A **YOLO detecta**, mas para **rastrear** é necessário acoplar um **tracker** como SORT ou DeepSORT. Essa combinação é conhecida como **YOLO + SORT/DeepSORT**, muito usada em sistemas de vigilância e análise de vídeo.

Quer que eu monte um **exemplo prático em Python** mostrando como usar **YOLO + DeepSORT** para rastrear pessoas em um vídeo?
---

Beleza 👌
Vou te mostrar um **exemplo em Python** de como usar **YOLO + DeepSORT** para rastrear pessoas em um vídeo.

---

### 1. Instalar dependências

No terminal do VS Code (ou prompt de comando), execute:

```bash
pip install ultralytics deep-sort-realtime opencv-python
```

---

### 2. Código de exemplo

Crie um arquivo chamado **yolo\_deepsort\_tracking.py** com o seguinte conteúdo:

```python
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
```

---

### 3. Como funciona

1. O **YOLOv8** detecta objetos no vídeo.
2. Apenas pessoas (`cls == 0`) são consideradas.
3. As caixas detectadas são passadas para o **DeepSORT**, que gera IDs e mantém cada pessoa rastreada ao longo dos frames.
4. O vídeo é exibido com bounding boxes e IDs.

---

👉 Pergunta: você quer que eu prepare também um **README passo a passo** (como rodar, trocar vídeo por webcam, ajustar parâmetros) para deixar o projeto redondinho?
---

Perfeito 👌
Segue um **README completo** para o projeto **YOLO + DeepSORT Tracking**.

---

# YOLO + DeepSORT Tracking

Este projeto implementa **detecção e rastreamento de pessoas em tempo real** utilizando **YOLOv8** e **DeepSORT**.
É possível usar tanto **vídeos gravados** quanto a **webcam** do computador.

---

## 🚀 Requisitos

* Python 3.8+
* Pip atualizado

---

## 📦 Instalação

1. Clone o repositório ou copie o código para uma pasta:

   ```bash
   git clone https://github.com/seuusuario/yolo-deepsort-tracking.git
   cd yolo-deepsort-tracking
   ```

2. Instale as dependências necessárias:

   ```bash
   pip install ultralytics deep-sort-realtime opencv-python
   ```

---

## ▶️ Como Executar

1. Coloque um vídeo na pasta do projeto, por exemplo `video.mp4`.

2. Execute o script:

   ```bash
   python yolo_deepsort_tracking.py
   ```

3. Para usar a **webcam** em vez de vídeo, edite esta linha no código:

   ```python
   cap = cv2.VideoCapture(0)
   ```

---

## ⚙️ Configurações Importantes

* **Modelo YOLO**:
  No código, o modelo carregado é o `yolov8s.pt`.
  Você pode trocar por outros modelos disponíveis:

  * `yolov8n.pt` → Mais rápido, menos preciso.
  * `yolov8m.pt` → Equilíbrio.
  * `yolov8l.pt` → Mais pesado e mais preciso.

* **Classes detectadas**:
  Atualmente, só pessoas (`cls == 0`).
  Para rastrear também carros, motos, etc., basta remover o filtro de classe no código.

* **Parâmetros do DeepSORT**:

  * `max_age=30` → quantos frames o ID persiste sem ver o objeto.
  * Pode ser ajustado conforme a aplicação (ex.: em vídeos com muitas oclusões, aumente este valor).

---

## 🎯 Funcionalidades

✅ Detecção de pessoas em tempo real
✅ Atribuição de **IDs únicos** para cada pessoa
✅ Rastreamento robusto mesmo em cruzamentos/oclusões
✅ Suporte a webcam e arquivos de vídeo

---

## 🖥️ Atalhos

* Pressione **Q** para encerrar a execução.

---

## 📌 Exemplo visual (esperado)

Cada pessoa detectada aparece com uma **caixa verde** e um **ID único**, que permanece enquanto ela estiver na cena.

---


