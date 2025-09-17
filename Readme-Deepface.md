Perfeito 🚀 Vou deixar o README mais moderno e visual com **emojis** e seções bem destacadas.

---

# 📖 README — YOLO + DeepSORT + DeepFace

**🔎 Descrição curta:**
Sistema de **detecção (YOLOv8)** + **rastreamento (DeepSORT)** + **reconhecimento facial (DeepFace)**.
Ele detecta pessoas em vídeo, atribui um `track_id` único a cada uma, compara com fotos cadastradas e salva um log de identificações.

---

## 🛠️ Requisitos

* 🐍 **Python 3.8+**
* 📦 **pip atualizado**
* ⚡ **GPU/CUDA opcional** (recomendado para desempenho — instale PyTorch com CUDA se disponível)

---

## ⚙️ Instalação

```bash
# 1️⃣ Criar ambiente virtual
python -m venv venv

# Ativar venv (Linux/macOS)
source venv/bin/activate

# Ativar venv (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 2️⃣ Atualizar pip e instalar dependências
pip install --upgrade pip
pip install ultralytics deepface opencv-python deep-sort-realtime numpy
```

💡 Para usar **GPU**, instale PyTorch com CUDA antes (siga docs oficiais do PyTorch).

---

## 📂 Estrutura do Projeto

```
yolo-deepface-multi/
│── cadastro.py                  # captura fotos de pessoas
│── build_embeddings.py          # gera embeddings.pkl
│── yolo_face_gallery_tracker.py # rastreamento + reconhecimento + log
│── gallery/                     # fotos cadastradas (ex: joao_1.jpg, maria_1.jpg)
│── embeddings.pkl               # gerado a partir da galeria
│── log_identificacoes.csv       # registros das identificações
│── video.mp4                    # vídeo de teste (opcional)
```

---

## 👤 Passo 1 — Cadastrar Pessoas

📸 Execute o cadastro para salvar fotos na pasta `gallery/`:

```bash
python cadastro.py
```

👉 Digite o nome da pessoa, posicione-a na frente da webcam e pressione **ESPAÇO** para capturar cada foto.
👉 Pressione **ESC** para finalizar.
👉 As imagens ficam salvas como `nome_1.jpg`, `nome_2.jpg`...

---

## 🧠 Passo 2 — Gerar Embeddings

Cria o arquivo `embeddings.pkl` com representações faciais da galeria.

```bash
python build_embeddings.py
```

---

## 🎥 Passo 3 — Rodar Rastreamento + Reconhecimento

Execute o rastreamento com YOLO + DeepSORT + DeepFace:

```bash
python yolo_face_gallery_tracker.py
```

* Por padrão, roda no `video.mp4`.
* Para usar a **webcam**, edite no código:

```python
cap = cv2.VideoCapture("video.mp4")
# altere para:
cap = cv2.VideoCapture(0)
```

---

## 🔄 Como Funciona

1️⃣ **YOLOv8** detecta pessoas em cada frame.
2️⃣ **DeepSORT** gera um `track_id` único por indivíduo.
3️⃣ Para cada novo `track_id`:

* recorta o rosto,
* gera embedding com **DeepFace**,
* compara com `embeddings.pkl`,
* atribui nome se a distância < **0.7** (ajustável).
  4️⃣ Resultados aparecem na tela e são salvos em `log_identificacoes.csv`.

---

## 📝 Exemplo de Log

Arquivo `log_identificacoes.csv`:

```
timestamp,track_id,identity,distance
2025-09-17 18:35:01,1,joao,0.4231
2025-09-17 18:35:02,2,maria,0.3675
2025-09-17 18:35:04,3,Desconhecido,0.9123
```

---

## 🔧 Parâmetros Ajustáveis

* 📌 **Modelo YOLO**: troque `"yolov8n.pt"` por `"yolov8s.pt"`, `"yolov8m.pt"`...
* 🎯 **Threshold** de reconhecimento (`0.7` default). Menor = mais rígido.
* 👥 **DeepSORT**: ajuste `max_age`, `n_init`, `max_cosine_distance` para melhorar rastreamento.
* ⏱️ Controle de log: pode ser configurado para evitar registros duplicados em curtos intervalos.

---

## ⚡ Dicas de Performance & Precisão

* Use **várias fotos por pessoa** em diferentes ângulos e iluminação.
* O modelo **ArcFace** (DeepFace) costuma ser o mais estável.
* Com **GPU**, o desempenho melhora drasticamente.
* Se detectar muitos falsos positivos → aumente confiança do YOLO (`conf > 0.5`).
* Para ambientes complexos, combine com **Person Re-ID** (OSNet, FastReID).

---

## 🛑 Problemas Comuns

* ❌ **Câmera não abre** → teste índices `0`, `1`, `2`.
* ⚠️ **Erro DeepFace** → use `enforce_detection=False` (já no código).
* 🐢 **Lento** → use GPU, YOLO mais leve (`yolov8n`) ou reduza resolução do vídeo.
* 🔄 **Identificações repetidas** → já mitigado com `track_id`, mas pode aumentar intervalo mínimo de log.

---

## ✅ Conclusão

Esse sistema entrega:

* 🚀 **Detecção rápida com YOLOv8**
* 👤 **Rastreamento consistente com DeepSORT**
* 🧠 **Reconhecimento facial via embeddings DeepFace**
* 📝 **Log estruturado para auditoria e análise**

---

