FROM python:3.11-slim

# Dependências de sistema: ffmpeg para decodificar áudio/vídeo,
# libs que o OpenCV e o MediaPipe esperam encontrar mesmo em modo headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baixa os 3 modelos do MediaPipe já no build, para não pagar esse tempo
# (e essa banda) a cada cold start do servidor.
RUN mkdir -p models && \
    wget -q -O models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task && \
    wget -q -O models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task && \
    wget -q -O models/gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task

COPY app/ app/

# Pré-baixa o modelo do Whisper também no build, pelo mesmo motivo.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
