"""
API do Ensaio — recebe um vídeo, devolve transcrição + linguagem corporal
num JSON só. Isso substitui o fluxo manual (Descript + script Python na
mão + colar JSON) por um único upload.

Rodar localmente para testar:
    uvicorn app.main:app --reload --port 8000
Depois abrir http://localhost:8000/docs para testar pelo navegador.
"""

import os
import tempfile
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.transcribe import transcribe_audio
from app.body_analysis import analyze_body_language

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ensaio-api")

app = FastAPI(title="Ensaio — API de análise de discurso")

# Em produção, trocar "*" pela URL real do site do frontend, para não
# aceitar chamada de qualquer origem.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = 500


@app.get("/")
def health_check():
    return {"status": "ok", "service": "ensaio-api"}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    if not video.content_type or not (video.content_type.startswith("video/") or video.content_type.startswith("audio/")):
        raise HTTPException(status_code=400, detail="Envie um arquivo de vídeo ou áudio.")

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        size = 0
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(status_code=413, detail=f"Arquivo maior que {MAX_UPLOAD_MB}MB.")
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        result = {"arquivo": video.filename}

        try:
            logger.info("Transcrevendo %s", video.filename)
            result["transcricao"] = transcribe_audio(tmp_path)
        except Exception as e:
            logger.exception("Falha na transcrição")
            result["transcricao"] = None
            result["erro_transcricao"] = str(e)

        try:
            logger.info("Analisando linguagem corporal %s", video.filename)
            result["linguagem_corporal"] = analyze_body_language(tmp_path)
        except Exception as e:
            logger.exception("Falha na análise corporal")
            result["linguagem_corporal"] = None
            result["erro_linguagem_corporal"] = str(e)

        return result
    finally:
        os.unlink(tmp_path)
