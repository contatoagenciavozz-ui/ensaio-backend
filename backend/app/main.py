"""
API do Ensaio — recebe um vídeo, devolve transcrição + linguagem corporal
num JSON só.

Processamento é feito em segundo plano (job assíncrono com polling), não
numa única chamada bloqueante: vídeo pode levar minutos para processar, e
a maioria dos proxies na internet (incluindo o do Render) derruba conexões
HTTP paradas por muito tempo. O fluxo é:

    1. POST /analyze  -> responde na hora com {"job_id": "..."}
    2. GET /status/{job_id} -> chamado repetidamente (polling) até
       status virar "done" (ou "error")

Rodar localmente para testar:
    uvicorn app.main:app --reload --port 8000
Depois abrir http://localhost:8000/docs
"""

import os
import uuid
import tempfile
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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

# Estado dos jobs em memória — simples e suficiente para uma única
# instância. Se o serviço reiniciar no meio de um job, esse job se perde;
# aceitável para o estágio atual do projeto.
jobs: dict = {}


@app.get("/")
def health_check():
    return {"status": "ok", "service": "ensaio-api"}


def process_video(job_id: str, tmp_path: str, filename: str):
    try:
        result = {"arquivo": filename}

        try:
            logger.info("[%s] Transcrevendo", job_id)
            result["transcricao"] = transcribe_audio(tmp_path)
        except Exception as e:
            logger.exception("[%s] Falha na transcrição", job_id)
            result["transcricao"] = None
            result["erro_transcricao"] = str(e)

        try:
            logger.info("[%s] Analisando linguagem corporal", job_id)
            result["linguagem_corporal"] = analyze_body_language(tmp_path)
        except Exception as e:
            logger.exception("[%s] Falha na análise corporal", job_id)
            result["linguagem_corporal"] = None
            result["erro_linguagem_corporal"] = str(e)

        jobs[job_id] = {"status": "done", "result": result}
        logger.info("[%s] Concluído", job_id)
    except Exception as e:
        logger.exception("[%s] Falha inesperada", job_id)
        jobs[job_id] = {"status": "error", "detail": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/analyze")
async def analyze(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
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

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing"}
    background_tasks.add_task(process_video, job_id, tmp_path, video.filename)

    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id não encontrado.")
    return job
