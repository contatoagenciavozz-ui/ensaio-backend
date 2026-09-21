"""
Transcrição de fala usando faster-whisper (mesmo motor por trás de
ferramentas como o Descript, rodando localmente no servidor — sem custo
de API por chamada).
"""

from faster_whisper import WhisperModel

# "small" é um bom equilíbrio para português em CPU: mais preciso que
# "base", ainda razoável em velocidade sem GPU. Trocar para "medium" se a
# precisão não for suficiente — custa mais tempo de processamento.
MODEL_SIZE = "small"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe_audio(path: str) -> dict:
    model = get_model()
    segments, info = model.transcribe(path, language="pt", vad_filter=True)

    words = []
    full_text_parts = []
    for seg in segments:
        full_text_parts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                words.append({"palavra": w.word.strip(), "inicio_s": round(w.start, 2), "fim_s": round(w.end, 2)})

    full_text = " ".join(full_text_parts).strip()
    duration_s = info.duration
    word_count = len(full_text.split()) if full_text else 0
    wpm = round(word_count / (duration_s / 60), 1) if duration_s and duration_s > 0 else None

    return {
        "transcricao": full_text,
        "duracao_s": round(duration_s, 2) if duration_s else None,
        "palavras_total": word_count,
        "palavras_por_minuto": wpm,
        "idioma_detectado": info.language,
        "confianca_idioma": round(info.language_probability, 2) if info.language_probability else None,
        "palavras_com_tempo": words[:2000],  # limite de segurança para respostas muito longas
    }
