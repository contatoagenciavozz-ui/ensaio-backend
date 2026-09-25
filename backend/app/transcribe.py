"""
Transcrição de fala usando faster-whisper (mesmo motor por trás de
ferramentas como o Descript, rodando localmente no servidor — sem custo
de API por chamada).

O modelo NÃO fica em cache entre chamadas de propósito: no plano
gratuito do Render (512MB de RAM), manter o Whisper carregado ao mesmo
tempo que os modelos do MediaPipe estoura a memória e derruba o
processo sem aviso. Carregar, usar e descartar custa alguns segundos a
mais por vídeo, mas evita esse problema.
"""

import gc
import os
from faster_whisper import WhisperModel

# "base" é mais seguro em memória que "small" para caber nos 512MB do
# plano gratuito ao lado dos modelos do MediaPipe. Se o servidor for para
# um plano com mais RAM, pode voltar para "small" ou "medium" via a
# variável de ambiente WHISPER_MODEL_SIZE, sem mexer no código.
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")


def transcribe_audio(path: str) -> dict:
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    try:
        segments, info = model.transcribe(path, language="pt", vad_filter=True)

        words = []
        full_text_parts = []
        logprob_weighted_sum = 0.0
        logprob_total_duration = 0.0
        for seg in segments:
            full_text_parts.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    words.append({"palavra": w.word.strip(), "inicio_s": round(w.start, 2), "fim_s": round(w.end, 2)})
            seg_dur = max(0.0, seg.end - seg.start)
            if seg_dur > 0 and seg.avg_logprob is not None:
                logprob_weighted_sum += seg.avg_logprob * seg_dur
                logprob_total_duration += seg_dur

        full_text = " ".join(full_text_parts).strip()
        duration_s = info.duration
        word_count = len(full_text.split()) if full_text else 0
        wpm = round(word_count / (duration_s / 60), 1) if duration_s and duration_s > 0 else None

        clareza = None
        if logprob_total_duration > 0:
            avg_logprob = logprob_weighted_sum / logprob_total_duration
            # avg_logprob do Whisper é log-probabilidade média por token —
            # quanto mais perto de 0, mais "confiante" o modelo ficou do que
            # ouviu. Faixas abaixo são chute calibrado por observação geral
            # do modelo, não estudo formal — e o número mistura dicção com
            # ruído de fundo, qualidade do microfone e sotaque, não isola
            # articulação sozinha.
            if avg_logprob > -0.35:
                clareza_label = "alta"
            elif avg_logprob > -0.6:
                clareza_label = "média"
            else:
                clareza_label = "baixa"
            clareza = {
                "estimativa": clareza_label,
                "confianca": "baixa — mistura dicção com ruído de fundo, microfone e sotaque; não isola articulação sozinha"
            }

        return {
            "transcricao": full_text,
            "duracao_s": round(duration_s, 2) if duration_s else None,
            "palavras_total": word_count,
            "palavras_por_minuto": wpm,
            "idioma_detectado": info.language,
            "confianca_idioma": round(info.language_probability, 2) if info.language_probability else None,
            "palavras_com_tempo": words[:2000],
            "clareza_de_fala_estimada": clareza,
        }
    finally:
        # Libera o modelo da memória explicitamente antes do MediaPipe
        # carregar os dele, em vez de esperar o coletor de lixo decidir
        # sozinho quando fazer isso.
        del model
        gc.collect()
