"""
Gestos específicos a partir dos pontos (landmarks) do MediaPipe.

Regras geométricas simples, sem modelo treinado. Cada regra devolve só
"parece que aconteceu" — nunca um veredito sobre o que o gesto significa.
Tudo aqui é função pura (recebe pontos, devolve rótulos), então dá para testar
sem vídeo e sem o MediaPipe instalado.

Convenção dos 21 pontos da mão (MediaPipe Hands):
    0 pulso
    1-4 polegar (4 = ponta)   5-8 indicador (8 = ponta)
    9-12 médio (12 = ponta)   13-16 anelar (16 = ponta)   17-20 mínimo (20 = ponta)
Os pontos vêm normalizados (0 a 1) em relação à largura e à altura da imagem;
por isso multiplicamos x por w e y por h antes de medir distâncias.

Limiares são de trabalho, ainda não calibrados com vídeos reais.
"""
import math

# --- limiares de trabalho -------------------------------------------------
OK_TIP_DIST = 0.45          # ponta do polegar a ponta do indicador, em tamanhos de palma
PYRAMID_TIP_DIST = 0.55     # distância média entre pontas correspondentes das duas mãos
PYRAMID_WRIST_MIN = 0.9     # pulsos afastados (palmas separadas); abaixo disso é "mãos juntas/oração"
FINGER_EXT = 1.25           # ponta / base do dedo, em relação ao pulso: acima = estendido
FINGER_CURL = 1.05          # abaixo = dobrado
POINT_UP_DEG = 35           # indicador a até 35° da vertical conta como "levantado"
MIN_HAND_PX = 40            # mão menor que isso (em pixels, pulso a base do médio) = pequena demais, ignora
HIP_ELBOW_OUT = 0.15        # cotovelo para fora do ombro, em comprimentos de tronco
WINK_CLOSED = 0.6           # olho fechado
WINK_OPEN = 0.3             # olho aberto
HAND_OVERLAP_MAX = 0.15     # mãos que se sobrepõem mais que isso (mãos unidas, entrelaçadas): os pontos ficam confusos, não classifica
WRISTS_TOGETHER = 0.4        # pulsos a menos de 40% do tronco um do outro = mãos juntas (braços soltos ao lado ficam em ~60% ou mais)
SINGLE_HAND = ('ok', 'indicador_levantado', 'apontando', 'dedo_do_meio')
MIN_SAMPLES = 2             # quadros seguidos para virar episódio (piscada: 1)
GAP_S = 0.6

ROTULOS = {
    "ok": "gesto de OK (polegar e indicador em círculo)",
    "indicador_levantado": "dedo indicador levantado",
    "apontando": "apontando com o indicador",
    "dedo_do_meio": "dedo do meio levantado",
    "piramide": "pirâmide de mãos (pontas dos dedos juntas)",
    "mao_na_cintura": "mão na cintura (cotovelo aberto)",
    "piscada": "piscada de um olho só",
}


def _pt(lm, w, h):
    return (lm.x * w, lm.y * h)


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _hand_pts(lms, w, h):
    return [_pt(p, w, h) for p in lms]


def hand_scale(P):
    """Tamanho da palma em pixels: pulso até a base do dedo médio."""
    return _d(P[0], P[9])


def finger_state(P):
    """Para cada dedo (indicador, médio, anelar, mínimo): 'est', 'dob' ou 'meio'."""
    out = {}
    for name, (mcp, tip) in {"ind": (5, 8), "med": (9, 12), "ane": (13, 16), "min": (17, 20)}.items():
        base = _d(P[0], P[mcp])
        if base < 1e-6:
            out[name] = "meio"
            continue
        r = _d(P[0], P[tip]) / base
        out[name] = "est" if r >= FINGER_EXT else ("dob" if r <= FINGER_CURL else "meio")
    return out


def classify_hand(lms, w, h):
    """Rótulos de uma mão só. Devolve lista (pode ser vazia)."""
    P = _hand_pts(lms, w, h)
    s = hand_scale(P)
    if s < MIN_HAND_PX:
        return []
    f = finger_state(P)
    labels = []

    # OK: polegar e indicador se tocam, médio/anelar/mínimo abertos
    if _d(P[4], P[8]) / s < OK_TIP_DIST and f["med"] == "est" and f["ane"] == "est" and f["min"] != "dob":
        labels.append("ok")

    # só o indicador estendido
    if f["ind"] == "est" and f["med"] == "dob" and f["ane"] == "dob" and f["min"] == "dob":
        vx, vy = P[8][0] - P[5][0], P[8][1] - P[5][1]
        ang = abs(math.degrees(math.atan2(vx, -vy)))  # 0 = para cima
        labels.append("indicador_levantado" if ang <= POINT_UP_DEG else "apontando")

    # só o médio estendido
    if f["med"] == "est" and f["ind"] == "dob" and f["ane"] == "dob" and f["min"] == "dob":
        labels.append("dedo_do_meio")
    return labels


def is_pyramid(lms_a, lms_b, w, h):
    """Pontas dos dedos das duas mãos se tocando, com as palmas afastadas."""
    A, B = _hand_pts(lms_a, w, h), _hand_pts(lms_b, w, h)
    s = (hand_scale(A) + hand_scale(B)) / 2
    if hand_scale(A) < MIN_HAND_PX or hand_scale(B) < MIN_HAND_PX:
        return False
    tips = [8, 12, 16, 20]
    mean_tip = sum(_d(A[i], B[i]) for i in tips) / len(tips)
    if mean_tip / s > PYRAMID_TIP_DIST:
        return False
    if _d(A[0], B[0]) / s < PYRAMID_WRIST_MIN:   # pulsos colados = oração / mãos juntas
        return False
    # os dedos precisam estar estendidos (não punho contra punho)
    ext = sum(1 for H in (A, B) for st in finger_state(H).values() if st != "dob")
    return ext >= 6


def is_wink(blink_left, blink_right):
    """Um olho fechado e o outro aberto no mesmo quadro."""
    if blink_left is None or blink_right is None:
        return False
    hi, lo = max(blink_left, blink_right), min(blink_left, blink_right)
    return hi >= WINK_CLOSED and lo <= WINK_OPEN


def is_hand_on_hip(shoulder, hip, elbow, wrist, other_shoulder, trunk_len, wrist_ok, elbow_ok, arm_dropped):
    """Pulso na linha da cintura e cotovelo aberto para fora do corpo."""
    if not (wrist_ok and elbow_ok and arm_dropped) or not trunk_len:
        return False
    cx = (shoulder[0] + other_shoulder[0]) / 2
    out = abs(elbow[0] - cx) - abs(shoulder[0] - cx)
    return out / trunk_len > HIP_ELBOW_OUT


def _bbox(lms, w, h):
    P = _hand_pts(lms, w, h)
    xs, ys = [p[0] for p in P], [p[1] for p in P]
    return (min(xs), min(ys), max(xs), max(ys))


def _overlap(a, b):
    """Interseção das caixas das duas mãos, em fração da menor."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: max(1e-6, (r[2] - r[0]) * (r[3] - r[1]))
    return (ix * iy) / min(area(a), area(b))


def frame_events(t, hands, w, h, blink_l=None, blink_r=None):
    """hands: lista de listas de landmarks (uma por mão detectada no quadro).
    Devolve [(t, rotulo)]."""
    ev = []
    boxes = [_bbox(lms, w, h) for lms in hands]
    for i, lms in enumerate(hands):
        # mãos unidas ou entrelaçadas: o modelo confunde os dedos e inventa gestos
        if any(_overlap(boxes[i], boxes[j]) > HAND_OVERLAP_MAX for j in range(len(hands)) if j != i):
            continue
        for lab in classify_hand(lms, w, h):
            ev.append((t, lab))
    if len(hands) >= 2 and is_pyramid(hands[0], hands[1], w, h):
        ev.append((t, "piramide"))
    if is_wink(blink_l, blink_r):
        ev.append((t, "piscada"))
    return ev


def group(events, seconds_per_sample, total_samples):
    """Junta quadros seguidos do mesmo gesto em episódios.
    events: [(t, rotulo)]. Devolve dict rotulo -> info (só os que apareceram)."""
    by = {}
    for t, lab in events:
        by.setdefault(lab, []).append(t)
    out = {}
    for lab, ts in by.items():
        ts.sort()
        eps, cur = [], None
        for t in ts:
            if cur and t - cur["fim"] <= max(GAP_S, seconds_per_sample * 1.5):
                cur["fim"], cur["n"] = t, cur["n"] + 1
            else:
                if cur:
                    eps.append(cur)
                cur = {"inicio": t, "fim": t, "n": 1}
        if cur:
            eps.append(cur)
        need = 1 if lab == "piscada" else MIN_SAMPLES
        eps = [e for e in eps if e["n"] >= need]
        if not eps:
            continue
        frames = sum(e["n"] for e in eps)
        out[lab] = {
            "descricao": ROTULOS.get(lab, lab),
            "total_episodios": len(eps),
            "pct_dos_quadros": round(100 * frames / total_samples, 1) if total_samples else None,
            "episodios": [{"inicio_s": round(e["inicio"], 1), "fim_s": round(e["fim"], 1)} for e in eps[:15]],
        }
    return out
