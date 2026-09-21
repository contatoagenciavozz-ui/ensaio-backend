"""
Análise de linguagem corporal — contato visual, olhar, postura, mãos e
expressão facial. Adaptado do script original (analise_corporal.py) para
rodar como parte de uma API, em vez de linha de comando.

A lógica é a mesma que já foi validada localmente — só a forma de chamar
mudou (função em vez de argumento de terminal).
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

FACE_MODEL_PATH = "models/face_landmarker.task"
POSE_MODEL_PATH = "models/pose_landmarker_lite.task"
GESTURE_MODEL_PATH = "models/gesture_recognizer.task"

SAMPLE_FPS = 5
GAZE_DOWN_PITCH_DEG = 12
GAZE_AWAY_YAW_DEG = 20
ARM_DROP_RATIO = 0.08
HAND_MOVEMENT_ACTIVE_THRESHOLD = 0.02
WRIST_VISIBILITY_THRESHOLD = 0.3
POCKET_STREAK_SECONDS = 1.5
SMILE_SCORE_THRESHOLD = 0.35
SERIOUS_SCORE_THRESHOLD = 0.08
UNUSUAL_GESTURE_CONFIDENCE = 0.5
MIN_EPISODE_GAP_S = 0.6


@dataclass
class FrameSignals:
    t: float
    has_face: bool = False
    yaw_deg: float = None
    pitch_deg: float = None
    smile_score: float = None
    has_pose: bool = False
    left_arm_dropped: bool = None
    right_arm_dropped: bool = None
    left_wrist: tuple = None
    right_wrist: tuple = None
    left_wrist_visible: bool = None
    right_wrist_visible: bool = None


def get_blendshape_score(categories, name):
    for c in categories:
        if c.category_name == name:
            return c.score
    return 0.0


def estimate_head_pose(face_landmarks, image_w, image_h):
    idxs = {"nose": 1, "chin": 152, "eye_l": 263, "eye_r": 33, "mouth_l": 287, "mouth_r": 57}
    image_points = np.array([
        (face_landmarks[idxs["nose"]].x * image_w, face_landmarks[idxs["nose"]].y * image_h),
        (face_landmarks[idxs["chin"]].x * image_w, face_landmarks[idxs["chin"]].y * image_h),
        (face_landmarks[idxs["eye_l"]].x * image_w, face_landmarks[idxs["eye_l"]].y * image_h),
        (face_landmarks[idxs["eye_r"]].x * image_w, face_landmarks[idxs["eye_r"]].y * image_h),
        (face_landmarks[idxs["mouth_l"]].x * image_w, face_landmarks[idxs["mouth_l"]].y * image_h),
        (face_landmarks[idxs["mouth_r"]].x * image_w, face_landmarks[idxs["mouth_r"]].y * image_h),
    ], dtype=np.float64)
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
    ], dtype=np.float64)
    focal_length = image_w
    center = (image_w / 2, image_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not ok:
        return None, None
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
    yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    return yaw, pitch


def check_arm_dropped(shoulder, hip, wrist):
    trunk_len = math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1])
    if trunk_len < 1e-6:
        return None
    dist_wrist_hip = math.hypot(wrist[0] - hip[0], wrist[1] - hip[1])
    return (dist_wrist_hip / trunk_len) < ARM_DROP_RATIO


def group_unusual_gestures(raw_events, max_gap_s):
    if not raw_events:
        return []
    by_hand = {}
    for t, hand, category, score in raw_events:
        by_hand.setdefault(hand, []).append((t, category, score))
    episodes = []
    for hand, events in by_hand.items():
        events.sort(key=lambda e: e[0])
        current = None
        for t, category, score in events:
            if current is None or t - current["end"] > max_gap_s:
                if current:
                    episodes.append(current)
                current = {"mao": hand, "inicio_s": round(t, 1), "end": t, "categoria_mais_vista": category, "menor_confianca": score, "n_quadros": 1}
            else:
                current["end"] = t
                current["n_quadros"] += 1
                current["menor_confianca"] = min(current["menor_confianca"], score)
            current["fim_s"] = round(current["end"], 1)
        if current:
            episodes.append(current)
    episodes.sort(key=lambda e: e["inicio_s"])
    for e in episodes:
        del e["end"]
    return episodes


def longest_invisible_streak_seconds(pose_frames, side_attr):
    longest = 0
    current = 0
    current_start = None
    best_start = None
    for s in pose_frames:
        visible = getattr(s, side_attr)
        if visible is False:
            if current == 0:
                current_start = s.t
            current += 1
        else:
            if current > longest:
                longest = current
                best_start = current_start
            current = 0
    if current > longest:
        longest = current
        best_start = current_start
    if longest == 0 or len(pose_frames) < 2:
        return 0.0, None
    avg_gap = (pose_frames[-1].t - pose_frames[0].t) / max(1, len(pose_frames) - 1)
    return round(longest * avg_gap, 1), (round(best_start, 1) if best_start is not None else None)


def analyze_body_language(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Não consegui abrir o vídeo: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, round(src_fps / SAMPLE_FPS))

    face_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
    )
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO, num_poses=1,
    )
    gesture_options = mp_vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO, num_hands=2,
    )

    signals = []
    prev_wrists = {"left": None, "right": None}
    hand_movement_frames = {"left": 0, "right": 0}
    both_wrists_hidden_frames = 0
    both_wrists_hidden_longest = 0
    both_wrists_hidden_current = 0
    unusual_gesture_raw = []

    with mp_vision.FaceLandmarker.create_from_options(face_options) as face_lm, \
         mp_vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         mp_vision.GestureRecognizer.create_from_options(gesture_options) as gesture_lm:

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            t_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            sig = FrameSignals(t=t_ms / 1000.0)

            face_result = face_lm.detect_for_video(mp_image, t_ms)
            if face_result.face_landmarks:
                yaw, pitch = estimate_head_pose(face_result.face_landmarks[0], w, h)
                if yaw is not None:
                    sig.has_face = True
                    sig.yaw_deg = yaw
                    sig.pitch_deg = pitch
                if face_result.face_blendshapes:
                    cats = face_result.face_blendshapes[0]
                    sig.smile_score = (get_blendshape_score(cats, "mouthSmileLeft") + get_blendshape_score(cats, "mouthSmileRight")) / 2

            pose_result = pose_lm.detect_for_video(mp_image, t_ms)
            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks[0]
                l_sh, r_sh = (lm[11].x, lm[11].y), (lm[12].x, lm[12].y)
                l_hip, r_hip = (lm[23].x, lm[23].y), (lm[24].x, lm[24].y)
                l_wr, r_wr = (lm[15].x, lm[15].y), (lm[16].x, lm[16].y)

                sig.has_pose = True
                sig.left_wrist, sig.right_wrist = l_wr, r_wr
                sig.left_arm_dropped = check_arm_dropped(l_sh, l_hip, l_wr)
                sig.right_arm_dropped = check_arm_dropped(r_sh, r_hip, r_wr)
                sig.left_wrist_visible = lm[15].visibility > WRIST_VISIBILITY_THRESHOLD
                sig.right_wrist_visible = lm[16].visibility > WRIST_VISIBILITY_THRESHOLD

                for side, wrist in (("left", l_wr), ("right", r_wr)):
                    if prev_wrists[side] is not None:
                        d = math.hypot(wrist[0] - prev_wrists[side][0], wrist[1] - prev_wrists[side][1])
                        if d > HAND_MOVEMENT_ACTIVE_THRESHOLD:
                            hand_movement_frames[side] += 1
                    prev_wrists[side] = wrist

                if not sig.left_wrist_visible and not sig.right_wrist_visible:
                    both_wrists_hidden_frames += 1
                    both_wrists_hidden_current += 1
                    both_wrists_hidden_longest = max(both_wrists_hidden_longest, both_wrists_hidden_current)
                else:
                    both_wrists_hidden_current = 0

            gesture_result = gesture_lm.recognize_for_video(mp_image, t_ms)
            if gesture_result.gestures:
                for hand_idx, hand_gestures in enumerate(gesture_result.gestures):
                    if not hand_gestures:
                        continue
                    top = hand_gestures[0]
                    hand_label = "desconhecida"
                    if gesture_result.handedness and hand_idx < len(gesture_result.handedness):
                        hand_label = gesture_result.handedness[hand_idx][0].category_name
                    if top.category_name == "None" or top.score < UNUSUAL_GESTURE_CONFIDENCE:
                        unusual_gesture_raw.append((sig.t, hand_label, top.category_name, round(top.score, 2)))

            signals.append(sig)
            frame_idx += 1

    cap.release()
    seconds_per_sample = frame_interval / max(src_fps, 1)
    return summarize(signals, hand_movement_frames, both_wrists_hidden_frames, both_wrists_hidden_longest, seconds_per_sample, unusual_gesture_raw)


def summarize(signals, hand_movement_frames, both_wrists_hidden_frames, both_wrists_hidden_longest, seconds_per_sample, unusual_gesture_raw):
    face_frames = [s for s in signals if s.has_face]
    n_face = len(face_frames)
    down_frames = sum(1 for s in face_frames if s.pitch_deg is not None and s.pitch_deg < -GAZE_DOWN_PITCH_DEG)
    away_frames = sum(1 for s in face_frames if s.yaw_deg is not None and abs(s.yaw_deg) > GAZE_AWAY_YAW_DEG)
    forward_frames = n_face - down_frames - away_frames

    smile_frames = [s for s in face_frames if s.smile_score is not None]
    n_smile = len(smile_frames)
    smiling_frames = sum(1 for s in smile_frames if s.smile_score > SMILE_SCORE_THRESHOLD)
    serious_frames = sum(1 for s in smile_frames if s.smile_score < SERIOUS_SCORE_THRESHOLD)

    pose_frames = [s for s in signals if s.has_pose]
    n_pose = len(pose_frames)
    left_dropped = sum(1 for s in pose_frames if s.left_arm_dropped)
    right_dropped = sum(1 for s in pose_frames if s.right_arm_dropped)

    def pct(n, total):
        return round(100 * n / total, 1) if total else None

    left_streak_s, left_streak_start = longest_invisible_streak_seconds(pose_frames, "left_wrist_visible")
    right_streak_s, right_streak_start = longest_invisible_streak_seconds(pose_frames, "right_wrist_visible")
    behind_back_s = round(both_wrists_hidden_longest * seconds_per_sample, 1)

    expressao_label = None
    if n_smile:
        smile_pct_val = pct(smiling_frames, n_smile)
        serious_pct_val = pct(serious_frames, n_smile)
        if smile_pct_val is not None and smile_pct_val > 40:
            expressao_label = "sorriso frequente"
        elif serious_pct_val is not None and serious_pct_val > 60:
            expressao_label = "expressão muito séria"
        else:
            expressao_label = "neutra predominante"

    gesture_episodes = group_unusual_gestures(unusual_gesture_raw, MIN_EPISODE_GAP_S)

    return {
        "quadros_analisados": len(signals),
        "quadros_com_rosto_detectado": n_face,
        "contato_visual_estimado_pct": pct(forward_frames, n_face),
        "olhou_para_baixo_pct": pct(down_frames, n_face),
        "desviou_o_olhar_pct": pct(away_frames, n_face),
        "expressao_facial": {
            "classificacao": expressao_label,
            "sorriso_pct": pct(smiling_frames, n_smile),
            "serio_pct": pct(serious_frames, n_smile),
        } if n_smile else None,
        "quadros_com_pose_detectada": n_pose,
        "mao_na_linha_da_cintura_esquerda_pct": pct(left_dropped, n_pose),
        "mao_na_linha_da_cintura_direita_pct": pct(right_dropped, n_pose),
        "mao_esquerda_em_movimento_pct": pct(hand_movement_frames["left"], n_pose),
        "mao_direita_em_movimento_pct": pct(hand_movement_frames["right"], n_pose),
        "maos_atras_das_costas": {
            "maior_periodo_ambas_ocultas_s": behind_back_s,
            "confianca": "baixa — não distingue mãos atrás das costas de virar de costas ou sair do quadro"
        } if behind_back_s >= POCKET_STREAK_SECONDS else None,
        "mao_esquerda_possivel_bolso": {
            "maior_periodo_sem_deteccao_s": left_streak_s, "inicio_aprox_s": left_streak_start,
            "confianca": "baixa — não distingue bolso de virar de costas ou sair do quadro"
        } if left_streak_s >= POCKET_STREAK_SECONDS else None,
        "mao_direita_possivel_bolso": {
            "maior_periodo_sem_deteccao_s": right_streak_s, "inicio_aprox_s": right_streak_start,
            "confianca": "baixa — não distingue bolso de virar de costas ou sair do quadro"
        } if right_streak_s >= POCKET_STREAK_SECONDS else None,
        "formato_de_mao_incomum": {
            "total_episodios": len(gesture_episodes), "episodios": gesture_episodes[:20],
            "nota": "Formato de mão que não bateu com os 7 gestos conhecidos do MediaPipe (ou bateu com baixa confiança) — inclui coisas banais como ajustar o microfone, coçar o rosto ou segurar um cartão. Lista de 'dá uma olhada aqui', não classificação do que é."
        } if gesture_episodes else None,
        "aviso": (
            "Percentuais por quadro amostrado, não por tempo de fala. Limiares são heurísticos — "
            "calibre contra vídeos anotados manualmente antes de confiar no valor absoluto. Sinais de "
            "oclusão (bolso, atrás das costas, formato incomum) são pistas de baixa confiança, nunca "
            "veredito automático."
        ),
    }
