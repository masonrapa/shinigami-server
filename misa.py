import os
import sys
import time
import cv2
import joblib
import traceback
import threading
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from deepface import DeepFace

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

DATASET_DIR          = "dataset"
CHECKED_FILE         = "checked.data"
MODEL_FILE           = "death_note.pkl"

DETECTION_BACKENDS   = ["retinaface", "mediapipe", "opencv"]
MODEL_NAME           = "ArcFace"

MAX_IMG_WORKERS      = max(4, os.cpu_count() or 8)

DEEPFACE_MAX_WORKERS = 1

SCROLL_SECONDS       = 10
HEADLESS             = False

PAGELOAD_TIMEOUT_S   = 30
REQ_TIMEOUT_S        = 10
REQ_RETRIES          = 3
REQ_BACKOFF          = 0.7

IMG_EXTS             = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
CDN_PREFIX           = "https://scontent-mad"

def now():
    return datetime.now().strftime("%H:%M:%S")

def log(msg: str):
    tname = threading.current_thread().name
    print(f"[{now()}][{tname}] {msg}", flush=True)

def log_exc(prefix: str, e: Exception):
    log(f"{prefix}: {e.__class__.__name__}: {e}")
    print("".join(traceback.format_exc()), file=sys.stderr, flush=True)

def make_session():
    s = requests.Session()
    retries = Retry(
        total=REQ_RETRIES,
        backoff_factor=REQ_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

REQ = make_session()

MODEL = None
MODEL_LOCK = threading.Lock()

def load_model_incremental(path: str):
    if not os.path.isfile(path):
        log("ℹ️  death_note.pkl no existe, se creará nuevo.")
        return {}
    data = joblib.load(path)
    changed = False
    for k, v in list(data.items()):
        if isinstance(v, list):
            data[k] = {"mean": v, "count": 1}
            changed = True
    if changed:
        joblib.dump(data, path)
        log("🔁 Convertido PKL a formato incremental (mean+count).")
    return data

def save_model_incremental(path: str):
    joblib.dump(MODEL, path)
    log(f"💾 Guardado PKL: {path} (usuarios={len(MODEL)})")

def update_mean_online(old_mean, old_count, new_emb):
    old = np.array(old_mean, dtype=np.float32)
    e   = np.array(new_emb,  dtype=np.float32)
    new_count = old_count + 1
    new_mean  = old + (e - old) / float(new_count)
    return new_mean.tolist(), new_count

def processed_list_path(user: str):
    return os.path.join(DATASET_DIR, user, "processed_images.txt")

def load_processed_images(user: str):
    p = processed_list_path(user)
    if not os.path.isfile(p):
        return set()
    with open(p, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def append_processed_images(user: str, new_names):
    if not new_names:
        return
    p = processed_list_path(user)
    with open(p, "a", encoding="utf-8") as f:
        for name in new_names:
            f.write(name + "\n")

def _dp_extract_faces(image_path: str, backends):
    for backend in backends:
        try:
            _ = DeepFace.extract_faces(img_path=image_path, detector_backend=backend, enforce_detection=True)
            if _ and len(_) > 0:
                return True
        except Exception:
            continue
    return False

def _dp_represent_augmented(image_path: str, backends, model_name: str):
    from PIL import Image, ImageEnhance, ImageOps
    import numpy as np

    def augment(image):
        return [
            image,
            image.rotate(10),
            image.rotate(-10),
            ImageOps.mirror(image),
            ImageEnhance.Brightness(image).enhance(0.8),
            ImageEnhance.Brightness(image).enhance(1.2)
        ]

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return []

    embs = []
    for aug in augment(img):
        arr = np.array(aug)
        for backend in backends:
            try:
                res = DeepFace.represent(
                    img_path=arr, model_name=model_name,
                    enforce_detection=True, detector_backend=backend
                )
                if res and "embedding" in res[0]:
                    e = res[0]["embedding"]
                    if isinstance(e, list) and len(e) == 512:
                        embs.append(e)
                        break
            except Exception:
                continue
    return embs

DEEPFACE_EXEC = None

def deepface_detect_faces(image_path: str) -> bool:
    fut = DEEPFACE_EXEC.submit(_dp_extract_faces, image_path, DETECTION_BACKENDS)
    log(f"🧪 [DF DETECT] Encolado -> {os.path.basename(image_path)}")
    ok = fut.result()
    log(f"🧪 [DF DETECT] Resultado={ok} <- {os.path.basename(image_path)}")
    return ok

def deepface_embed_augs(image_path: str):
    fut = DEEPFACE_EXEC.submit(_dp_represent_augmented, image_path, DETECTION_BACKENDS, MODEL_NAME)
    log(f"🧪 [DF EMBED]  Encolado -> {os.path.basename(image_path)}")
    embs = fut.result()
    log(f"🧪 [DF EMBED]  Devueltos={len(embs)} <- {os.path.basename(image_path)}")
    return embs

def detect_with_haar(image_bgr):
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(70, 70))
    return len(faces) > 0

def is_face_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        log(f"❌ Corrupta: {image_path} -> borrar")
        try: os.remove(image_path)
        except: pass
        return False

    t0 = time.perf_counter()
    haar_ok = detect_with_haar(img)
    t1 = time.perf_counter()
    deep_ok = deepface_detect_faces(image_path)
    t2 = time.perf_counter()

    log("-" * 70)
    log(f"📷 {os.path.basename(image_path)} | HAAR={haar_ok} ({t1-t0:.2f}s)  DEEPFACE={deep_ok} ({t2-t1:.2f}s)")
    log("-" * 70)
    if not (haar_ok and deep_ok):
        log(f"🗑️  Sin cara -> {image_path}")
        try: os.remove(image_path)
        except: log(f"⚠️ No se pudo borrar: {image_path}")
        return False

    return True

def build_driver():
    log("🧭 Construyendo driver Chrome…")
    options = Options()
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")
    options.add_argument("--silent")
    if HEADLESS:
        options.add_argument("--headless=new")
    options.page_load_strategy = "eager"

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(PAGELOAD_TIMEOUT_S)
    driver.implicitly_wait(5)
    log("✅ Driver listo (Selenium Manager)")
    return driver

def handle_cookies(driver):
    try:
        btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Decline optional cookies']"))
        )
        btn.click()
        log("🍪 Cookies rechazadas")
    except:
        log("🍪 Sin pop-up cookies")

def login_instagram(driver, username: str, password: str):
    log("🔐 Login Instagram…")
    driver.get("https://www.instagram.com/accounts/login/")
    handle_cookies(driver)
    WebDriverWait(driver, 25).until(EC.presence_of_element_located((By.NAME, "username")))
    driver.find_element(By.NAME, "username").send_keys(username)
    driver.find_element(By.NAME, "password").send_keys(password + Keys.RETURN)
    time.sleep(10)
    log("✅ Sesión iniciada")

def obtener_url_primera_publicacion(driver):
    try:
        post = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/p/')]"))
        )
        href = post.get_attribute("href")
        log(f"🔗 Primera publicación: {href}")
        return href
    except:
        return None

def es_menor_de_5_anios(valor_datetime):
    try:
        fecha = datetime.strptime(valor_datetime, "%Y-%m-%dT%H:%M:%S.000Z")
        hace_5 = datetime.now() - timedelta(days=5*365)
        return fecha > hace_5
    except:
        return False

def save_profile_html(driver, user: str):
    user_dir = os.path.join(DATASET_DIR, user)
    os.makedirs(user_dir, exist_ok=True)
    info_path = os.path.join(user_dir, "info.data")
    try:
        html = driver.page_source
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"💾 info.data guardado: {info_path} (len={len(html)})")
    except Exception as e:
        log_exc(f"⚠️ info.data {user}", e)

def dload(user: str, url: str, num: int):
    os.makedirs(os.path.join(DATASET_DIR, user), exist_ok=True)
    name = f"{num:05d}.jpg"
    r = REQ.get(url, timeout=REQ_TIMEOUT_S)
    r.raise_for_status()
    out = os.path.join(DATASET_DIR, user, name)
    with open(out, "wb") as f:
        f.write(r.content)
    return name

def download_images_for_user(driver, user: str):
    base_dir = os.path.join(DATASET_DIR, user)
    os.makedirs(base_dir, exist_ok=True)

    log(f"🌐 Abriendo perfil: {user}")
    driver.get(f"https://www.instagram.com/{user}")
    time.sleep(2)

    save_profile_html(driver, user)

    try:
        driver.find_element(By.XPATH, "//*[contains(text(), 'Esta cuenta es privada')]")
        log(f"🔒 {user} es privado. Carpeta creada vacía.")
        return (False, 0)
    except:
        pass

    try:
        driver.find_element(By.XPATH, "//*[contains(text(), 'Aún no hay publicaciones')]")
        log(f"📭 {user} no tiene publicaciones.")
        return (True, 0)
    except:
        pass

    post_url = obtener_url_primera_publicacion(driver)
    if not post_url:
        log(f"❌ {user} no tiene publicaciones visibles.")
        return (True, 0)

    driver.get(post_url)
    time.sleep(1)

    try:
        fecha_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "time")))
        fecha_valor = fecha_element.get_attribute("datetime")
        log(f"🕒 Fecha primera publicación: {fecha_valor}")
        if not es_menor_de_5_anios(fecha_valor):
            log(f"📆 {user} tiene publicaciones antiguas (>5 años). Se omite descarga.")
            return (True, 0)
    except Exception as e:
        log_exc(f"⚠️ No se pudo obtener la fecha en {user}", e)
        return (True, 0)

    driver.get(f"https://www.instagram.com/{user}")
    time.sleep(2)
    actions = ActionChains(driver)
    for _ in range(10):
        try:
            actions.send_keys(Keys.TAB).perform()
        except:
            break
    end_time = time.time() + SCROLL_SECONDS
    while time.time() < end_time:
        actions.send_keys(Keys.DOWN).perform()

    copla = CDN_PREFIX
    num = 0
    img_tags = driver.find_elements(By.TAG_NAME, "img")
    log(f"🔎 {user}: imágenes en DOM: {len(img_tags)}")

    for i in range(len(img_tags)):
        try:
            img = driver.find_elements(By.TAG_NAME, "img")[i]
            code = img.get_attribute("src")
            if code and (copla in code):
                num += 1
                url = copla + code.split(copla)[1].split('"')[0]
                saved = dload(user, url, num)
                if num % 10 == 0:
                    log(f"⬇️ {user}: {num} imgs… (última={saved})")
        except Exception as e:
            log_exc(f"⚠️ Error al obtener imagen {i} de {user}", e)

    log(f"⬇️  {user}: total descargadas = {num}")
    return (True, num)

def integrate_image_if_face(user: str, image_name: str):
    user_dir = os.path.join(DATASET_DIR, user)
    img_path = os.path.join(user_dir, image_name)

    already = load_processed_images(user)
    if image_name in already:
        log(f"⏩ {user}/{image_name} ya integrado antes.")
        return

    if not is_face_image(img_path):
        return

    embs = deepface_embed_augs(img_path)
    if not embs:
        log(f"⚠️ {user}/{image_name}: sin embeddings.")
        return

    with MODEL_LOCK:
        if user not in MODEL:
            MODEL[user] = {"mean": [0.0]*512, "count": 0}
        mean = MODEL[user]["mean"]
        count = MODEL[user]["count"]
        for e in embs:
            mean, count = update_mean_online(mean, count, e)
        MODEL[user]["mean"]  = mean
        MODEL[user]["count"] = count
        append_processed_images(user, [image_name])
        save_model_incremental(MODEL_FILE)
        log(f"✅ Integrada {user}/{image_name} (count={count})")

def process_user(user: str, username: str, password: str):
    t_start = time.perf_counter()
    user_dir = os.path.join(DATASET_DIR, user)
    os.makedirs(user_dir, exist_ok=True)
    driver = None
    try:
        log(f"🚀 INICIO usuario: {user}")
        driver = build_driver()
        login_instagram(driver, username, password)

        is_public, downloaded = download_images_for_user(driver, user)
        log(f"📊 {user}: is_public={is_public}, descargadas={downloaded}")

        images = [f for f in os.listdir(user_dir) if f.lower().endswith(IMG_EXTS)]
        images.sort()
        log(f"🧹 {user}: comprobando {len(images)} imagen(es)…")

        if images:
            with ThreadPoolExecutor(max_workers=MAX_IMG_WORKERS) as pool:
                futures = [pool.submit(integrate_image_if_face, user, name) for name in images]
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        log_exc(f"💥 Integración {user}", e)

        t_elapsed = time.perf_counter() - t_start
        log(f"🏁 FIN usuario: {user} (t={t_elapsed:.1f}s)")

    except Exception as e:
        log_exc(f"💥 Error global en {user}", e)
    finally:
        try:
            if driver:
                driver.quit()
                log(f"🧹 Driver cerrado para {user}")
        except Exception as e:
            log_exc("⚠️ Cierre driver", e)

def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    if not os.path.isfile(CHECKED_FILE):
        log("❌ Falta 'checked.data'")
        sys.exit(1)

    with open(CHECKED_FILE, "r", encoding="utf-8") as f:
        users = [u.strip() for u in f if u.strip()]

    global MODEL
    MODEL = load_model_incremental(MODEL_FILE)

    credentials = [
        ["user", "passw"] #Aquí irían las credenciales (user:pasw)
    ]
    usr, pwd = credentials[0]

    log(f"📌 Usuarios a procesar: {len(users)} | ImgWorkers={MAX_IMG_WORKERS} | DFWorkers={DEEPFACE_MAX_WORKERS}")

    global DEEPFACE_EXEC
    DEEPFACE_EXEC = ProcessPoolExecutor(max_workers=DEEPFACE_MAX_WORKERS)
    log("🧪 Pool DeepFace creado")

    t0 = time.perf_counter()

    for u in users:
        try:
            process_user(u, usr, pwd)
        except Exception as e:
            log_exc(f"💥 Error al procesar {u}", e)

    DEEPFACE_EXEC.shutdown(wait=True, cancel_futures=False)
    log("🧪 Pool DeepFace cerrado")

    t1 = time.perf_counter()
    log(f"✅ Proceso completo finalizado (t={t1-t0:.1f}s).")

if __name__ == "__main__":
    main()