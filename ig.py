import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
import time
import random
import os

options = uc.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--disable-notifications")
# options.add_argument("--headless")
# (a gusto propio, puedes comentar o descomentar esta linea, solo que para debug, recomiendo activarla...)

# Credenciales (rotan entre sesiones)
credentials = [
    ["user", "pasw"] #Aquí las credenciales de la/las cuentas (varias en caso de que IG bloquee la cuenta por bot)
]
acc = 0

def wrt(text, file):
    if text not in open(file + ".data", "r", encoding="utf-8").read():
        with open(file + ".data", "a", encoding="utf-8") as f:
            f.write("\n" + text)

def login():
    global acc, driver
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    # options.add_argument("--headless")
    # mismo que antes

    creden = credentials[acc]
    acc = (acc + 1) % len(credentials)

    driver = uc.Chrome(options=options)
    driver.get("https://www.instagram.com/accounts/login/")

    try:
        reject_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Rechazar cookies opcionales']"))
        )
        reject_button.click()
    except TimeoutException:
        print("No se encontró el botón de cookies. Continuando...")

    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.NAME, "username")))

    username_input = driver.find_element(By.NAME, "username")
    password_input = driver.find_element(By.NAME, "password")

    username_input.send_keys(creden[0])
    password_input.send_keys(creden[1])
    print("🔐 Logging in as", creden[0])
    password_input.send_keys(Keys.RETURN)

    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "//nav")))
    except TimeoutException:
        print("⚠️ No se pudo confirmar login. Verifica cuenta.")
        driver.quit()
        exit()

login()

while True:
    for target in open("followers.data", "r", encoding="utf-8").read().split("\n"):
        if target and target not in open("checked.data", "r", encoding="utf-8").read() and target != "instagram":
            print(f"🔎 Visitando perfil: {target}")
            driver.get(f"https://www.instagram.com/{target}")
            if driver.find_elements(By.XPATH, "//*[contains(text(), 'Publicaciones')]"):
                wrt(target, "checked")
            else:
                continue
            time.sleep(random.uniform(2.0, 4.0))

            try:
                if driver.find_elements(By.XPATH, "//*[contains(text(), 'Esta cuenta es privada')]") or \
                   driver.find_elements(By.XPATH, "//*[contains(text(), 'Aún no hay publicaciones')]"):
                    print("Cuenta privada o sin publicaciones.")
                    continue

                followers_link = WebDriverWait(driver, 15).until(
                    EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "seguidores"))
                )
                followers_link.click()
                time.sleep(random.uniform(3.0, 5.0))

                actions = ActionChains(driver)
                for _ in range(4):
                    time.sleep(0.3)
                    actions.send_keys(Keys.TAB).perform()

                end_time = time.time() + 60
                while time.time() < end_time:
                    actions.send_keys(Keys.DOWN).perform()

                followers_popup = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, "//div[@role='dialog']"))
                )
                popup_html = followers_popup.get_attribute("outerHTML")
                listado = popup_html.split("Foto del perfil de ")
                listado.pop(0)

                for l in listado:
                    username = l.split('"')[0]
                    print(f"[{target}] ➕ {username}")
                    wrt(username, "followers")

                if driver.find_elements(By.XPATH, "//*[contains(text(), 'Ver todas las sugerencias')]"):
                    print("⚠️ SUGERENCIAS DETECTADAS. Reiniciando sesión...")
                    driver.quit()
                    login()
                    break

            except Exception as e:
                print(f"❌ Error analizando {target}: {str(e)}")
