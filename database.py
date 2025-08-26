from googlesearch import search
from fuzzywuzzy import fuzz
import pygetwindow as gw
import pycountry
import requests
import sqlite3
asoc = {}

nombres_comunes = [
    "Antonio", "Mykel", "Jose", "Manuel", "Francisco", "David", "Juan", "Javier", "Luis", "Carlos", "Daniel",
    "Miguel", "Rafael", "Alejandro", "Pedro", "Fernando", "Jorge", "Sergio", "Alberto", "Alvaro", "Pablo",
    "Ramon", "Vicente", "Angel", "Mario", "Diego", "Ruben", "Adrian", "Andres", "Joaquin",
    "Enrique", "Cristian", "Eduardo", "Santiago", "Ivan", "Ricardo", "Emilio", "Esteban", "Hector", "Tomas",
    "Gabriel", "Gonzalo", "Julio", "Nicolas", "Oscar", "Sebastian", "Hugo", "Felipe",
    "Alfonso", "Roberto", "Ismael", "Mateo", "Samuel", "Rodrigo", "Gerardo", "Jaime", "Iker", "Marcos",
    "John", "Michael", "James", "Robert", "David", "William", "Joseph", "Charles", "Thomas", "Daniel",
    "Matthew", "Anthony", "Steven", "Paul", "Andrew", "Joshua", "Kevin", "Brian",
    "George", "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan", "Jacob", "Nicholas", "Adam",
    "Ethan", "Isaac", "Victor", "Dylan", "Luis", "Carlos", "Felipe", "Mateo", "Rafael"
]

def getcountry(codigo):
    return next((pais.name for pais in pycountry.countries if pais.alpha_2 == codigo.upper()),codigo)

def hack(nick):
    global asoc
    payload = ""
    conn = sqlite3.connect("dbs/"+nick[0].lower()+".db")
    cursor = conn.cursor()
    cursor.execute("SELECT nick, ip, password FROM clients")
    lines = cursor.fetchall()
    for line in lines:
        nick_db, ip, password = line
        if fuzz.ratio(nick.lower(), nick_db.lower()) > 85:
            password_info = password if password else "not"
            payload += "🚨 "+nick_db+" 🚨\n"
            payload += "🌍 LOCALIZADO: "+geoloc(ip)+"\n"
            payload += "⚠️ HACKED! IP: "+ip+"\n"
            if (password_info != "not"): 
                payload += "💀 KILLED! PASSWORD: "+password_info+"\n"
            payload += "\n"
    conn.close()
    return payload

def nombrar(ig):
    response = requests.get(f"https://www.instagram.com/{ig}/").text
    for n in nombres_comunes:
        if (n.lower() in response.lower()):
            return "👁  POSIBLE NOMBRE: "+n
        return ""

def geoloc(ip):
    url = f"https://ipinfo.io/{ip}/json"
    response = requests.get(url)
    data = response.json()
    pais = getcountry(data.get("country", "N/A"))
    ciudad = data.get("city", "N/A")
    coordenadas = data.get("loc", "N/A")
    return f"{pais} - {ciudad} ({coordenadas})"

def dox(nick):
    global asoc
    payload = ""
    data = []
    for j in search("intext:" + nick, num_results=20, sleep_interval=3):
        if j not in data:
            data.append(j)
    if (len(data) > 0):
        for d in data:
            if "instagram" in d:
                payload += nombrar(d.split("/")[3])
                break
        payload += "🚨 SCANNING: " + nick+"\n"
        for d in data:
            if (len(d) < 70):
                prefix = "[OTHER 🔍] "
                if "instagram.com" in d:
                    prefix = "[INSTA 🔍] "
                elif "twitter.com" in d:
                    prefix = "[TWITT 🔍] "
                elif "namemc.com" in d:
                    prefix = "[MINEC 🔍] "
                elif "facebook.com" in d:
                    prefix = "[FACEB 🔍] "
                elif "pinterest.com" in d:
                    prefix = "[PINTE 🔍] "
                elif "start.gg" in d:
                    prefix = "[SMASH 🔍] "
                elif "linkedin.com" in d:
                    prefix = "[LINKD 🔍] "
                elif "tiktok.com" in d:
                    prefix = "[TKTOK 🔍] "
                elif "threads.net" in d:
                    prefix = "[THRDS 🔍] "
                payload += (prefix + d+"\n\n")
        return payload