# app.py
from __future__ import annotations
import os, json, uuid
from typing import Dict, List
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from flask import send_from_directory
from src.captioner import generate_caption 

from src.geo_labeler_ext import GeoVisionLabeler, LabelerConfig

# ---------------- Config ----------------
SECRET_KEY = os.environ.get("FLASK_SECRET", "devsecret")
UPLOAD_DIR = "uploads"
USER_DB = "users.json"   # stored usernames + hashed passwords if "remember me"
ALLOWED_EXT = {"jpg","jpeg","png","tif","tiff"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = SECRET_KEY

# ---------- users storage (simple JSON file) ----------
def load_users() -> Dict[str,str]:
    if os.path.exists(USER_DB):
        return json.load(open(USER_DB, "r"))
    return {}

def save_users(d: Dict[str,str]):
    with open(USER_DB, "w") as f:
        json.dump(d, f, indent=2)

# --------- auth helpers ----------
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap

# --------- Routes ----------
@app.route("/", methods=["GET"])
def root():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        remember = request.form.get("remember") == "on"

        users = load_users()
        # If user exists: verify. If not exists and remember checked: create.
        if username in users:
            if not check_password_hash(users[username], password):
                flash("Invalid credentials.", "error")
                return render_template("login.html")
        else:
            if remember:
                users[username] = generate_password_hash(password)
                save_users(users)
            else:
                # ephemeral user (session-only)
                pass

        session["user"] = username
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

def allowed(filename: str)->bool:
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXT

# Initialize labeler once (classes editable on dashboard via textarea)
DEFAULT_CLASSES = ["Buildings","No Buildings"]
CFG = LabelerConfig(classes=DEFAULT_CLASSES)
LABELER = GeoVisionLabeler(CFG)

@app.route("/classify", methods=["POST"])
@login_required
def classify():
    # Read classes from form (one per line)
    classes_txt = request.form.get("classes","").strip()
    classes = [c.strip() for c in classes_txt.splitlines() if c.strip()] or DEFAULT_CLASSES
    # Update runtime classes
    global LABELER, CFG
    CFG = LabelerConfig(classes=classes)
    LABELER = GeoVisionLabeler(CFG)

    # (optional) captions on/off
    want_captions = request.form.get("captions") == "on"

    files = request.files.getlist("images")
    out: List[Dict] = []
    for f in files:
        if f and allowed(f.filename):
            fname = secure_filename(f.filename)
            # save with a UUID prefix to avoid collisions
            saved_name = f"{uuid.uuid4().hex}_{fname}"
            path = os.path.join(UPLOAD_DIR, saved_name)
            f.save(path)

            # (Optional) captions
            description = ""
            if want_captions:
                from PIL import Image
                pil = Image.open(path).convert("RGB")
                description = generate_caption(pil, prompt="Brief, concrete satellite caption:")

            # (Optional) if you later pass an LLM label, put it here instead of None
            result = LABELER.classify_image(path, llm_pred_label=None, description=description)

            # add a URL we can show as a thumbnail in results
            result["image_url"] = url_for("uploaded_file", filename=saved_name)
            out.append(result)

    return render_template("results.html", results=out, classes=classes)



@app.route("/api/results_json", methods=["POST"])
@login_required
def api_results_json():
    data = request.get_json(force=True)
    # expects { "images":[{"path":...}, ...], "classes":[...], "captions":true/false }
    classes = data.get("classes") or DEFAULT_CLASSES
    do_caps = bool(data.get("captions", False))
    # update labeler
    global LABELER, CFG
    CFG = LabelerConfig(classes=classes)
    LABELER = GeoVisionLabeler(CFG)

    out = []
    for item in data.get("images", []):
        p = item["path"]
        desc = ""
        if do_caps:
            desc = "A road with sparse vegetation nearby"  # placeholder
        out.append(LABELER.classify_image(p, llm_pred_label=None, description=desc))
    return jsonify(out)

@app.route("/export_csv", methods=["POST"])
@login_required
def export_csv():
    """
    Body: JSON list like:
      [{"filename":..., "predicted_label":..., "confidence":..., "description":..., "locality":..., "mode":...}, ...]
    Returns: CSV file for manual QA.
    """
    import csv, time
    rows = request.get_json(force=True) or []
    fields = ["filename","predicted_label","confidence","description","locality","mode"]
    os.makedirs("data", exist_ok=True)
    fn = f"labels_{int(time.time())}.csv"
    fp = os.path.join("data", fn)
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return send_file(fp, as_attachment=True, download_name=fn)

@app.route("/healthz")
def healthz():
    return {"ok": True}, 200
