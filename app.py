"""
RoadSense AI — Flask REST API + Web UI
YOLOv8 Object Detection + Groq LLM Agentic Safety Alerts + GPS Incident Logging
SDG 3: Good Health and Well-Being (Road Safety)
SDG 11: Sustainable Cities and Communities
Author: SwasthikaSelvakumar
"""

import os
import io
import json
import logging
import datetime
import requests
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama3-8b-8192"
MODEL_PATH     = os.environ.get("MODEL_PATH", "best.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.40"))
ALERT_WEBHOOK  = os.environ.get("ALERT_WEBHOOK", "")
INCIDENTS_FILE = "incidents.json"

CLASS_LABELS = {0: "pothole", 1: "manhole", 2: "crack"}
RISK_MAP     = {"pothole": "HIGH", "manhole": "HIGH", "crack": "MEDIUM"}


def area_severity(box_area, img_area):
    ratio = box_area / img_area if img_area > 0 else 0
    if ratio > 0.10:
        return "LARGE"
    elif ratio > 0.03:
        return "MEDIUM"
    return "SMALL"


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("alerts.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app   = Flask(__name__, template_folder="templates")
model = None


def load_model():
    global model
    if model is None:
        logger.info("Loading YOLOv8 model from %s …", MODEL_PATH)
        model = YOLO(MODEL_PATH)
        logger.info("YOLOv8 model loaded.")
    return model


# ─────────────────────────────────────────────
# INCIDENT LOG (Enhancement 2 — GPS)
# ─────────────────────────────────────────────
def load_incidents():
    if os.path.exists(INCIDENTS_FILE):
        try:
            with open(INCIDENTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_incident(record):
    incidents = load_incidents()
    incidents.insert(0, record)
    with open(INCIDENTS_FILE, "w") as f:
        json.dump(incidents[:500], f, indent=2)  # keep latest 500


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
def call_groq_llm(detections):
    if not detections:
        det_text = "No road defects detected. Road appears safe."
    else:
        lines = [
            f"- {d['class'].upper()} | size: {d['size']} | confidence: {d['confidence']*100:.1f}%"
            for d in detections
        ]
        det_text = "\n".join(lines)

    prompt = f"""
You are an intelligent road-safety AI assistant for a smart city system.
A YOLOv8 computer vision model has scanned a road image and found:

{det_text}

Respond ONLY with a valid JSON object (no markdown, no backticks, no extra text):
{{
  "explanation"        : "<one clear sentence describing the road condition and danger level>",
  "severity_score"     : <integer 1-10, where 10 is most dangerous>,
  "corrective_action"  : "<one specific action for road authorities or drivers>",
  "alert_priority"     : "<LOW | MEDIUM | HIGH | CRITICAL>"
}}
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 300,
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(raw)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        has_high = any(RISK_MAP.get(d["class"], "LOW") == "HIGH" for d in detections)
        return {
            "explanation": (
                f"Detected {len(detections)} road defect(s). Immediate inspection recommended."
                if detections
                else "Road surface appears clear."
            ),
            "severity_score": 7 if has_high else (4 if detections else 1),
            "corrective_action": (
                "Notify municipal road authority for urgent repair."
                if has_high
                else "Schedule routine inspection."
            ),
            "alert_priority": "HIGH" if has_high else ("MEDIUM" if detections else "LOW"),
        }


# ─────────────────────────────────────────────
# AGENTIC ALERT LOOP
# ─────────────────────────────────────────────
def agentic_alert(detections, llm_result, image_name, lat=None, lng=None):
    priority    = llm_result.get("alert_priority", "LOW")
    alert_fired = priority in ("HIGH", "CRITICAL")

    alert_record = {
        "timestamp":      datetime.datetime.utcnow().isoformat() + "Z",
        "image":          image_name,
        "detections":     detections,
        "severity_score": llm_result.get("severity_score"),
        "explanation":    llm_result.get("explanation"),
        "action":         llm_result.get("corrective_action"),
        "priority":       priority,
        "gps":            {"lat": lat, "lng": lng} if lat and lng else None,
    }

    # Persist GPS incident
    if lat and lng:
        save_incident(alert_record)
        logger.info("📍 GPS incident logged: lat=%s lng=%s priority=%s", lat, lng, priority)

    if alert_fired:
        logger.warning("🚨 ROAD ALERT [%s] | %s", priority, image_name)
        if ALERT_WEBHOOK:
            try:
                requests.post(ALERT_WEBHOOK, json=alert_record, timeout=5)
            except Exception as exc:
                logger.error("Webhook error: %s", exc)

    return {"alert_fired": alert_fired, "priority": priority}


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH,
        "version": "1.0.0",
        "sdg": [
            "SDG 3 — Good Health and Well-Being",
            "SDG 11 — Sustainable Cities and Communities",
        ],
    }), 200


@app.route("/classes", methods=["GET"])
def classes():
    return jsonify({"classes": CLASS_LABELS, "risk_map": RISK_MAP}), 200


@app.route("/incidents", methods=["GET"])
def get_incidents():
    """Return all GPS-tagged incident records."""
    return jsonify({"incidents": load_incidents(), "total": len(load_incidents())}), 200


@app.route("/incidents", methods=["DELETE"])
def clear_incidents():
    """Clear all incidents."""
    if os.path.exists(INCIDENTS_FILE):
        os.remove(INCIDENTS_FILE)
    return jsonify({"status": "cleared"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send file with key 'image'."}), 400

    file     = request.files["image"]
    filename = file.filename or "uploaded_image.jpg"
    raw      = file.read()
    if not raw:
        return jsonify({"error": "Empty image file."}), 400

    # Optional GPS coordinates (Enhancement 2)
    lat = request.form.get("lat") or request.args.get("lat")
    lng = request.form.get("lng") or request.args.get("lng")
    try:
        lat = float(lat) if lat else None
        lng = float(lng) if lng else None
    except ValueError:
        lat = lng = None

    clf = load_model()
    try:
        img        = Image.open(io.BytesIO(raw)).convert("RGB")
        img_w, img_h = img.size
        img_area   = img_w * img_h
        results    = clf.predict(source=img, conf=CONF_THRESHOLD, verbose=False)
        boxes      = results[0].boxes
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        return jsonify({"error": f"Inference failed: {str(exc)}"}), 500

    detections = []
    if boxes is not None and len(boxes):
        for box in boxes:
            cls_id        = int(box.cls[0])
            conf          = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area      = (x2 - x1) * (y2 - y1)
            cls_name      = CLASS_LABELS.get(cls_id, f"class_{cls_id}")
            detections.append({
                "class":      cls_name,
                "confidence": round(conf, 4),
                "risk":       RISK_MAP.get(cls_name, "LOW"),
                "size":       area_severity(box_area, img_area),
                "bbox":       [round(x1), round(y1), round(x2), round(y2)],
            })
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    llm_result = call_groq_llm(detections)
    alert_info = agentic_alert(detections, llm_result, filename, lat=lat, lng=lng)

    response = {
        "image":               filename,
        "total_defects_found": len(detections),
        "detections":          detections,
        "llm_analysis": {
            "explanation":       llm_result.get("explanation"),
            "severity_score":    llm_result.get("severity_score"),
            "corrective_action": llm_result.get("corrective_action"),
            "alert_priority":    llm_result.get("alert_priority"),
        },
        "alert": alert_info,
        "gps":   {"lat": lat, "lng": lng} if lat and lng else None,
        "sdg":   "SDG 3 & SDG 11 — Road Safety & Sustainable Cities",
    }

    logger.info(
        "Processed: %s | Defects: %d | Priority: %s | GPS: %s",
        filename, len(detections), llm_result.get("alert_priority"),
        f"{lat},{lng}" if lat else "none",
    )
    return jsonify(response), 200


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting RoadSense AI on port %d …", port)
    app.run(host="0.0.0.0", port=port, debug=False)