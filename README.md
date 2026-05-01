# 🚧 Pothole Detection — YOLOv8 + LLM Agentic Alerts

> **SDG 3 — Good Health & Well-Being** | **SDG 11 — Sustainable Cities**  
> Real-time road defect detection with AI-powered safety alerts

---

## 📌 Overview

This project detects road defects (potholes, manholes, cracks) using **YOLOv8**
and enhances it with a **Groq LLM agentic loop** that automatically:
- Generates natural language road safety explanations
- Assigns a severity score (1–10)
- Recommends corrective actions
- Fires alerts when risk is HIGH or CRITICAL

---

## 🏗️ Architecture

```
Road Image → YOLOv8 Detection → Detections List
                                      ↓
                              Groq LLM (LLaMA 3)
                                      ↓
                    Explanation + Severity + Action
                                      ↓
                         Agentic Alert Loop
                    (Log / Webhook if HIGH/CRITICAL)
                                      ↓
                              JSON Response
```

---

## 🚀 Quick Start

### Option 1 — Run Directly with Python

```bash
git clone https://github.com/venkatavamsi01/Pothole-Detection-using-YOLOV8.git
cd Pothole-Detection-using-YOLOV8

pip install -r requirements.txt

export GROQ_API_KEY=your_groq_key_here
python app.py
```

### Option 2 — Run with Docker

```bash
export GROQ_API_KEY=your_groq_key_here
./build_and_run.sh all
```

---

## 🔌 API Endpoints

| Method | Endpoint     | Description                        |
|--------|--------------|------------------------------------|
| GET    | `/health`    | Health check + model info          |
| GET    | `/classes`   | List detection classes + risk map  |
| POST   | `/predict`   | Detect defects + LLM analysis      |

### Example Request

```bash
curl -X POST http://localhost:5000/predict \
     -F "image=@road_sample.jpg"
```

### Example Response

```json
{
  "image": "road_sample.jpg",
  "total_defects_found": 2,
  "detections": [
    {
      "class": "pothole",
      "confidence": 0.87,
      "risk": "HIGH",
      "size": "LARGE",
      "bbox": [120, 340, 280, 480]
    }
  ],
  "llm_analysis": {
    "explanation": "A large pothole detected at road center poses high risk for two-wheelers.",
    "severity_score": 8,
    "corrective_action": "Notify municipal authority for urgent repair within 24 hours.",
    "alert_priority": "HIGH"
  },
  "alert": {
    "alert_fired": true,
    "priority": "HIGH"
  },
  "sdg": "SDG 3 & SDG 11 — Road Safety & Sustainable Cities"
}
```

---

## 🐳 Docker

```bash
# Pull from DockerHub
docker pull venkatavamsi01/pothole-detector:latest

# Run
docker run -p 5000:5000 \
  -e GROQ_API_KEY=your_key \
  venkatavamsi01/pothole-detector:latest
```

🔗 DockerHub: https://hub.docker.com/r/venkatavamsi01/pothole-detector

---

## 📁 Project Structure

```
├── app.py               # Flask API + YOLOv8 + Groq LLM agentic loop
├── Dockerfile           # Container definition
├── build_and_run.sh     # Shell script: build / run / push / tag
├── requirements.txt     # Python dependencies
├── best.pt              # YOLOv8 trained weights (download separately)
└── README.md
```

---

## 🌍 SDG Alignment

| SDG | Goal | How This Project Contributes |
|-----|------|------------------------------|
| SDG 3 | Good Health & Well-Being | Prevents road accidents caused by potholes |
| SDG 11 | Sustainable Cities | Enables smart city road monitoring systems |

---

## 🔑 Environment Variables

| Variable         | Required | Description                    |
|------------------|----------|--------------------------------|
| `GROQ_API_KEY`   | Yes      | Groq API key for LLM analysis  |
| `MODEL_PATH`     | No       | Path to YOLOv8 weights (default: `best.pt`) |
| `CONF_THRESHOLD` | No       | Detection confidence (default: `0.40`) |
| `ALERT_WEBHOOK`  | No       | URL to fire HIGH-risk alerts   |

---

## 📜 License

MIT License — Original model by [HussainM899](https://github.com/HussainM899/Pothole-Detection-using-YOLOV8).
LLM integration and deployment by venkatavamsi01.
