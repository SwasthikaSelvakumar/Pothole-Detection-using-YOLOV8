# 🛣️ RoadSense AI — CUDA-Accelerated Intelligent Road Defect Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688?style=for-the-badge&logo=fastapi)
![YOLOv8](https://img.shields.io/badge/YOLOv8m-Ultralytics-FF6B35?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-T4_GPU-76B900?style=for-the-badge&logo=nvidia)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**SDG 3 — Good Health & Well-Being &nbsp;|&nbsp; SDG 11 — Sustainable Cities & Communities**

[🔍 Live Demo](#-quick-start) • [📄 Paper](#-paper) • [🐳 DockerHub](https://hub.docker.com/r/swaszz/pothole-detector) • [📊 API Docs](#-api-endpoints)

</div>

---

## 📌 Overview

**RoadSense AI** is a GPU-accelerated intelligent road defect detection system that combines a fine-tuned **YOLOv8m** object detection model with a novel **three-agent Large Language Model (LLM) pipeline** powered by the Groq inference engine. The system detects potholes, cracks, and manholes in road images and automatically generates structured safety assessments, risk scores, and corrective action recommendations.

The system is deployed as a **FastAPI REST service** with an interactive three-tab web dashboard, GPS incident logging, and a real-time analytics dashboard — all containerized via Docker and deployed through GitHub Actions CI/CD.

---

## 🏗️ System Architecture

```
Road Image Input
       ↓
┌─────────────────────────────────────────┐
│   YOLOv8m (CUDA T4 GPU, FP16 AMP)      │  ← Object Detection
│   mAP@50: 70.18% | 16.5x GPU Speedup   │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│         Multi-Agent LLM Pipeline        │
│  Agent 1 → Damage Assessment            │
│  Agent 2 → Risk Scoring (1–10)          │  ← Groq LLaMA3-8B
│  Agent 3 → Corrective Action            │
│  Supervisor → Final Synthesis           │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│   FastAPI REST API + Web Dashboard      │  ← Deployment
│   GPS Logging | Analytics | Alerts      │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│   Docker Container → DockerHub          │  ← CI/CD via GitHub Actions
└─────────────────────────────────────────┘
```

---

## ✨ Key Features

- 🚀 **CUDA GPU Acceleration** — Trained on NVIDIA T4 with FP16 mixed precision (16.5x faster than CPU)
- 🤖 **3-Agent LLM Pipeline** — Concurrent damage assessment, risk scoring, and corrective action agents
- 🗺️ **GPS Incident Logging** — Location-tagged detections with OpenStreetMap integration and CSV export
- 📊 **Real-time Dashboard** — Defect distribution charts, severity trends, priority breakdown
- 🔌 **FastAPI REST API** — Auto-generated Swagger UI at `/docs`
- 🐳 **Docker Containerized** — One-command deployment
- 🌍 **SDG Aligned** — SDG 3 (Road Safety) + SDG 11 (Smart Cities)

---

## 📁 Project Structure

```
Pothole-Detection-using-YOLOV8/
│
├── app.py                        # FastAPI backend + CUDA inference + 3-agent LLM
├── Dockerfile                    # Lightweight container definition
├── build_and_run.sh              # Shell script: build / run / push
├── requirements.txt              # Python dependencies
├── benchmark.py                  # CPU vs GPU benchmark script
├── best.pt                       # Fine-tuned YOLOv8m weights (CUDA T4)
├── train_colab.ipynb             # Google Colab GPU training notebook
│
├── templates/
│   └── index.html                # Full web UI (3 tabs: Detect, Incidents, Dashboard)
│
├── .github/
│   └── workflows/
│       └── docker.yml            # GitHub Actions CI/CD → DockerHub
│
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — Run with Python (Local)

```bash
# 1. Clone the repo
git clone https://github.com/SwasthikaSelvakumar/Pothole-Detection-using-YOLOV8.git
cd Pothole-Detection-using-YOLOV8

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY=your_key_here    # Linux/Mac
$env:GROQ_API_KEY="your_key_here"   # Windows

# 5. Run the app
python app.py
```

Open 👉 http://localhost:5000

### Option 2 — Run with Docker

```bash
docker pull swaszz/pothole-detector:latest

docker run -p 5000:5000 \
  -e GROQ_API_KEY=your_key_here \
  swaszz/pothole-detector:latest
```

### Option 3 — Build and Run with Shell Script

```bash
chmod +x build_and_run.sh

export GROQ_API_KEY=your_key_here

./build_and_run.sh all      # build + run + push
./build_and_run.sh build    # build only
./build_and_run.sh run      # run only
./build_and_run.sh push     # push to DockerHub
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI dashboard |
| `GET` | `/health` | Health check + CUDA stats |
| `GET` | `/gpu` | Live GPU memory monitoring |
| `GET` | `/classes` | Detection classes + risk map |
| `POST` | `/predict` | Full inference pipeline |
| `GET` | `/docs` | Auto-generated Swagger UI |

### Example Request

```bash
curl -X POST http://localhost:5000/predict \
     -F "image=@road_photo.jpg"
```

### Example Response

```json
{
  "image": "road_photo.jpg",
  "total_defects_found": 1,
  "detections": [
    {
      "class": "pothole",
      "confidence": 0.8912,
      "risk": "HIGH",
      "size": "LARGE",
      "bbox": [120, 340, 280, 480]
    }
  ],
  "llm_analysis": {
    "explanation": "A large pothole detected at road center poses high collision risk.",
    "severity_score": 8,
    "corrective_action": "Notify municipal authority for urgent repair within 24 hours.",
    "alert_priority": "HIGH"
  },
  "multi_agent_llm": {
    "agent_1_damage": { "damage_type": "pothole", "damage_severity": "SEVERE" },
    "agent_2_risk":   { "risk_score": 8, "accident_probability": "HIGH" },
    "agent_3_action": { "immediate_action": "Place warning signs", "priority": "P1" },
    "supervisor_summary": { "final_risk_level": "HIGH", "final_severity_score": 8 }
  },
  "performance": {
    "inference_ms": 193.2,
    "llm_ms": 520.0,
    "total_ms": 713.2,
    "device": "cpu"
  }
}
```

---

## 📊 Model Performance

### Detection Metrics (YOLOv8m)

| Metric | Value |
|--------|-------|
| mAP@50 | 70.18% |
| mAP@50-95 | 38.45% |
| Precision | 0.7750 |
| Recall | 0.5895 |
| F1-Score | 0.6696 |

### CUDA Benchmark (NVIDIA T4 GPU, 50 runs)

| Configuration | Mean Latency | FPS | Speedup |
|--------------|-------------|-----|---------|
| CPU (baseline) | 842.3 ms | 1.2 | 1.0x |
| GPU FP32 | 68.4 ms | 14.6 | 12.3x |
| GPU FP16 (AMP) | 51.2 ms | 19.5 | **16.5x** |

### Comparison with Previous Methods

| Method | mAP@50 | GPU | LLM | Deployment |
|--------|--------|-----|-----|-----------|
| Haar Cascade (2018) | 41.2% | ❌ | ❌ | None |
| CNN + SVM (2019) | 53.7% | ❌ | ❌ | None |
| VGG16 Transfer (2020) | 58.4% | ❌ | ❌ | None |
| ResNet50 (2021) | 63.1% | ❌ | ❌ | None |
| YOLOv5s CPU (2022) | 65.8% | ❌ | ❌ | Flask |
| YOLOv8n CPU (2023) | 68.3% | ❌ | ❌ | Flask |
| **RoadSense AI (Ours)** | **70.2%** | **✅ T4** | **✅ 3-Agent** | **Docker+FastAPI** |

---

## 🧠 Multi-Agent LLM Architecture

```
Detection Result
      ↓
┌─────────────────────────────────────────────────┐
│              asyncio.gather() — Concurrent       │
│                                                  │
│  [Agent 1]          [Agent 2]        [Agent 3]  │
│  Damage             Risk             Corrective  │
│  Assessment         Scoring          Action      │
│  - Damage type      - Risk score     - 24hr fix  │
│  - Severity         - Probability    - Long-term │
│  - Affected area    - Risk level     - Cost est. │
└─────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────┐
│              Supervisor Agent                    │
│  Synthesizes all 3 → Executive summary          │
│  Final risk level + severity score + top action │
└─────────────────────────────────────────────────┘
```

---

## 🌍 SDG Alignment

| SDG | Goal | Contribution |
|-----|------|-------------|
| **SDG 3** | Good Health & Well-Being | Prevents road accidents by detecting hazards proactively |
| **SDG 11** | Sustainable Cities | Enables smart city road monitoring with GPS + dashboard |

---

## 🔧 GPU Training (Google Colab)

The model was trained on **NVIDIA T4 GPU** using Google Colab with:
- **FP16 Mixed Precision** (`amp=True`) — 40% memory reduction
- **cuDNN Benchmark Mode** — Auto-optimized CUDA kernels
- **AdamW Optimizer** with Cosine LR scheduling
- **80 epochs**, batch size 16, imgsz 640

To retrain:
1. Open `train_colab.ipynb` in Google Colab
2. Set Runtime → T4 GPU
3. Run all cells — downloads dataset, trains, benchmarks, exports

---

## 🐳 Docker & CI/CD

The project uses **GitHub Actions** to automatically build and push the Docker image to DockerHub on every push to `main`:

```yaml
# .github/workflows/docker.yml
on:
  push:
    branches: [main]
```

DockerHub: 👉 https://hub.docker.com/r/swaszz/pothole-detector

---

## 📄 Paper

This project is submitted to **Software Impacts (Elsevier)** journal.

**Title:** RoadSense AI: A CUDA-Accelerated Intelligent Road Defect Detection System with Multi-Agent LLM Pipeline for Smart City Safety

**Authors:** Swasthika S

**Institution:** SRM Institute of Science and Technology, Tiruchirappalli

---

## 🗂️ Version History

| Version | Description |
|---------|-------------|
| v1.0.0 | Initial Flask API + YOLOv8 + Groq LLM |
| v2.0.0 | CUDA GPU training + FastAPI + 3-agent LLM + Web UI + Docker |

---

## 📜 License

MIT License — Original YOLOv8 pothole detection base by [HussainM899](https://github.com/HussainM899/Pothole-Detection-using-YOLOV8). LLM integration, CUDA training, multi-agent pipeline, and deployment by SwasthikaSelvakumar.

---

<div align="center">
Made with ❤️ for SDG 3 & SDG 11 
</div>
