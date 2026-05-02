"""
RoadSense AI — CUDA-Enhanced FastAPI Backend
YOLOv8m GPU Inference + Multi-Agent Groq LLM Pipeline
SDG 3 & SDG 11 | SwasthikaSelvakumar

CUDA Features:
- GPU inference with torch.cuda
- FP16 half-precision inference
- GPU memory monitoring
- CUDA-accelerated image preprocessing
- Async FastAPI for concurrent requests
"""

import os, io, json, time, logging, datetime, asyncio
import requests as req_lib
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from ultralytics import YOLO
import torchvision.transforms as T

# ─────────────────────────────────────────────
# CUDA SETUP
# ─────────────────────────────────────────────
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE         = 'cuda:0' if CUDA_AVAILABLE else 'cpu'
USE_FP16       = CUDA_AVAILABLE   # Half precision on GPU only

if CUDA_AVAILABLE:
    torch.backends.cudnn.benchmark = True   # Optimize CUDA kernels
    torch.backends.cudnn.enabled   = True
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEM  = torch.cuda.get_device_properties(0).total_memory / 1e9
else:
    GPU_NAME = 'CPU (no CUDA)'
    GPU_MEM  = 0.0

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama3-8b-8192"
MODEL_PATH     = os.environ.get("MODEL_PATH", "best.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.40"))
IMG_SIZE       = 640

CLASS_LABELS = {0: "pothole", 1: "manhole", 2: "crack"}
RISK_MAP     = {"pothole": "HIGH", "manhole": "HIGH", "crack": "MEDIUM"}

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
# FASTAPI APP
# ─────────────────────────────────────────────
app       = FastAPI(
    title       = "RoadSense AI — CUDA-Enhanced Pothole Detection",
    description = "YOLOv8m GPU inference + Multi-Agent Groq LLM | SDG 3 & SDG 11",
    version     = "2.0.0",
)
templates = Jinja2Templates(directory="templates")
yolo_model = None

# ─────────────────────────────────────────────
# CUDA-ACCELERATED PREPROCESSING
# ─────────────────────────────────────────────
# GPU transform pipeline — runs on CUDA tensor
gpu_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

def preprocess_on_gpu(img: Image.Image) -> torch.Tensor:
    """Move image preprocessing to GPU for speed."""
    tensor = gpu_transform(img.convert("RGB"))
    if CUDA_AVAILABLE:
        tensor = tensor.to(DEVICE)
        if USE_FP16:
            tensor = tensor.half()  # FP16 on GPU
    return tensor.unsqueeze(0)      # Add batch dim

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
def load_yolo():
    global yolo_model
    if yolo_model is None:
        logger.info("Loading YOLOv8m on %s ...", DEVICE)
        yolo_model = YOLO(MODEL_PATH)
        if CUDA_AVAILABLE:
            # Warm up GPU with dummy inference
            logger.info("Warming up CUDA kernels...")
            dummy = Image.new('RGB', (640, 640), color=(128,128,128))
            for _ in range(3):
                yolo_model.predict(
                    source=dummy, device=0,
                    half=USE_FP16, verbose=False
                )
            torch.cuda.synchronize()
            logger.info("GPU warm-up complete.")
        logger.info("Model loaded on %s (FP16=%s)", DEVICE, USE_FP16)
    return yolo_model

# ─────────────────────────────────────────────
# GPU MEMORY MONITOR
# ─────────────────────────────────────────────
def get_gpu_stats() -> dict:
    if not CUDA_AVAILABLE:
        return {"available": False, "device": "CPU"}
    torch.cuda.synchronize()
    return {
        "available"        : True,
        "device_name"      : GPU_NAME,
        "total_memory_gb"  : round(GPU_MEM, 2),
        "allocated_mb"     : round(torch.cuda.memory_allocated(0) / 1e6, 1),
        "reserved_mb"      : round(torch.cuda.memory_reserved(0)  / 1e6, 1),
        "free_mb"          : round((torch.cuda.get_device_properties(0).total_memory
                                    - torch.cuda.memory_allocated(0)) / 1e6, 1),
        "fp16_enabled"     : USE_FP16,
        "cudnn_benchmark"  : torch.backends.cudnn.benchmark,
    }

# ─────────────────────────────────────────────
# AREA SEVERITY
# ─────────────────────────────────────────────
def area_severity(box_area: float, img_area: float) -> str:
    r = box_area / img_area if img_area > 0 else 0
    return "LARGE" if r > 0.10 else ("MEDIUM" if r > 0.03 else "SMALL")

# ─────────────────────────────────────────────
# MULTI-AGENT GROQ LLM PIPELINE
# ─────────────────────────────────────────────
def call_groq(prompt: str, max_tokens: int = 300) -> str:
    """Single Groq API call."""
    if not GROQ_API_KEY:
        return "{}"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    try:
        r = req_lib.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("Groq call failed: %s", e)
        return "{}"


async def run_multi_agent_pipeline(detections: list) -> dict:
    """
    3-Agent LLM Pipeline — runs concurrently using asyncio.
    Agent 1: Damage Assessment Agent
    Agent 2: Risk Scoring Agent
    Agent 3: Corrective Action Agent
    Supervisor: Synthesizes all 3 outputs
    """
    det_summary = "\n".join([
        f"- {d['class'].upper()} | size: {d['size']} | confidence: {d['confidence']*100:.1f}%"
        for d in detections
    ]) if detections else "No defects detected."

    # ── Agent 1: Damage Assessment ─────────────
    agent1_prompt = f"""
You are a Road Damage Assessment Agent.
Detected road defects:
{det_summary}

Respond ONLY in JSON (no markdown):
{{"damage_type": "<type of damage>", "affected_area": "<estimated area>", "damage_severity": "<MINOR|MODERATE|SEVERE>"}}
"""

    # ── Agent 2: Risk Scoring ──────────────────
    agent2_prompt = f"""
You are a Road Safety Risk Scoring Agent.
Detected road defects:
{det_summary}

Respond ONLY in JSON (no markdown):
{{"risk_score": <1-10>, "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>", "accident_probability": "<LOW|MEDIUM|HIGH>"}}
"""

    # ── Agent 3: Corrective Action ─────────────
    agent3_prompt = f"""
You are a Municipal Road Maintenance Action Agent.
Detected road defects:
{det_summary}

Respond ONLY in JSON (no markdown):
{{"immediate_action": "<action within 24hrs>", "long_term_fix": "<permanent solution>", "estimated_cost": "<LOW|MEDIUM|HIGH>", "priority": "<P1|P2|P3>"}}
"""

    # Run all 3 agents concurrently
    loop = asyncio.get_event_loop()
    a1, a2, a3 = await asyncio.gather(
        loop.run_in_executor(None, call_groq, agent1_prompt, 200),
        loop.run_in_executor(None, call_groq, agent2_prompt, 200),
        loop.run_in_executor(None, call_groq, agent3_prompt, 200),
    )

    # Parse agent outputs
    def safe_parse(raw):
        try:
            return json.loads(raw)
        except:
            return {}

    agent1_out = safe_parse(a1)
    agent2_out = safe_parse(a2)
    agent3_out = safe_parse(a3)

    # ── Supervisor Agent: Synthesize ───────────
    supervisor_prompt = f"""
You are a Road Safety Supervisor AI that synthesizes reports from 3 specialist agents.

Damage Assessment Agent report:
{json.dumps(agent1_out)}

Risk Scoring Agent report:
{json.dumps(agent2_out)}

Corrective Action Agent report:
{json.dumps(agent3_out)}

Respond ONLY in JSON (no markdown):
{{
  "executive_summary": "<2 sentence overall assessment>",
  "final_risk_level"  : "<LOW|MEDIUM|HIGH|CRITICAL>",
  "final_severity_score": <1-10>,
  "top_action"        : "<single most important action to take now>"
}}
"""

    supervisor_raw = await loop.run_in_executor(None, call_groq, supervisor_prompt, 300)
    supervisor_out = safe_parse(supervisor_raw)

    # Fallback if LLM fails
    if not supervisor_out:
        has_high = any(RISK_MAP.get(d["class"], "LOW") == "HIGH" for d in detections)
        supervisor_out = {
            "executive_summary"   : f"Detected {len(detections)} road defect(s)." if detections else "Road appears clear.",
            "final_risk_level"    : "HIGH" if has_high else ("MEDIUM" if detections else "LOW"),
            "final_severity_score": 7 if has_high else (4 if detections else 1),
            "top_action"          : "Notify municipal authority for urgent repair." if has_high else "Schedule routine inspection.",
        }

    return {
        "agent_1_damage"    : agent1_out,
        "agent_2_risk"      : agent2_out,
        "agent_3_action"    : agent3_out,
        "supervisor_summary": supervisor_out,
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {
        "status" : "ok",
        "version": "2.0.0",
        "model"  : MODEL_PATH,
        "cuda"   : get_gpu_stats(),
        "sdg"    : ["SDG 3 — Good Health and Well-Being",
                    "SDG 11 — Sustainable Cities and Communities"],
    }


@app.get("/gpu")
async def gpu_info():
    """Dedicated GPU stats endpoint."""
    return get_gpu_stats()


@app.get("/classes")
async def classes():
    return {"classes": CLASS_LABELS, "risk_map": RISK_MAP}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    CUDA-accelerated pothole detection + multi-agent LLM analysis.
    """
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    raw      = await image.read()
    filename = image.filename or "uploaded.jpg"

    if not raw:
        raise HTTPException(400, "Empty image file.")

    # ── 1. CUDA Inference ─────────────────────
    clf = load_yolo()

    t_inference_start = time.perf_counter()
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    try:
        img      = Image.open(io.BytesIO(raw)).convert("RGB")
        img_w, img_h = img.size
        img_area = img_w * img_h

        results  = clf.predict(
            source  = img,
            conf    = CONF_THRESHOLD,
            device  = 0 if CUDA_AVAILABLE else 'cpu',
            half    = USE_FP16,
            verbose = False,
        )
        boxes = results[0].boxes

    except Exception as e:
        logger.error("Inference error: %s", e)
        raise HTTPException(500, f"Inference failed: {str(e)}")

    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - t_inference_start) * 1000

    # ── 2. Parse detections ───────────────────
    detections = []
    if boxes is not None and len(boxes):
        for box in boxes:
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cls_name = CLASS_LABELS.get(cls_id, f"class_{cls_id}")
            detections.append({
                "class"     : cls_name,
                "confidence": round(conf, 4),
                "risk"      : RISK_MAP.get(cls_name, "LOW"),
                "size"      : area_severity((x2-x1)*(y2-y1), img_area),
                "bbox"      : [round(x1), round(y1), round(x2), round(y2)],
            })
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # ── 3. Multi-Agent LLM ────────────────────
    t_llm_start = time.perf_counter()
    llm_result  = await run_multi_agent_pipeline(detections)
    llm_ms      = (time.perf_counter() - t_llm_start) * 1000

    # ── 4. Agentic Alert ──────────────────────
    final_priority = llm_result["supervisor_summary"].get("final_risk_level", "LOW")
    alert_fired    = final_priority in ("HIGH", "CRITICAL")

    alert_record = {
        "timestamp"     : datetime.datetime.utcnow().isoformat() + "Z",
        "image"         : filename,
        "detections"    : detections,
        "final_priority": final_priority,
        "severity_score": llm_result["supervisor_summary"].get("final_severity_score"),
        "top_action"    : llm_result["supervisor_summary"].get("top_action"),
    }
    if alert_fired:
        logger.warning("🚨 ALERT [%s] | %s", final_priority, json.dumps(alert_record))

    # ── 5. Response ───────────────────────────
    total_ms = inference_ms + llm_ms

    response = {
        "image"            : filename,
        "total_defects_found": len(detections),
        "detections"       : detections,
        "multi_agent_llm"  : llm_result,
        "llm_analysis"     : {
            "explanation"      : llm_result["supervisor_summary"].get("executive_summary", ""),
            "severity_score"   : llm_result["supervisor_summary"].get("final_severity_score", 0),
            "corrective_action": llm_result["supervisor_summary"].get("top_action", ""),
            "alert_priority"   : llm_result["supervisor_summary"].get("final_risk_level", "LOW"),
        },
        "alert"            : {
            "alert_fired": alert_fired,
            "priority"   : final_priority,
        },
        "performance"      : {
            "inference_ms"  : round(inference_ms, 2),
            "llm_ms"        : round(llm_ms, 2),
            "total_ms"      : round(total_ms, 2),
            "device"        : DEVICE,
            "fp16_used"     : USE_FP16,
            "gpu_mem_used_mb": round(torch.cuda.memory_allocated(0)/1e6, 1)
                               if CUDA_AVAILABLE else 0,
        },
        "sdg": "SDG 3 & SDG 11 — Road Safety & Sustainable Cities",
    }

    logger.info("Processed: %s | Defects: %d | Priority: %s | "
                "Inference: %.1fms | LLM: %.1fms | Device: %s",
                filename, len(detections), final_priority,
                inference_ms, llm_ms, DEVICE)

    return response


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    logger.info("RoadSense AI v2.0 starting on port %d", port)
    logger.info("CUDA available: %s | Device: %s | FP16: %s",
                CUDA_AVAILABLE, DEVICE, USE_FP16)
    uvicorn.run("app:app", host="0.0.0.0", port=port,
                reload=False, workers=1)
