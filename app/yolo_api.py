
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
import numpy as np
import cv2
import io, os, time, logging
import torch

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# ────────────────────────────────────────────
# Config
MODEL_PATH      = "runs/detect/best_model.pt"
MAX_WAIT_TIME   = 600   # 10 phút
WAIT_INTERVAL   = 10    # Kiểm tra mỗi 10 s
STATIC_DIR      = "static"

# ────────────────────────────────────────────
# FastAPI + Metrics
app = FastAPI(title="YOLOv8 Inference API with Prometheus")

Instrumentator().instrument(app).expose(app)      # /metrics

REQUEST_TOTAL   = Counter("api_requests_total",       "Tổng request tới /predict")
REQUEST_ERRORS  = Counter("api_request_errors_total", "Tổng lỗi xảy ra ở /predict")

INFER_SEC       = Histogram("inference_seconds",       "Thời gian inference (CPU)")
GPU_INFER_SEC   = Histogram(
    "inference_gpu_seconds", "Thời gian inference (GPU)",
    buckets=(.001, .005, .01, .05, .1, .5, 1, 2)
)
CONFIDENCE_HIST = Histogram(
    "predict_confidence", "Độ tin cậy bbox đầu tiên",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# ────────────────────────────────────────────
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log")
    ],
)
log = logging.getLogger(__name__)

# ────────────────────────────────────────────
# Helper: chờ file model
def wait_for_model(path: str):
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > MAX_WAIT_TIME:
            raise RuntimeError(f"Timeout waiting for {path} after {MAX_WAIT_TIME} s")
        log.info("Waiting for %s … retry in %ss", path, WAIT_INTERVAL)
        time.sleep(WAIT_INTERVAL)
    log.info("Model %s is ready", path)

wait_for_model(MODEL_PATH)
model = YOLO(MODEL_PATH)

# ────────────────────────────────────────────
# Static files
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    log.warning("Static directory '%s' not found; skipping mount.", STATIC_DIR)

# ────────────────────────────────────────────
def draw_boxes(img: np.ndarray, res) -> np.ndarray:
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf           = float(box.conf[0])
        cls            = int(box.cls[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.putText(
            img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return img

# ────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        return "<h1>YOLOv8 Inference Service</h1><p>Upload an image to /predict</p>"
    with open(index_file, encoding="utf-8") as f:
        return f.read()

# ────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    REQUEST_TOTAL.inc()

    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("File gửi lên không phải hình ảnh hợp lệ")

        # ------------------ Inference timing ------------------
        cpu_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_start = time.perf_counter()

        results = model.predict(image, save=False)[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            GPU_INFER_SEC.observe(time.perf_counter() - gpu_start)

        INFER_SEC.observe(time.perf_counter() - cpu_start)

        if results.boxes.conf.numel():
            CONFIDENCE_HIST.observe(float(results.boxes.conf[0]))

        # ------------------ Trả kết quả -----------------------
        image_out = draw_boxes(image.copy(), results)
        ok, buf = cv2.imencode(".jpg", image_out)
        if not ok:
            raise RuntimeError("Lỗi encode ảnh đầu ra")
        return StreamingResponse(io.BytesIO(buf.tobytes()),
                                 media_type="image/jpeg")

    except Exception as exc:
        REQUEST_ERRORS.inc()
        log.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc