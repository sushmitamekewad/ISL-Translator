import sys
import os
import json
import base64
from typing import Optional

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

try:
    from src.core.infer import ISLInferenceEngine
    from src.core.aggregator import SignAggregator
    from src.models.factory import get_model_specs
except ModuleNotFoundError:
    from core.infer import ISLInferenceEngine
    from core.aggregator import SignAggregator
    from models.factory import get_model_specs

# Initialize FastAPI app
app = FastAPI(
    title="Indian Sign Language (ISL) Translator API",
    description="Real-Time Letter-by-Letter ISL Recognition, Word Formation, and Sentence Translation API with Cloud & Offline Support",
    version="1.0.0"
)

# CORS configuration for cloud and local deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment and default parameters
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mobilenet_v3_small")
CONFIDENCE_THRESH = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
CONSECUTIVE_FRAMES = int(os.getenv("CONSECUTIVE_FRAMES_THRESHOLD", "8"))
SPACE_TIMEOUT = float(os.getenv("SPACE_TIMEOUT_SECONDS", "2.0"))

# Global Engine & Aggregator instances
inference_engine = ISLInferenceEngine(model_name=DEFAULT_MODEL)
aggregator = SignAggregator(
    confidence_threshold=CONFIDENCE_THRESH,
    consecutive_frames_threshold=CONSECUTIVE_FRAMES,
    space_timeout_seconds=SPACE_TIMEOUT
)

# Request schemas
class FramePayload(BaseModel):
    image_base64: str

class ActionPayload(BaseModel):
    action: str  # 'backspace' | 'space' | 'clear' | 'select_suggestion'
    value: Optional[str] = None

class ModelSwitchPayload(BaseModel):
    model_name: str  # 'mobilenet_v3_small' | 'efficientnet_lite0' | 'yolo11n'


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active_model": inference_engine.model_name,
        "device": inference_engine.device,
        "engine": "onnx" if inference_engine.use_onnx else "pytorch",
        "classes_count": len(inference_engine.classes)
    }


@app.get("/models/info")
def get_models_metadata():
    """Returns architecture specs and comparative trade-offs for candidate models."""
    return {
        "active_model": inference_engine.model_name,
        "specs": get_model_specs()
    }


@app.post("/models/switch")
def switch_model(payload: ModelSwitchPayload):
    """Dynamically switch the active neural network backend."""
    global inference_engine
    try:
        inference_engine = ISLInferenceEngine(model_name=payload.model_name)
        return {
            "success": True,
            "message": f"Successfully switched backend to {payload.model_name}",
            "device": inference_engine.device
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/image")
async def predict_single_image(file: UploadFile = File(...)):
    """Receives an image file and returns letter classification and probabilities."""
    contents = await file.read()
    res = inference_engine.predict(contents)
    return res


@app.post("/stream/frame")
def process_stream_frame(payload: FramePayload):
    """
    Main real-time endpoint for webcam frames.
    Classifies sign and updates temporal word/sentence aggregator state.
    """
    pred_res = inference_engine.predict(payload.image_base64)
    letter = pred_res["predicted_letter"]
    conf = pred_res["confidence"]

    agg_res = aggregator.process_frame_prediction(letter=letter, confidence=conf)

    return {
        "inference": pred_res,
        "aggregation": agg_res
    }


@app.post("/aggregator/action")
def execute_aggregator_action(payload: ActionPayload):
    """Control word and sentence buffers (backspace, space, clear, accept suggestion)."""
    act = payload.action.lower()
    if act == "backspace":
        aggregator.backspace()
    elif act == "space":
        aggregator.commit_current_word()
    elif act == "clear":
        aggregator.clear()
    elif act == "select_suggestion" and payload.value:
        aggregator.select_suggestion(payload.value)
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {payload.action}")

    return {
        "current_word": aggregator.current_word,
        "sentence_words": aggregator.sentence_words,
        "full_sentence": aggregator.get_full_sentence(),
        "suggestions": aggregator.get_word_suggestions()
    }


@app.websocket("/ws/stream")
async def websocket_video_stream(websocket: WebSocket):
    """
    High-frequency duplex WebSocket connection for webcam streaming.
    Client continuously sends raw/base64 frames; server yields real-time translations.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            frame_b64 = message.get("frame")
            if not frame_b64:
                continue

            pred_res = inference_engine.predict(frame_b64)
            agg_res = aggregator.process_frame_prediction(
                letter=pred_res["predicted_letter"],
                confidence=pred_res["confidence"]
            )

            response = {
                "letter": pred_res["predicted_letter"],
                "confidence": pred_res["confidence"],
                "latency_ms": pred_res["latency_ms"],
                "top_3": pred_res["top_3"],
                "state": agg_res
            }
            await websocket.send_text(json.dumps(response))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Error]: {e}")


# Static files mount
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = os.path.join("static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return "<h1>ISL Translator API is running. Visit /docs for Swagger UI.</h1>"
