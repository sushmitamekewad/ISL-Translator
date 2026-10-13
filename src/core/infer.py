import sys
import os
import io
import time
import base64
from typing import Dict, Any, List, Optional, Union
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torchvision.transforms as transforms
try:
    from src.models.factory import create_model
except ModuleNotFoundError:
    from models.factory import create_model

# Default 26 alphabet classes for ISL
DEFAULT_CLASSES = [chr(ord('a') + i) for i in range(26)]

class ISLInferenceEngine:
    """
    Unified Inference Engine for Indian Sign Language.
    Supports both PyTorch and ONNX Runtime execution for cloud & offline deployment.
    """
    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        weights_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        classes: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.classes = classes or DEFAULT_CLASSES
        self.onnx_path = onnx_path
        self.use_onnx = onnx_path is not None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.pytorch_model = None
        self.ort_session = None

        # Auto-detect trained checkpoint if not provided
        if not self.use_onnx and weights_path is None:
            candidate_checkpoint = os.path.join(PROJECT_ROOT, "models", "checkpoints", f"{model_name}_best.pth")
            if os.path.exists(candidate_checkpoint):
                weights_path = candidate_checkpoint
                print(f"[InferenceEngine] Automatically detected trained checkpoint: {weights_path}")

        if self.use_onnx:
            self._init_onnx(onnx_path)
        else:
            self._init_pytorch(model_name, weights_path)

    def _init_pytorch(self, model_name: str, weights_path: Optional[str]):
        print(f"[InferenceEngine] Initializing PyTorch model: {model_name} on {self.device}...")
        self.pytorch_model = create_model(model_name, num_classes=len(self.classes), pretrained=(weights_path is None))
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.pytorch_model.load_state_dict(state.get("state_dict", state))
            print(f"[InferenceEngine] Loaded trained weights from: {weights_path}")
        self.pytorch_model.to(self.device)
        self.pytorch_model.eval()

    def _init_onnx(self, onnx_path: str):
        import onnxruntime as ort
        print(f"[InferenceEngine] Initializing ONNX Runtime session: {onnx_path}...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name

    def preprocess_image(self, image_input: Union[Image.Image, bytes, str, np.ndarray]) -> torch.Tensor:
        """Accepts PIL image, raw bytes, base64 string, or OpenCV numpy array."""
        if isinstance(image_input, str):
            # Base64 string
            if "base64," in image_input:
                image_input = image_input.split("base64,")[1]
            image_bytes = base64.b64decode(image_input)
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input type")

        tensor = self.transform(pil_img).unsqueeze(0)
        return tensor

    def predict(self, image_input: Union[Image.Image, bytes, str, np.ndarray]) -> Dict[str, Any]:
        """
        Runs inference and returns:
          - predicted_letter (str)
          - confidence (float)
          - top_3 (list of dicts)
          - latency_ms (float)
        """
        start_time = time.perf_counter()
        img_tensor = self.preprocess_image(image_input)

        if self.use_onnx:
            np_in = img_tensor.numpy().astype(np.float32)
            logits = self.ort_session.run(None, {self.input_name: np_in})[0]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            probs = probs[0]
        else:
            img_tensor = img_tensor.to(self.device)
            with torch.no_grad():
                logits = self.pytorch_model(img_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        top_indices = np.argsort(probs)[::-1][:3]
        top_3 = [
            {"label": self.classes[idx].upper(), "confidence": float(probs[idx])}
            for idx in top_indices
        ]

        best_label = top_3[0]["label"]
        best_conf = top_3[0]["confidence"]

        return {
            "predicted_letter": best_label,
            "confidence": best_conf,
            "top_3": top_3,
            "latency_ms": round(latency_ms, 2),
            "device": self.device,
            "engine": "onnx" if self.use_onnx else f"pytorch ({self.model_name})"
        }
