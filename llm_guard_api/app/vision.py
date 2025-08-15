import os
from typing import Dict, List, Optional

import numpy as np

_ocr_models: Dict[str, object] = {}
_yolo_model: Optional[object] = None
_yolo_confidence_threshold: float = 0.25


def initialize_vision_models(
    *,
    languages: List[str] = None,
    yolo_onnx_path: Optional[str] = None,
    yolo_confidence_threshold: float = 0.25,
) -> None:
    global _ocr_models, _yolo_model, _yolo_confidence_threshold

    if languages is None:
        languages = ["en", "ru"]

    # Init OCR models (ONNXRuntime)
    for lang in languages:
        if lang in _ocr_models:
            continue
        try:
            from paddleocr import PaddleOCR  # type: ignore

            _ocr_models[lang] = PaddleOCR(
                use_angle_cls=True,
                use_onnx=True,
                lang=lang,
            )
            # Trigger model warmup/download
            _ = _ocr_models[lang].ocr(np.zeros((32, 128, 3), dtype=np.uint8), cls=True)
        except Exception as exc:
            # Leave missing language silently; will be handled at call time
            continue

    # Init YOLO ONNX if provided
    _yolo_confidence_threshold = float(yolo_confidence_threshold)
    if yolo_onnx_path and _yolo_model is None:
        try:
            from ultralytics import YOLO  # type: ignore

            _yolo_model = YOLO(yolo_onnx_path)
        except Exception:
            _yolo_model = None


def get_ocr_models() -> List[object]:
    # Return all loaded OCR models
    return list(_ocr_models.values())


def get_yolo_model():
    return _yolo_model


def get_yolo_confidence_threshold() -> float:
    return _yolo_confidence_threshold