import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore


class DependencyError(RuntimeError):
    pass


# BGR colors for OpenCV
PATTERN_STYLES: Dict[str, Dict[str, Any]] = {
    "CREDIT_CARD_RE": {"label": "Credit Card"},
    "EMAIL_ADDRESS_RE": {"label": "Email"},
    "US_SSN_RE": {"label": "SSN"},
    "PHONE_NUMBER_WITH_EXT": {"label": "Phone"},
    "PHONE_NUMBER_ZH": {"label": "Phone"},
    "UUID": {"label": "UUID"},
}

DEFAULT_STYLE = {"label": "Sensitive"}

# Colors (BGR)
MASK_FILL_COLOR = (0, 0, 255)         # red mask fill
LABEL_BG_COLOR = (0, 0, 255)          # red label bar background
LABEL_TEXT_COLOR = (255, 255, 255)    # white label text
LABEL_SHADOW_COLOR = (0, 0, 0)        # black shadow for text


class ConfidentialImageScanner:
    def __init__(
        self,
        *,
        patterns: Optional[List[str]] = None,
        redact_mode: str = "partial",
        yolo_model_path: Optional[str] = None,
        yolo_confidence_threshold: float = 0.25,
    ) -> None:
        self.redact_mode = redact_mode
        self.patterns = self._compile_patterns(patterns)
        self.yolo_model = None
        self.yolo_confidence_threshold = yolo_confidence_threshold

        # Models are initialized globally in app.vision
        self._paddle_ocr = None
        self.yolo_model = None

    def _ensure_cv2(self) -> None:
        if cv2 is None:
            raise DependencyError(
                "OpenCV is required. Install 'opencv-python-headless' to enable image scanning."
            )

    def _get_ocr_models(self):
        try:
            from .vision import get_ocr_models

            models = get_ocr_models()
            if not models:
                raise DependencyError("PaddleOCR models are not initialized")
            return models
        except Exception as exc:
            raise DependencyError("PaddleOCR is required and must be initialized") from exc

    @staticmethod
    def _compile_patterns(patterns: Optional[List[str]]):
        import re

        compiled: List[Tuple[str, Any]] = []
        source_patterns: List[Tuple[str, str]] = []

        if patterns and len(patterns) > 0:
            for idx, expr in enumerate(patterns):
                source_patterns.append((f"CUSTOM_{idx}", expr))
        else:
            # Fallback to a useful subset of built-in sensitive patterns
            try:
                from llm_guard.input_scanners.anonymize_helpers.regex_patterns import (
                    DEFAULT_REGEX_PATTERNS,
                )

                for pat in DEFAULT_REGEX_PATTERNS:
                    if "expressions" in pat and "name" in pat:
                        # Only include high-signal confidential types
                        if pat.get("name") in {
                            "CREDIT_CARD_RE",
                            "EMAIL_ADDRESS_RE",
                            "US_SSN_RE",
                            "PHONE_NUMBER_WITH_EXT",
                            "PHONE_NUMBER_ZH",
                            "UUID",
                        }:
                            for expr in pat["expressions"]:
                                source_patterns.append((str(pat["name"]), str(expr)))
            except Exception:
                # Minimal safe defaults if import fails
                source_patterns.extend(
                    [
                        ("CREDIT_CARD_RE", r"(?:(4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})|(3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5})|(3(?:0[0-5]|[68]\d)\d{11}))"),
                        ("EMAIL_ADDRESS_RE", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
                        ("US_SSN_RE", r"\b\d{3}-\d{2}-\d{4}\b"),
                        ("UUID", r"[a-f0-9]{8}\-[a-f0-9]{4}\-[a-f0-9]{4}\-[a-f0-9]{4}\-[a-f0-9]{12}"),
                    ]
                )

        for name, expr in source_patterns:
            try:
                compiled.append((name, re.compile(expr)))
            except Exception:
                # Skip invalid regex
                continue
        return compiled

    @staticmethod
    def _bytes_to_image(image_bytes: bytes) -> np.ndarray:
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # type: ignore
        if img is None:
            raise ValueError("Unable to decode image bytes")
        return img

    @staticmethod
    def _image_to_base64(img: np.ndarray, format: str = "PNG") -> str:
        import PIL.Image  # type: ignore

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type: ignore
        pil_img = PIL.Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _expand_box(xyxy: Tuple[int, int, int, int], w: int, h: int, pad: int = 8) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = xyxy
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
        return x1, y1, x2, y2

    def _detect_candidates_yolo(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            from .vision import get_yolo_model, get_yolo_confidence_threshold

            model = get_yolo_model()
            if model is None:
                return []
            conf = get_yolo_confidence_threshold()
            results = model.predict(img, conf=conf, verbose=False)
        except Exception:
            return []

        boxes: List[Tuple[int, int, int, int]] = []
        h, w = img.shape[:2]
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            for b in r.boxes:
                # b.xyxy is a tensor [[x1,y1,x2,y2]]
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                boxes.append(self._expand_box((x1, y1, x2, y2), w, h, pad=12))
        return boxes

    def _detect_and_recognize(self, img: np.ndarray) -> List[Tuple[List[Tuple[int, int]], str, float]]:
        ocr_models = self._get_ocr_models()
        h, w = img.shape[:2]

        # Candidate crops from YOLO to guide OCR
        candidate_boxes = self._detect_candidates_yolo(img)
        results: List[Tuple[List[Tuple[int, int]], str, float]] = []

        def run_ocr(region: np.ndarray, x_offset: int = 0, y_offset: int = 0):
            for ocr in ocr_models:
                ocr_result = ocr.ocr(region, cls=True)
                if not ocr_result:
                    continue
                for line in ocr_result[0]:
                    pts = line[0]
                    txt = line[1][0]
                    conf = float(line[1][1])
                    # pts: 4 points [[x,y],...]
                    quad = [(int(p[0]) + x_offset, int(p[1]) + y_offset) for p in pts]
                    results.append((quad, txt, conf))

        if len(candidate_boxes) == 0:
            run_ocr(img)
        else:
            for (x1, y1, x2, y2) in candidate_boxes:
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                run_ocr(crop, x_offset=x1, y_offset=y1)

        return results

    @staticmethod
    def _quad_to_bbox(quad: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _draw_label(img: np.ndarray, bbox: Tuple[int, int, int, int], label: str):
        x1, y1, x2, y2 = bbox
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX  # type: ignore
        scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)  # type: ignore
        pad = 6
        # Position label bar above the box if room, else inside top of box
        bar_x1 = x1
        bar_x2 = max(x2, x1 + text_w + 2 * pad)
        bar_h = text_h + 2 * pad
        bar_y2 = max(y1, bar_h)
        bar_y1 = bar_y2 - bar_h
        # Draw filled rect for label background (black)
        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), LABEL_BG_COLOR, thickness=-1)  # type: ignore
        # Draw label text white with black shadow
        text_x = bar_x1 + pad
        text_y = bar_y2 - pad
        cv2.putText(img, label, (text_x, text_y), font, scale, LABEL_SHADOW_COLOR, thickness + 2, cv2.LINE_AA)  # type: ignore
        cv2.putText(img, label, (text_x, text_y), font, scale, LABEL_TEXT_COLOR, thickness, cv2.LINE_AA)  # type: ignore

    def _mask_bbox_with_label(
        self,
        img: np.ndarray,
        bbox: Tuple[int, int, int, int],
        *,
        label: str,
        mode: str = "partial",
    ) -> None:
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        if mode == "full":
            cv2.rectangle(img, (x1, y1), (x2, y2), MASK_FILL_COLOR, thickness=-1)  # type: ignore
        else:
            # Partial: mask central band 60%
            band_x1 = int(x1 + 0.2 * w)
            band_x2 = int(x1 + 0.8 * w)
            cv2.rectangle(img, (band_x1, y1), (band_x2, y2), MASK_FILL_COLOR, thickness=-1)  # type: ignore

        # Label on top (white text)
        self._draw_label(img, bbox, label)

    @staticmethod
    def _get_style_for_pattern(pattern_name: str) -> Dict[str, Any]:
        return PATTERN_STYLES.get(pattern_name, DEFAULT_STYLE)

    def scan_image(
        self,
        image_bytes: bytes,
    ) -> Tuple[str, bool, float, Dict[str, Any]]:
        self._ensure_cv2()
        img = self._bytes_to_image(image_bytes)
        h, w = img.shape[:2]

        ocr_lines = self._detect_and_recognize(img)

        # Evaluate regex matches and redact
        detections: List[Dict[str, Any]] = []
        risk_score = 0.0
        is_valid = True

        for quad, txt, conf in ocr_lines:
            if not txt or conf < 0.3:
                continue
            bbox = self._quad_to_bbox(quad)
            matched = False
            for name, regex in self.patterns:
                m = regex.search(txt)
                if m:
                    matched = True
                    style = self._get_style_for_pattern(name)
                    detections.append(
                        {
                            "pattern": name,
                            "label": style.get("label"),
                            "text": txt,
                            "confidence": conf,
                            "bbox": {
                                "x1": int(bbox[0]),
                                "y1": int(bbox[1]),
                                "x2": int(bbox[2]),
                                "y2": int(bbox[3]),
                            },
                        }
                    )
                    # Redact and label (black mask, white label)
                    self._mask_bbox_with_label(
                        img,
                        bbox,
                        label=str(style.get("label", name)),
                        mode=self.redact_mode,
                    )
            if matched:
                risk_score = max(risk_score, 0.9)  # escalate risk when any match found
                is_valid = False

        redacted_b64 = self._image_to_base64(img)
        meta = {
            "detections": detections,
            "width": int(w),
            "height": int(h),
        }
        return redacted_b64, is_valid, float(risk_score), meta