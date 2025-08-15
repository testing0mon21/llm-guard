# Image Scanner API

Endpoint: `POST /scan/image`

Request JSON:
```
{
  "image_base64": "<base64 PNG/JPEG>",
  "patterns": ["\\b\d{3}-\d{2}-\d{4}\\b"],
  "redact_mode": "partial",  
  "yolo_model_path": ""  
}
```

Response JSON:
```
{
  "is_valid": false,
  "risk": 0.9,
  "redacted_image_base64": "<base64 PNG>",
  "detections": [
    {"pattern": "US_SSN_RE", "text": "123-45-6789", "confidence": 0.98, "bbox": {"x1": 10, "y1": 20, "x2": 180, "y2": 55}}
  ],
  "width": 800,
  "height": 600
}
```

Notes:
- Requires `opencv-python-headless`, `pillow`, `paddleocr`, and optionally `ultralytics` with a YOLO11 model.
- YOLO is optional; OCR runs on the full image when not provided.
