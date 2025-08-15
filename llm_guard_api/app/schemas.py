from typing import Dict, List

from pydantic import BaseModel, Field


class ScanImageRequest(BaseModel):
    image_base64: str = Field(title="Base64-encoded image (PNG/JPEG)")
    patterns: List[str] = Field(title="Custom regex patterns", default=[])
    redact_mode: str = Field(title="Redaction mode: partial|full", default="partial")
    yolo_model_path: str = Field(title="Path to YOLO11 model (optional)", default="")


class DetectionBBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionItem(BaseModel):
    pattern: str
    text: str
    confidence: float
    bbox: DetectionBBox


class ScanImageResponse(BaseModel):
    is_valid: bool = Field(title="Whether the image is safe")
    risk: float = Field(title="Risk score")
    redacted_image_base64: str = Field(title="Redacted image in base64 (PNG)")
    detections: List[DetectionItem] = Field(title="Detections", default=[])
    width: int
    height: int


class ScanPromptRequest(BaseModel):
    prompt: str = Field(title="Prompt")
    scanners_suppress: List[str] = Field(title="Scanners to suppress", default=[])


class ScanPromptResponse(BaseModel):
    is_valid: bool = Field(title="Whether the prompt is safe")
    scanners: Dict[str, float] = Field(title="Risk scores of individual scanners")


class AnalyzePromptRequest(ScanPromptRequest):
    pass


class AnalyzePromptResponse(ScanPromptResponse):
    sanitized_prompt: str = Field(title="Sanitized prompt")


class ScanOutputRequest(BaseModel):
    prompt: str = Field(title="Prompt")
    output: str = Field(title="Model output")
    scanners_suppress: List[str] = Field(title="Scanners to suppress", default=[])


class ScanOutputResponse(BaseModel):
    is_valid: bool = Field(title="Whether the output is safe")
    scanners: Dict[str, float] = Field(title="Risk scores of individual scanners")


class AnalyzeOutputRequest(ScanOutputRequest):
    pass


class AnalyzeOutputResponse(ScanOutputResponse):
    sanitized_output: str = Field(title="Sanitized output")


# Новые схемы для деобфускации
class DeobfuscateRequest(BaseModel):
    text: str = Field(title="Обфусцированный текст (ответ LLM)")
    session_id: str = Field(title="ID сессии обфускации")
    scanner: str = Field(title="Имя сканера, который выполнил обфускацию", default="CodeCipherObfuscator")


class DeobfuscateResponse(BaseModel):
    deobfuscated_text: str = Field(title="Деобфусцированный текст")
    is_valid: bool = Field(title="Успешность деобфускации")
    error: str = Field(title="Сообщение об ошибке (если есть)", default=None)
