from typing import Dict, List, Any

from pydantic import BaseModel, Field


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


# Схемы для временного шифрования и обфускации промптов
class ObfuscatePromptRequest(BaseModel):
    prompt: str = Field(title="Исходный промпт для обфускации")
    session_id: str = Field(title="Идентификатор сессии")
    ttl_minutes: int = Field(title="Время жизни в минутах", default=30)
    use_ephemeral: bool = Field(title="Использовать эфемерные ключи", default=True)
    llm_provider: str = Field(title="LLM провайдер (openai, anthropic, etc.)", default="openai")


class ObfuscatePromptResponse(BaseModel):
    obfuscated_data: Dict[str, Any] = Field(title="Обфусцированные данные")
    session_id: str = Field(title="Идентификатор сессии")
    expires_at: float = Field(title="Время истечения (timestamp)")
    llm_instructions: str = Field(title="Инструкции для LLM о временных ограничениях")
    is_valid: bool = Field(title="Успешность обфускации")
    error: str = Field(title="Сообщение об ошибке", default=None)


class DeobfuscateResponseRequest(BaseModel):
    obfuscated_response: str = Field(title="Обфусцированный ответ от LLM")
    session_id: str = Field(title="Идентификатор сессии")


class DeobfuscateResponseResponse(BaseModel):
    deobfuscated_response: str = Field(title="Деобфусцированный ответ")
    is_valid: bool = Field(title="Успешность деобфускации")
    error: str = Field(title="Сообщение об ошибке", default=None)


class LLMProxyRequest(BaseModel):
    prompt: str = Field(title="Промпт для LLM")
    session_id: str = Field(title="Идентификатор сессии")
    llm_provider: str = Field(title="LLM провайдер", default="openai")
    model: str = Field(title="Модель LLM", default="gpt-3.5-turbo")
    ttl_minutes: int = Field(title="Время жизни шифрования", default=30)
    use_obfuscation: bool = Field(title="Использовать обфускацию", default=True)
    max_tokens: int = Field(title="Максимальное количество токенов", default=1000)
    temperature: float = Field(title="Температура генерации", default=0.7)


class LLMProxyResponse(BaseModel):
    response: str = Field(title="Ответ от LLM")
    session_id: str = Field(title="Идентификатор сессии")
    was_obfuscated: bool = Field(title="Был ли промпт обфусцирован")
    tokens_used: int = Field(title="Использовано токенов", default=0)
    is_valid: bool = Field(title="Успешность запроса")
    error: str = Field(title="Сообщение об ошибке", default=None)
