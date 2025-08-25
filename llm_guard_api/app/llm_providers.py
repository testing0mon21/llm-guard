"""
Модуль для интеграции с различными LLM провайдерами.
Поддерживает OpenAI, Anthropic и другие популярные провайдеры.
"""

import asyncio
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import aiohttp
import json
import structlog

LOGGER = structlog.getLogger(__name__)


class LLMProvider(ABC):
    """Базовый класс для LLM провайдеров."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    @abstractmethod
    async def send_request(
        self, 
        prompt: str, 
        model: str, 
        max_tokens: int, 
        temperature: float,
        **kwargs
    ) -> Tuple[str, int, bool, str]:
        """
        Отправляет запрос к LLM провайдеру.
        
        Returns:
            Tuple: (ответ, использованные_токены, успешность, ошибка)
        """
        pass
        
    @abstractmethod
    def get_supported_models(self) -> list:
        """Возвращает список поддерживаемых моделей."""
        pass


class OpenAIProvider(LLMProvider):
    """Провайдер для OpenAI API."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.openai.com/v1"
        self.supported_models = [
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
        ]
        
    async def send_request(
        self, 
        prompt: str, 
        model: str = "gpt-3.5-turbo", 
        max_tokens: int = 1000, 
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[str, int, bool, str]:
        """Отправляет запрос к OpenAI API."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        LOGGER.error("OpenAI API error", status=response.status, error=error_text)
                        return "", 0, False, f"API error: {response.status} - {error_text}"
                    
                    data = await response.json()
                    
                    if "choices" not in data or not data["choices"]:
                        return "", 0, False, "No response from OpenAI"
                    
                    response_text = data["choices"][0]["message"]["content"]
                    tokens_used = data.get("usage", {}).get("total_tokens", 0)
                    
                    return response_text, tokens_used, True, ""
                    
        except asyncio.TimeoutError:
            return "", 0, False, "Request timeout"
        except Exception as e:
            LOGGER.error("OpenAI request failed", error=str(e))
            return "", 0, False, f"Request failed: {str(e)}"
            
    def get_supported_models(self) -> list:
        return self.supported_models


class AnthropicProvider(LLMProvider):
    """Провайдер для Anthropic Claude API."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.anthropic.com/v1"
        self.supported_models = [
            "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
        
    async def send_request(
        self, 
        prompt: str, 
        model: str = "claude-3-5-sonnet-20241022", 
        max_tokens: int = 1000, 
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[str, int, bool, str]:
        """Отправляет запрос к Anthropic API."""
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        LOGGER.error("Anthropic API error", status=response.status, error=error_text)
                        return "", 0, False, f"API error: {response.status} - {error_text}"
                    
                    data = await response.json()
                    
                    if "content" not in data or not data["content"]:
                        return "", 0, False, "No response from Anthropic"
                    
                    response_text = data["content"][0]["text"]
                    tokens_used = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)
                    
                    return response_text, tokens_used, True, ""
                    
        except asyncio.TimeoutError:
            return "", 0, False, "Request timeout"
        except Exception as e:
            LOGGER.error("Anthropic request failed", error=str(e))
            return "", 0, False, f"Request failed: {str(e)}"
            
    def get_supported_models(self) -> list:
        return self.supported_models


class LocalLLMProvider(LLMProvider):
    """Провайдер для локальных LLM (Ollama, etc.)."""
    
    def __init__(self, api_key: str = "", base_url: str = "http://localhost:11434"):
        super().__init__(api_key)
        self.base_url = base_url
        self.supported_models = ["llama2", "codellama", "mistral", "phi"]
        
    async def send_request(
        self, 
        prompt: str, 
        model: str = "llama2", 
        max_tokens: int = 1000, 
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[str, int, bool, str]:
        """Отправляет запрос к локальному LLM."""
        
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            },
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        LOGGER.error("Local LLM API error", status=response.status, error=error_text)
                        return "", 0, False, f"API error: {response.status} - {error_text}"
                    
                    data = await response.json()
                    
                    if "response" not in data:
                        return "", 0, False, "No response from local LLM"
                    
                    response_text = data["response"]
                    tokens_used = len(response_text.split())  # Приблизительная оценка
                    
                    return response_text, tokens_used, True, ""
                    
        except asyncio.TimeoutError:
            return "", 0, False, "Request timeout"
        except Exception as e:
            LOGGER.error("Local LLM request failed", error=str(e))
            return "", 0, False, f"Request failed: {str(e)}"
            
    def get_supported_models(self) -> list:
        return self.supported_models


class LLMProviderFactory:
    """Фабрика для создания LLM провайдеров."""
    
    _providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "local": LocalLLMProvider,
        "ollama": LocalLLMProvider
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, **kwargs) -> Optional[LLMProvider]:
        """
        Создает экземпляр LLM провайдера.
        
        Args:
            provider_name: Имя провайдера
            **kwargs: Дополнительные параметры для провайдера
            
        Returns:
            Экземпляр провайдера или None если провайдер не поддерживается
        """
        if provider_name not in cls._providers:
            LOGGER.error("Unsupported provider", provider=provider_name)
            return None
            
        provider_class = cls._providers[provider_name]
        
        # Получаем API ключ из переменных окружения или параметров
        api_key = kwargs.get("api_key")
        if not api_key:
            env_key_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "local": "",
                "ollama": ""
            }
            env_key = env_key_map.get(provider_name, "")
            if env_key:
                api_key = os.environ.get(env_key)
                
        if not api_key and provider_name in ["openai", "anthropic"]:
            LOGGER.error("API key required for provider", provider=provider_name)
            return None
            
        try:
            if provider_name in ["local", "ollama"]:
                base_url = kwargs.get("base_url", "http://localhost:11434")
                return provider_class(api_key or "", base_url)
            else:
                return provider_class(api_key)
                
        except Exception as e:
            LOGGER.error("Failed to create provider", provider=provider_name, error=str(e))
            return None
    
    @classmethod
    def get_supported_providers(cls) -> list:
        """Возвращает список поддерживаемых провайдеров."""
        return list(cls._providers.keys())


class LLMProxy:
    """
    Прокси для работы с различными LLM провайдерами.
    Поддерживает обфускацию промптов и кэширование ответов.
    """
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.response_cache: Dict[str, Dict] = {}
        self.max_cache_size = 1000
        
    def add_provider(self, name: str, provider: LLMProvider):
        """Добавляет провайдера в прокси."""
        self.providers[name] = provider
        
    def get_provider(self, name: str) -> Optional[LLMProvider]:
        """Получает провайдера по имени."""
        if name not in self.providers:
            # Пытаемся создать провайдера автоматически
            provider = LLMProviderFactory.create_provider(name)
            if provider:
                self.providers[name] = provider
                return provider
            return None
        return self.providers[name]
        
    async def send_request(
        self,
        provider_name: str,
        prompt: str,
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        use_cache: bool = True,
        **kwargs
    ) -> Tuple[str, int, bool, str]:
        """
        Отправляет запрос через указанного провайдера.
        
        Args:
            provider_name: Имя провайдера
            prompt: Промпт для LLM
            model: Модель LLM
            max_tokens: Максимальное количество токенов
            temperature: Температура генерации
            use_cache: Использовать кэширование
            **kwargs: Дополнительные параметры
            
        Returns:
            Tuple: (ответ, токены, успешность, ошибка)
        """
        # Проверяем кэш
        if use_cache:
            cache_key = self._generate_cache_key(provider_name, prompt, model, max_tokens, temperature)
            if cache_key in self.response_cache:
                cached = self.response_cache[cache_key]
                # Проверяем время жизни кэша (5 минут)
                if time.time() - cached["timestamp"] < 300:
                    LOGGER.debug("Using cached response", cache_key=cache_key)
                    return cached["response"], cached["tokens"], cached["success"], cached["error"]
                else:
                    # Удаляем устаревшую запись
                    del self.response_cache[cache_key]
        
        # Получаем провайдера
        provider = self.get_provider(provider_name)
        if not provider:
            return "", 0, False, f"Provider {provider_name} not available"
            
        # Отправляем запрос
        start_time = time.time()
        response, tokens, success, error = await provider.send_request(
            prompt, model, max_tokens, temperature, **kwargs
        )
        elapsed_time = time.time() - start_time
        
        LOGGER.info(
            "LLM request completed",
            provider=provider_name,
            model=model,
            success=success,
            tokens=tokens,
            elapsed_time=round(elapsed_time, 2)
        )
        
        # Сохраняем в кэш при успехе
        if use_cache and success:
            cache_key = self._generate_cache_key(provider_name, prompt, model, max_tokens, temperature)
            self.response_cache[cache_key] = {
                "response": response,
                "tokens": tokens,
                "success": success,
                "error": error,
                "timestamp": time.time()
            }
            
            # Ограничиваем размер кэша
            if len(self.response_cache) > self.max_cache_size:
                # Удаляем самые старые записи
                oldest_keys = sorted(
                    self.response_cache.keys(),
                    key=lambda k: self.response_cache[k]["timestamp"]
                )[:100]
                for key in oldest_keys:
                    del self.response_cache[key]
        
        return response, tokens, success, error
        
    def _generate_cache_key(self, provider: str, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """Генерирует ключ для кэша."""
        import hashlib
        content = f"{provider}:{model}:{max_tokens}:{temperature}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def get_available_providers(self) -> Dict[str, list]:
        """Возвращает доступных провайдеров и их модели."""
        result = {}
        for name, provider in self.providers.items():
            result[name] = provider.get_supported_models()
        
        # Добавляем потенциально доступных провайдеров
        for provider_name in LLMProviderFactory.get_supported_providers():
            if provider_name not in result:
                provider = LLMProviderFactory.create_provider(provider_name)
                if provider:
                    result[provider_name] = provider.get_supported_models()
                    
        return result


# Глобальный экземпляр прокси
_llm_proxy = None

def get_llm_proxy() -> LLMProxy:
    """Получает глобальный экземпляр LLM прокси."""
    global _llm_proxy
    if _llm_proxy is None:
        _llm_proxy = LLMProxy()
    return _llm_proxy