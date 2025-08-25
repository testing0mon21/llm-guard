"""
Модуль для временного шифрования и обфускации промптов для LLM провайдеров.
Реализует time-based encryption с автоматическим истечением ключей.
"""

import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog

LOGGER = structlog.getLogger(__name__)


class TimeBasedCrypto:
    """
    Система временного шифрования с автоматическим истечением ключей.
    Использует время как динамический параметр в криптографических операциях.
    """
    
    def __init__(self, time_window_minutes: int = 30, salt_length: int = 32):
        """
        Инициализация системы временного шифрования.
        
        Args:
            time_window_minutes: Временное окно в минутах для действия ключа
            salt_length: Длина соли для генерации ключей
        """
        self.time_window_minutes = time_window_minutes
        self.salt_length = salt_length
        self.master_key = os.environ.get("TEMPORAL_MASTER_KEY", self._generate_master_key())
        
    def _generate_master_key(self) -> str:
        """Генерирует мастер-ключ если не задан в переменных окружения."""
        key = secrets.token_urlsafe(32)
        LOGGER.warning("Generated new master key. Set TEMPORAL_MASTER_KEY environment variable for production!")
        return key
        
    def _get_time_slot(self, timestamp: Optional[float] = None) -> int:
        """
        Получает временной слот для заданного времени.
        Время разбивается на слоты по time_window_minutes минут.
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Округляем время до ближайшего временного слота
        slot_seconds = self.time_window_minutes * 60
        return int(timestamp // slot_seconds)
        
    def _derive_key(self, time_slot: int, salt: bytes, context: str = "") -> bytes:
        """
        Выводит временный ключ из мастер-ключа, временного слота и соли.
        """
        # Создаем уникальный контекст для деривации ключа
        key_material = f"{self.master_key}:{time_slot}:{context}".encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        return kdf.derive(key_material)
        
    def encrypt_with_expiry(self, data: str, ttl_minutes: Optional[int] = None, context: str = "") -> Dict[str, Any]:
        """
        Шифрует данные с автоматическим истечением через заданное время.
        
        Args:
            data: Данные для шифрования
            ttl_minutes: Время жизни в минутах (по умолчанию time_window_minutes)
            context: Дополнительный контекст для деривации ключа
            
        Returns:
            Словарь с зашифрованными данными и метаданными
        """
        if ttl_minutes is None:
            ttl_minutes = self.time_window_minutes
            
        current_time = time.time()
        time_slot = self._get_time_slot(current_time)
        expiry_time = current_time + (ttl_minutes * 60)
        
        # Генерируем соль для этого конкретного шифрования
        salt = secrets.token_bytes(self.salt_length)
        
        # Выводим временный ключ
        derived_key = self._derive_key(time_slot, salt, context)
        
        # Создаем Fernet cipher
        fernet = Fernet(base64.urlsafe_b64encode(derived_key))
        
        # Создаем структуру данных с метаданными
        payload = {
            "data": data,
            "created_at": current_time,
            "expires_at": expiry_time,
            "time_slot": time_slot,
            "context": context
        }
        
        # Шифруем payload
        encrypted_data = fernet.encrypt(json.dumps(payload).encode())
        
        return {
            "encrypted_data": base64.urlsafe_b64encode(encrypted_data).decode(),
            "salt": base64.urlsafe_b64encode(salt).decode(),
            "time_slot": time_slot,
            "expires_at": expiry_time,
            "context": context
        }
        
    def decrypt_with_expiry(self, encrypted_package: Dict[str, Any]) -> Tuple[Optional[str], bool, str]:
        """
        Расшифровывает данные с проверкой истечения времени.
        
        Args:
            encrypted_package: Пакет с зашифрованными данными
            
        Returns:
            Tuple: (расшифрованные_данные, успешность, сообщение_об_ошибке)
        """
        try:
            current_time = time.time()
            
            # Проверяем истечение времени
            if current_time > encrypted_package["expires_at"]:
                return None, False, "Ключ истек - данные больше не доступны"
                
            # Восстанавливаем соль и временной слот
            salt = base64.urlsafe_b64decode(encrypted_package["salt"])
            time_slot = encrypted_package["time_slot"]
            context = encrypted_package.get("context", "")
            
            # Выводим тот же ключ
            derived_key = self._derive_key(time_slot, salt, context)
            fernet = Fernet(base64.urlsafe_b64encode(derived_key))
            
            # Расшифровываем данные
            encrypted_data = base64.urlsafe_b64decode(encrypted_package["encrypted_data"])
            decrypted_payload = fernet.decrypt(encrypted_data)
            payload = json.loads(decrypted_payload.decode())
            
            # Дополнительная проверка времени из payload
            if current_time > payload["expires_at"]:
                return None, False, "Данные истекли согласно внутренней метке времени"
                
            return payload["data"], True, "Успешно расшифровано"
            
        except Exception as e:
            LOGGER.error("Ошибка при расшифровке", error=str(e))
            return None, False, f"Ошибка расшифровки: {str(e)}"


class EphemeralKeyManager:
    """
    Менеджер эфемерных ключей для дополнительной безопасности.
    Реализует rolling key system с уникальными ключами для каждого сообщения.
    """
    
    def __init__(self, max_keys: int = 1000):
        """
        Инициализация менеджера эфемерных ключей.
        
        Args:
            max_keys: Максимальное количество ключей в памяти
        """
        self.max_keys = max_keys
        self.key_store: Dict[str, Dict] = {}
        self.used_keys: set = set()
        
    def generate_ephemeral_key(self, session_id: str, ttl_minutes: int = 5) -> Dict[str, Any]:
        """
        Генерирует эфемерный ключ для сессии.
        
        Args:
            session_id: Идентификатор сессии
            ttl_minutes: Время жизни ключа в минутах
            
        Returns:
            Метаданные эфемерного ключа
        """
        key_id = secrets.token_urlsafe(16)
        key_data = secrets.token_bytes(32)
        current_time = time.time()
        
        key_info = {
            "key_id": key_id,
            "key_data": key_data,
            "session_id": session_id,
            "created_at": current_time,
            "expires_at": current_time + (ttl_minutes * 60),
            "used": False
        }
        
        self.key_store[key_id] = key_info
        self._cleanup_expired_keys()
        
        return {
            "key_id": key_id,
            "session_id": session_id,
            "expires_at": key_info["expires_at"]
        }
        
    def use_ephemeral_key(self, key_id: str) -> Tuple[Optional[bytes], bool, str]:
        """
        Использует эфемерный ключ (одноразовое использование).
        
        Args:
            key_id: Идентификатор ключа
            
        Returns:
            Tuple: (ключ, успешность, сообщение)
        """
        if key_id not in self.key_store:
            return None, False, "Ключ не найден"
            
        key_info = self.key_store[key_id]
        current_time = time.time()
        
        if current_time > key_info["expires_at"]:
            del self.key_store[key_id]
            return None, False, "Ключ истек"
            
        if key_info["used"] or key_id in self.used_keys:
            return None, False, "Ключ уже использован"
            
        # Помечаем ключ как использованный
        key_info["used"] = True
        self.used_keys.add(key_id)
        key_data = key_info["key_data"]
        
        # Удаляем ключ из хранилища для безопасности
        del self.key_store[key_id]
        
        return key_data, True, "Ключ успешно использован"
        
    def _cleanup_expired_keys(self):
        """Очищает истекшие ключи из памяти."""
        current_time = time.time()
        expired_keys = [
            key_id for key_id, key_info in self.key_store.items()
            if current_time > key_info["expires_at"]
        ]
        
        for key_id in expired_keys:
            del self.key_store[key_id]
            self.used_keys.discard(key_id)
            
        # Ограничиваем размер хранилища
        if len(self.key_store) > self.max_keys:
            # Удаляем самые старые ключи
            sorted_keys = sorted(
                self.key_store.items(), 
                key=lambda x: x[1]["created_at"]
            )
            for key_id, _ in sorted_keys[:-self.max_keys]:
                del self.key_store[key_id]
                self.used_keys.discard(key_id)


class PromptObfuscator:
    """
    Система обфускации промптов для безопасной передачи LLM провайдерам.
    Комбинирует time-based encryption и ephemeral keys.
    """
    
    def __init__(self, time_window_minutes: int = 30):
        """
        Инициализация системы обфускации промптов.
        
        Args:
            time_window_minutes: Временное окно для time-based encryption
        """
        self.crypto = TimeBasedCrypto(time_window_minutes)
        self.ephemeral_manager = EphemeralKeyManager()
        
    def obfuscate_prompt(
        self, 
        prompt: str, 
        session_id: str, 
        ttl_minutes: Optional[int] = None,
        use_ephemeral: bool = True
    ) -> Dict[str, Any]:
        """
        Обфусцирует промпт для безопасной передачи LLM.
        
        Args:
            prompt: Исходный промпт
            session_id: Идентификатор сессии
            ttl_minutes: Время жизни в минутах
            use_ephemeral: Использовать ли эфемерные ключи
            
        Returns:
            Обфусцированный пакет данных
        """
        context = f"session:{session_id}"
        
        # Основное шифрование с временными ключами
        encrypted_package = self.crypto.encrypt_with_expiry(
            prompt, ttl_minutes, context
        )
        
        result = {
            "session_id": session_id,
            "encrypted_package": encrypted_package,
            "obfuscation_type": "time_based",
            "created_at": time.time()
        }
        
        # Дополнительное шифрование эфемерным ключом
        if use_ephemeral:
            ephemeral_key_info = self.ephemeral_manager.generate_ephemeral_key(
                session_id, ttl_minutes or 30
            )
            
            # Шифруем весь пакет эфемерным ключом
            ephemeral_key_data, success, message = self.ephemeral_manager.use_ephemeral_key(
                ephemeral_key_info["key_id"]
            )
            
            if success:
                fernet = Fernet(base64.urlsafe_b64encode(ephemeral_key_data))
                double_encrypted = fernet.encrypt(json.dumps(result).encode())
                
                result = {
                    "session_id": session_id,
                    "double_encrypted_data": base64.urlsafe_b64encode(double_encrypted).decode(),
                    "ephemeral_key_id": ephemeral_key_info["key_id"],
                    "obfuscation_type": "hybrid",
                    "created_at": time.time()
                }
                
        return result
        
    def deobfuscate_response(self, obfuscated_response: str, session_id: str) -> Tuple[Optional[str], bool, str]:
        """
        Деобфусцирует ответ от LLM провайдера.
        
        Args:
            obfuscated_response: Обфусцированный ответ
            session_id: Идентификатор сессии
            
        Returns:
            Tuple: (деобфусцированный_ответ, успешность, сообщение)
        """
        try:
            # Пытаемся найти зашифрованные блоки в ответе
            # В реальной реализации здесь был бы более сложный парсинг
            
            # Для демонстрации - простая деобфускация временного шифрования
            if isinstance(obfuscated_response, dict) and "encrypted_package" in obfuscated_response:
                return self.crypto.decrypt_with_expiry(obfuscated_response["encrypted_package"])
            
            # Если это просто строка, возвращаем как есть
            return obfuscated_response, True, "Деобфускация не требуется"
            
        except Exception as e:
            LOGGER.error("Ошибка деобфускации ответа", error=str(e))
            return None, False, f"Ошибка деобфускации: {str(e)}"
            
    def create_llm_instructions(self, ttl_minutes: int) -> str:
        """
        Создает инструкции для LLM о временных ограничениях.
        
        Args:
            ttl_minutes: Время жизни данных в минутах
            
        Returns:
            Текст инструкций для LLM
        """
        expiry_time = datetime.now() + timedelta(minutes=ttl_minutes)
        
        return f"""
ВАЖНО: Эти данные имеют временные ограничения безопасности.
- Время истечения: {expiry_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
- Не сохраняйте эту информацию после указанного времени
- Не используйте эти данные для обучения или долгосрочного хранения
- После истечения времени данные становятся недоступными для расшифровки

Обработайте запрос и верните ответ в том же формате шифрования, если это возможно.
"""


# Глобальные экземпляры для использования в API
_prompt_obfuscator = None

def get_prompt_obfuscator() -> PromptObfuscator:
    """Получает глобальный экземпляр обфускатора промптов."""
    global _prompt_obfuscator
    if _prompt_obfuscator is None:
        time_window = int(os.environ.get("TEMPORAL_TIME_WINDOW_MINUTES", "30"))
        _prompt_obfuscator = PromptObfuscator(time_window)
    return _prompt_obfuscator