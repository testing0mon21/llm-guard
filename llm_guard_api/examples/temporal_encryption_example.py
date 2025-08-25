#!/usr/bin/env python3
"""
Пример использования системы временного шифрования и обфускации промптов.
Демонстрирует различные сценарии использования API.
"""

import asyncio
import json
import os
import time
from datetime import datetime
import aiohttp


class TemporalEncryptionClient:
    """Клиент для работы с API временного шифрования."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = None
        
    async def __aenter__(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        self.session = aiohttp.ClientSession(headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def obfuscate_prompt(
        self, 
        prompt: str, 
        session_id: str, 
        ttl_minutes: int = 30,
        use_ephemeral: bool = True,
        llm_provider: str = "openai"
    ) -> dict:
        """Обфусцирует промпт с временным шифрованием."""
        
        payload = {
            "prompt": prompt,
            "session_id": session_id,
            "ttl_minutes": ttl_minutes,
            "use_ephemeral": use_ephemeral,
            "llm_provider": llm_provider
        }
        
        async with self.session.post(
            f"{self.base_url}/scan/prompt/obfuscate",
            json=payload
        ) as response:
            return await response.json()
            
    async def deobfuscate_response(self, obfuscated_response: str, session_id: str) -> dict:
        """Деобфусцирует ответ от LLM провайдера."""
        
        payload = {
            "obfuscated_response": obfuscated_response,
            "session_id": session_id
        }
        
        async with self.session.post(
            f"{self.base_url}/scan/prompt/deobfuscate_response",
            json=payload
        ) as response:
            return await response.json()
            
    async def llm_proxy_request(
        self,
        prompt: str,
        session_id: str,
        llm_provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        use_obfuscation: bool = True,
        ttl_minutes: int = 30,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> dict:
        """Отправляет запрос через LLM прокси с обфускацией."""
        
        payload = {
            "prompt": prompt,
            "session_id": session_id,
            "llm_provider": llm_provider,
            "model": model,
            "use_obfuscation": use_obfuscation,
            "ttl_minutes": ttl_minutes,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with self.session.post(
            f"{self.base_url}/llm/proxy",
            json=payload
        ) as response:
            return await response.json()
            
    async def get_providers(self) -> dict:
        """Получает список доступных LLM провайдеров."""
        
        async with self.session.get(f"{self.base_url}/llm/providers") as response:
            return await response.json()


async def example_basic_obfuscation():
    """Пример базовой обфускации промпта."""
    
    print("🔐 Пример базовой обфускации промпта")
    print("=" * 50)
    
    async with TemporalEncryptionClient() as client:
        # Исходный промпт с чувствительными данными
        sensitive_prompt = """
        Проанализируй финансовые данные компании:
        - Выручка: $10,000,000
        - Прибыль: $2,500,000
        - Количество сотрудников: 150
        - Секретный проект: Project Alpha
        
        Создай отчет о финансовом состоянии.
        """
        
        session_id = f"session_{int(time.time())}"
        
        # Обфусцируем промпт
        print(f"📝 Обфусцируем промпт для сессии: {session_id}")
        obfuscated = await client.obfuscate_prompt(
            prompt=sensitive_prompt,
            session_id=session_id,
            ttl_minutes=15,  # 15 минут жизни
            use_ephemeral=True
        )
        
        if obfuscated["is_valid"]:
            print("✅ Промпт успешно обфусцирован")
            print(f"📅 Истекает: {datetime.fromtimestamp(obfuscated['expires_at'])}")
            print(f"🔑 Тип обфускации: {obfuscated['obfuscated_data'].get('obfuscation_type', 'unknown')}")
            
            # Показываем инструкции для LLM
            print("\n📋 Инструкции для LLM:")
            print(obfuscated["llm_instructions"])
            
        else:
            print(f"❌ Ошибка обфускации: {obfuscated.get('error', 'Unknown error')}")


async def example_llm_proxy():
    """Пример использования LLM прокси с обфускацией."""
    
    print("\n🚀 Пример использования LLM прокси")
    print("=" * 50)
    
    async with TemporalEncryptionClient() as client:
        # Получаем доступных провайдеров
        print("📋 Получаем список провайдеров...")
        providers = await client.get_providers()
        
        print(f"🔌 Доступно провайдеров: {providers.get('total_providers', 0)}")
        for provider, models in providers.get("providers", {}).items():
            print(f"  - {provider}: {len(models)} моделей")
            
        # Отправляем запрос через прокси
        session_id = f"proxy_session_{int(time.time())}"
        prompt = "Объясни принципы временного шифрования простыми словами."
        
        print(f"\n💬 Отправляем запрос через прокси (сессия: {session_id})")
        print(f"📝 Промпт: {prompt[:50]}...")
        
        # Используем локальный провайдер если OpenAI недоступен
        response = await client.llm_proxy_request(
            prompt=prompt,
            session_id=session_id,
            llm_provider="local",  # Используем локальный LLM
            model="llama2",
            use_obfuscation=True,
            ttl_minutes=10
        )
        
        if response["is_valid"]:
            print("✅ Запрос выполнен успешно")
            print(f"🔐 Обфусцирован: {'Да' if response['was_obfuscated'] else 'Нет'}")
            print(f"🎯 Использовано токенов: {response['tokens_used']}")
            print(f"💬 Ответ: {response['response'][:200]}...")
        else:
            print(f"❌ Ошибка запроса: {response.get('error', 'Unknown error')}")


async def example_time_expiration():
    """Пример демонстрации истечения времени."""
    
    print("\n⏰ Пример истечения временных ключей")
    print("=" * 50)
    
    async with TemporalEncryptionClient() as client:
        session_id = f"expiry_test_{int(time.time())}"
        prompt = "Тестовое сообщение для демонстрации истечения ключей."
        
        # Создаем обфусцированный промпт с коротким TTL
        print("🔐 Создаем обфусцированный промпт с TTL = 1 минута")
        obfuscated = await client.obfuscate_prompt(
            prompt=prompt,
            session_id=session_id,
            ttl_minutes=1,  # Очень короткий TTL для демонстрации
            use_ephemeral=False  # Отключаем эфемерные ключи для простоты
        )
        
        if obfuscated["is_valid"]:
            print("✅ Промпт обфусцирован")
            print(f"📅 Истекает: {datetime.fromtimestamp(obfuscated['expires_at'])}")
            
            # Сразу пытаемся деобфусцировать
            print("\n🔓 Попытка деобфускации сразу после создания:")
            immediate_result = await client.deobfuscate_response(
                json.dumps(obfuscated["obfuscated_data"]),
                session_id
            )
            
            if immediate_result["is_valid"]:
                print("✅ Деобфускация успешна")
                print(f"📝 Результат: {immediate_result['deobfuscated_response'][:100]}...")
            else:
                print(f"❌ Ошибка: {immediate_result.get('error', 'Unknown error')}")
                
            # Ждем истечения времени
            print(f"\n⏳ Ожидание истечения ключа (65 секунд)...")
            await asyncio.sleep(65)
            
            # Пытаемся деобфусцировать после истечения
            print("🔓 Попытка деобфускации после истечения:")
            expired_result = await client.deobfuscate_response(
                json.dumps(obfuscated["obfuscated_data"]),
                session_id
            )
            
            if expired_result["is_valid"]:
                print("⚠️  Неожиданно: деобфускация все еще работает")
            else:
                print("✅ Ожидаемо: ключ истек, деобфускация невозможна")
                print(f"🔒 Сообщение: {expired_result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Ошибка создания: {obfuscated.get('error', 'Unknown error')}")


async def example_security_features():
    """Пример демонстрации функций безопасности."""
    
    print("\n🛡️  Демонстрация функций безопасности")
    print("=" * 50)
    
    async with TemporalEncryptionClient() as client:
        # Тест с эфемерными ключами
        session_id = f"security_test_{int(time.time())}"
        
        print("🔑 Тест эфемерных ключей (одноразовое использование)")
        
        # Создаем обфусцированный промпт с эфемерными ключами
        obfuscated = await client.obfuscate_prompt(
            prompt="Конфиденциальная информация для тестирования безопасности.",
            session_id=session_id,
            ttl_minutes=30,
            use_ephemeral=True  # Включаем эфемерные ключи
        )
        
        if obfuscated["is_valid"]:
            print("✅ Промпт обфусцирован с эфемерными ключами")
            
            # Показываем структуру обфусцированных данных (без секретных частей)
            obf_data = obfuscated["obfuscated_data"]
            print(f"📊 Тип обфускации: {obf_data.get('obfuscation_type', 'unknown')}")
            print(f"🆔 Сессия: {obf_data.get('session_id', 'unknown')}")
            print(f"⏰ Создано: {datetime.fromtimestamp(obf_data.get('created_at', 0))}")
            
            # Демонстрируем инструкции безопасности для LLM
            print(f"\n📋 Длина инструкций безопасности: {len(obfuscated['llm_instructions'])} символов")
            print("🔒 Инструкции включают временные ограничения и требования безопасности")
        else:
            print(f"❌ Ошибка: {obfuscated.get('error', 'Unknown error')}")


async def main():
    """Главная функция с примерами использования."""
    
    print("🚀 Система временного шифрования и обфускации промптов")
    print("🔐 Примеры использования API")
    print("=" * 80)
    
    # Проверяем переменные окружения
    print("🔧 Проверка конфигурации:")
    print(f"  TEMPORAL_MASTER_KEY: {'✅ Установлен' if os.getenv('TEMPORAL_MASTER_KEY') else '⚠️  Не установлен (будет сгенерирован)'}")
    print(f"  OPENAI_API_KEY: {'✅ Установлен' if os.getenv('OPENAI_API_KEY') else '⚠️  Не установлен'}")
    print(f"  ANTHROPIC_API_KEY: {'✅ Установлен' if os.getenv('ANTHROPIC_API_KEY') else '⚠️  Не установлен'}")
    print()
    
    try:
        # Запускаем примеры
        await example_basic_obfuscation()
        await example_llm_proxy()
        await example_time_expiration()
        await example_security_features()
        
        print("\n🎉 Все примеры выполнены успешно!")
        print("📚 Для получения дополнительной информации см. документацию API")
        
    except aiohttp.ClientError as e:
        print(f"\n❌ Ошибка подключения к API: {e}")
        print("🔧 Убедитесь, что сервер запущен на http://localhost:8000")
        
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())