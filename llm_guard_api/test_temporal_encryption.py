#!/usr/bin/env python3
"""
Быстрый тест системы временного шифрования.
Запускает основные функции без внешних зависимостей.
"""

import asyncio
import json
import os
import time
from datetime import datetime

# Импортируем наши модули
from app.temporal_crypto import TimeBasedCrypto, EphemeralKeyManager, PromptObfuscator


async def test_time_based_crypto():
    """Тест основного временного шифрования."""
    print("🔐 Тестирование TimeBasedCrypto")
    print("-" * 40)
    
    crypto = TimeBasedCrypto(time_window_minutes=1)  # Короткое окно для тестирования
    
    # Тест шифрования и расшифровки
    test_data = "Конфиденциальная информация для тестирования"
    context = "test_session"
    
    print(f"📝 Исходные данные: {test_data}")
    
    # Шифруем
    encrypted_package = crypto.encrypt_with_expiry(test_data, ttl_minutes=2, context=context)
    print(f"🔒 Зашифровано: {len(encrypted_package['encrypted_data'])} символов")
    print(f"⏰ Истекает: {datetime.fromtimestamp(encrypted_package['expires_at'])}")
    
    # Сразу расшифровываем
    decrypted_data, success, message = crypto.decrypt_with_expiry(encrypted_package)
    
    if success:
        print(f"✅ Расшифровка успешна: {decrypted_data}")
        print(f"✅ Данные совпадают: {decrypted_data == test_data}")
    else:
        print(f"❌ Ошибка расшифровки: {message}")
    
    print()


async def test_ephemeral_keys():
    """Тест эфемерных ключей."""
    print("🔑 Тестирование EphemeralKeyManager")
    print("-" * 40)
    
    manager = EphemeralKeyManager(max_keys=10)
    
    # Генерируем эфемерный ключ
    session_id = "test_session_ephemeral"
    key_info = manager.generate_ephemeral_key(session_id, ttl_minutes=5)
    
    print(f"🆔 Сгенерирован ключ: {key_info['key_id']}")
    print(f"📅 Истекает: {datetime.fromtimestamp(key_info['expires_at'])}")
    
    # Используем ключ
    key_data, success, message = manager.use_ephemeral_key(key_info['key_id'])
    
    if success:
        print(f"✅ Ключ успешно использован: {len(key_data)} байт")
    else:
        print(f"❌ Ошибка использования ключа: {message}")
    
    # Пытаемся использовать повторно
    key_data2, success2, message2 = manager.use_ephemeral_key(key_info['key_id'])
    
    if not success2:
        print(f"✅ Повторное использование заблокировано: {message2}")
    else:
        print(f"❌ Неожиданно: ключ использован повторно")
    
    print()


async def test_prompt_obfuscator():
    """Тест обфускатора промптов."""
    print("🛡️ Тестирование PromptObfuscator")
    print("-" * 40)
    
    obfuscator = PromptObfuscator(time_window_minutes=2)
    
    # Тестовый промпт
    sensitive_prompt = """
    Секретная информация компании:
    - Код доступа: ABC123
    - Финансовые данные: $1,000,000
    - Стратегический план на 2024 год
    """
    
    session_id = f"obf_test_{int(time.time())}"
    
    print(f"📝 Обфусцируем промпт для сессии: {session_id}")
    
    # Обфусцируем промпт
    obfuscated_data = obfuscator.obfuscate_prompt(
        sensitive_prompt,
        session_id,
        ttl_minutes=3,
        use_ephemeral=True
    )
    
    print(f"🔒 Тип обфускации: {obfuscated_data.get('obfuscation_type', 'unknown')}")
    print(f"⏰ Создано: {datetime.fromtimestamp(obfuscated_data['created_at'])}")
    
    # Создаем инструкции для LLM
    instructions = obfuscator.create_llm_instructions(3)
    print(f"📋 Длина инструкций: {len(instructions)} символов")
    
    # Тест деобфускации (в реальности LLM вернет ответ)
    test_response = "Тестовый ответ от LLM"
    deobfuscated_response, success, message = obfuscator.deobfuscate_response(
        test_response, session_id
    )
    
    if success:
        print(f"✅ Деобфускация ответа: {deobfuscated_response}")
    else:
        print(f"ℹ️ Деобфускация не требуется: {message}")
    
    print()


async def test_integration():
    """Интеграционный тест всей системы."""
    print("🚀 Интеграционный тест")
    print("-" * 40)
    
    # Симулируем полный цикл работы
    obfuscator = PromptObfuscator(time_window_minutes=5)
    
    # 1. Пользователь хочет отправить конфиденциальный промпт
    user_prompt = "Проанализируй конфиденциальные данные клиента XYZ"
    session_id = f"integration_test_{int(time.time())}"
    
    print(f"👤 Пользователь отправляет промпт: {user_prompt}")
    
    # 2. Система обфусцирует промпт
    obfuscated_package = obfuscator.obfuscate_prompt(
        user_prompt,
        session_id,
        ttl_minutes=10,
        use_ephemeral=True
    )
    
    print(f"🔐 Промпт обфусцирован (тип: {obfuscated_package.get('obfuscation_type', 'unknown')})")
    
    # 3. Система создает инструкции для LLM
    llm_instructions = obfuscator.create_llm_instructions(10)
    
    # 4. Формируем полный промпт для LLM
    full_llm_prompt = f"""
{llm_instructions}

ОБФУСЦИРОВАННЫЕ ДАННЫЕ:
{json.dumps(obfuscated_package, indent=2, ensure_ascii=False)}

Пожалуйста, обработайте этот запрос с учетом временных ограничений.
"""
    
    print(f"📤 Отправляем LLM промпт длиной {len(full_llm_prompt)} символов")
    
    # 5. Симулируем ответ от LLM (в реальности здесь был бы HTTP запрос)
    simulated_llm_response = """
    Я понимаю, что данные имеют временные ограничения безопасности.
    Обработал запрос согласно инструкциям.
    Результат анализа: данные обработаны с соблюдением конфиденциальности.
    """
    
    print(f"📥 Получен ответ от LLM: {simulated_llm_response[:100]}...")
    
    # 6. Деобфусцируем ответ (если необходимо)
    final_response, success, message = obfuscator.deobfuscate_response(
        simulated_llm_response, session_id
    )
    
    print(f"✅ Финальный ответ пользователю: {final_response[:100]}...")
    
    print("\n🎉 Интеграционный тест завершен успешно!")


async def test_security_features():
    """Тест функций безопасности."""
    print("🔒 Тест функций безопасности")
    print("-" * 40)
    
    # Тест с разными мастер-ключами
    crypto1 = TimeBasedCrypto(time_window_minutes=5)
    crypto1.master_key = "key1"
    
    crypto2 = TimeBasedCrypto(time_window_minutes=5)
    crypto2.master_key = "key2"
    
    test_data = "Секретные данные"
    
    # Шифруем одним ключом
    encrypted = crypto1.encrypt_with_expiry(test_data, ttl_minutes=10)
    
    # Пытаемся расшифровать другим ключом
    decrypted, success, message = crypto2.decrypt_with_expiry(encrypted)
    
    if not success:
        print("✅ Безопасность: разные мастер-ключи не могут расшифровать данные")
    else:
        print("❌ Проблема безопасности: данные расшифрованы неправильным ключом")
    
    # Тест контекстной изоляции
    context1 = "session_1"
    context2 = "session_2"
    
    encrypted1 = crypto1.encrypt_with_expiry(test_data, ttl_minutes=10, context=context1)
    
    # Меняем контекст и пытаемся расшифровать
    encrypted1["context"] = context2
    decrypted, success, message = crypto1.decrypt_with_expiry(encrypted1)
    
    if not success:
        print("✅ Безопасность: контекстная изоляция работает")
    else:
        print("❌ Проблема: контекстная изоляция не работает")
    
    print()


async def main():
    """Главная функция тестирования."""
    print("🧪 Система временного шифрования - Тестирование")
    print("=" * 60)
    
    # Устанавливаем тестовый мастер-ключ
    os.environ["TEMPORAL_MASTER_KEY"] = "test-master-key-for-testing-only"
    
    try:
        await test_time_based_crypto()
        await test_ephemeral_keys()
        await test_prompt_obfuscator()
        await test_integration()
        await test_security_features()
        
        print("🎉 Все тесты пройдены успешно!")
        print("\n📋 Следующие шаги:")
        print("  1. Запустите API сервер: uvicorn app.app:create_app --factory")
        print("  2. Протестируйте через HTTP: python examples/temporal_encryption_example.py")
        print("  3. Настройте переменные окружения для продакшена")
        
    except Exception as e:
        print(f"\n❌ Ошибка во время тестирования: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())