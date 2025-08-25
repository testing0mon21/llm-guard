# 🚀 Быстрый старт - Система временного шифрования промптов

Система для безопасной обфускации промптов при общении с LLM провайдерами с автоматическим истечением ключей.

## ✅ Что реализовано

### 🔐 Time-Based Encryption
- ✅ Автоматическое истечение ключей через заданное время
- ✅ Защита от долгосрочного хранения данных
- ✅ Синхронизация времени для корректной работы
- ✅ PBKDF2 для деривации ключей с солью

### 🔑 Ephemeral Keys  
- ✅ Уникальные ключи для каждого сообщения
- ✅ Одноразовое использование ключей
- ✅ Автоматическая очистка использованных ключей
- ✅ Rolling key system

### 🛡️ Prompt Obfuscation
- ✅ Гибридное шифрование (time-based + ephemeral)
- ✅ Инструкции безопасности для LLM
- ✅ Контекстная изоляция по сессиям
- ✅ Автоматическая деобфускация ответов

### 🌐 LLM Provider Integration
- ✅ Поддержка OpenAI, Anthropic, локальных LLM
- ✅ Прокси для безопасного общения
- ✅ Кэширование ответов
- ✅ Мониторинг токенов и времени ответа

### 📡 API Endpoints
- ✅ `POST /scan/prompt/obfuscate` - обфускация промптов
- ✅ `POST /scan/prompt/deobfuscate_response` - деобфускация ответов
- ✅ `POST /llm/proxy` - прокси для LLM провайдеров
- ✅ `GET /llm/providers` - список доступных провайдеров

## 🏃‍♂️ Запуск за 3 минуты

### 1. Установка зависимостей
```bash
cd llm_guard_api
pip install --break-system-packages cryptography aiohttp structlog fastapi uvicorn
```

### 2. Базовый тест
```bash
python3 test_temporal_encryption.py
```

### 3. Запуск API сервера
```bash
# Простой запуск
uvicorn app.app:create_app --factory --host 0.0.0.0 --port 8000

# Или используйте готовый скрипт
chmod +x run_temporal_encryption.sh
./run_temporal_encryption.sh
```

### 4. Проверка работы
```bash
# Проверка здоровья
curl http://localhost:8000/healthz

# Список провайдеров
curl http://localhost:8000/llm/providers

# Документация
open http://localhost:8000/docs
```

## 🔧 Настройка

### Переменные окружения
```bash
export TEMPORAL_MASTER_KEY="your-secure-32-char-key-here"
export TEMPORAL_TIME_WINDOW_MINUTES="30"
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Конфигурационный файл
См. `config/temporal_encryption.yml` для детальной настройки.

## 📝 Примеры использования

### Обфускация промпта
```bash
curl -X POST "http://localhost:8000/scan/prompt/obfuscate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Конфиденциальная информация",
    "session_id": "test_session_123",
    "ttl_minutes": 15,
    "use_ephemeral": true
  }'
```

### Прокси запрос к LLM
```bash
curl -X POST "http://localhost:8000/llm/proxy" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Объясни квантовые вычисления",
    "session_id": "my_session",
    "llm_provider": "openai",
    "model": "gpt-3.5-turbo",
    "use_obfuscation": true,
    "ttl_minutes": 30
  }'
```

### Python клиент
```python
import asyncio
from examples.temporal_encryption_example import TemporalEncryptionClient

async def main():
    async with TemporalEncryptionClient() as client:
        # Обфускация
        result = await client.obfuscate_prompt(
            prompt="Секретные данные",
            session_id="my_session"
        )
        
        # Прокси запрос
        response = await client.llm_proxy_request(
            prompt="Ваш вопрос",
            session_id="my_session"
        )
        print(response["response"])

asyncio.run(main())
```

## 🛡️ Безопасность

### Ключевые особенности
- 🔐 Временное шифрование с автоистечением
- 🔑 Эфемерные ключи одноразового использования
- 🛡️ Контекстная изоляция по сессиям
- ⏰ Инструкции временных ограничений для LLM
- 🔒 Защита от долгосрочного хранения

### Рекомендации продакшена
```bash
# Используйте криптографически стойкий мастер-ключ
export TEMPORAL_MASTER_KEY="$(openssl rand -base64 32)"

# Настройте короткие TTL для чувствительных данных
export TEMPORAL_TIME_WINDOW_MINUTES="10"

# Включите HTTPS для API
# Настройте мониторинг и алертинг
# Регулярно ротируйте ключи
```

## 📊 Мониторинг

### Метрики Prometheus
- `temporal_keys_generated` - сгенерированные ключи
- `ephemeral_keys_used` - использованные эфемерные ключи  
- `llm_requests_obfuscated` - обфусцированные запросы
- `llm_response_time` - время ответа LLM

### Логи
Структурированные логи в JSON формате с информацией о:
- ID сессий для трассировки
- Временных метках операций
- События безопасности
- Ошибки и предупреждения

## 🔍 Диагностика

### Проверка состояния
```bash
# Базовая проверка
curl http://localhost:8000/healthz

# Детальная информация
curl http://localhost:8000/llm/providers

# Метрики
curl http://localhost:8000/metrics
```

### Типичные проблемы

#### "Ключ истек"
- Увеличьте TTL в запросе
- Проверьте синхронизацию времени системы

#### "Провайдер недоступен"  
- Проверьте API ключи в переменных окружения
- Убедитесь в доступности интернета для внешних API

#### "Ошибки шифрования"
- Проверьте TEMPORAL_MASTER_KEY
- Убедитесь в установке всех зависимостей

## 📚 Файлы проекта

```
llm_guard_api/
├── app/
│   ├── temporal_crypto.py      # Временное шифрование
│   ├── llm_providers.py        # Интеграция с LLM
│   ├── schemas.py              # API схемы
│   └── app.py                  # FastAPI приложение
├── config/
│   └── temporal_encryption.yml # Конфигурация
├── examples/
│   └── temporal_encryption_example.py # Примеры
├── test_temporal_encryption.py # Тесты
├── run_temporal_encryption.sh  # Скрипт запуска
└── README_TEMPORAL_ENCRYPTION.md # Документация
```

## 🎯 Следующие шаги

1. **Продакшен деплой:**
   - Настройте HTTPS
   - Добавьте аутентификацию
   - Настройте мониторинг

2. **Расширение функций:**
   - Добавьте новых LLM провайдеров
   - Реализуйте кастомные алгоритмы шифрования
   - Интегрируйте с внешними vault системами

3. **Оптимизация:**
   - Настройте кэширование Redis
   - Добавьте балансировку нагрузки
   - Оптимизируйте производительность

## 🐛 Поддержка

При возникновении проблем:
1. Проверьте логи: `journalctl -u your-service`
2. Запустите тесты: `python3 test_temporal_encryption.py`
3. Проверьте переменные окружения
4. Создайте issue с подробным описанием

---

**🎉 Система готова к использованию!**

Теперь у вас есть полнофункциональная система временного шифрования промптов для безопасного общения с LLM провайдерами.