# Система временного шифрования и обфускации промптов

Эта система предоставляет безопасные методы для работы с LLM провайдерами, включая временное шифрование промптов с автоматическим истечением ключей.

## 🔐 Основные возможности

### Time-Based Encryption (Временное шифрование)
- Автоматическое истечение ключей через заданное время
- Защита от компрометации старых данных
- Минимальное управление ключами
- Синхронизация времени для корректной работы

### Ephemeral Keys (Эфемерные ключи)
- Уникальные ключи для каждого сообщения
- Одноразовое использование ключей
- Rolling key system с детерминистическими ключами
- Автоматическая очистка использованных ключей

### LLM Provider Integration
- Поддержка OpenAI, Anthropic, локальных LLM
- Прокси для безопасного общения
- Кэширование ответов
- Мониторинг использования токенов

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install cryptography aiohttp fastapi uvicorn
```

### 2. Настройка переменных окружения

```bash
export TEMPORAL_MASTER_KEY="your-secure-master-key-here"
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Запуск сервера

```bash
cd llm_guard_api
uvicorn app.app:create_app --host 0.0.0.0 --port 8000 --factory
```

### 4. Тестирование API

```bash
python examples/temporal_encryption_example.py
```

## 📡 API Endpoints

### POST /scan/prompt/obfuscate
Обфусцирует промпт с временным шифрованием.

**Запрос:**
```json
{
  "prompt": "Конфиденциальный промпт",
  "session_id": "session_123",
  "ttl_minutes": 30,
  "use_ephemeral": true,
  "llm_provider": "openai"
}
```

**Ответ:**
```json
{
  "obfuscated_data": {
    "encrypted_data": "...",
    "salt": "...",
    "time_slot": 12345,
    "expires_at": 1234567890.0,
    "context": "session:session_123"
  },
  "session_id": "session_123",
  "expires_at": 1234567890.0,
  "llm_instructions": "ВАЖНО: Эти данные имеют временные ограничения...",
  "is_valid": true,
  "error": null
}
```

### POST /scan/prompt/deobfuscate_response
Деобфусцирует ответ от LLM провайдера.

**Запрос:**
```json
{
  "obfuscated_response": "зашифрованный ответ",
  "session_id": "session_123"
}
```

### POST /llm/proxy
Прокси для безопасного общения с LLM провайдерами.

**Запрос:**
```json
{
  "prompt": "Ваш промпт",
  "session_id": "session_123",
  "llm_provider": "openai",
  "model": "gpt-3.5-turbo",
  "use_obfuscation": true,
  "ttl_minutes": 30,
  "max_tokens": 1000,
  "temperature": 0.7
}
```

**Ответ:**
```json
{
  "response": "Ответ от LLM",
  "session_id": "session_123",
  "was_obfuscated": true,
  "tokens_used": 150,
  "is_valid": true,
  "error": null
}
```

### GET /llm/providers
Получает список доступных LLM провайдеров и их моделей.

## 🛡️ Безопасность

### Временные ограничения
- Ключи автоматически истекают через заданное время
- Невозможность расшифровки после истечения
- Защита от долгосрочного хранения данных

### Эфемерные ключи
- Каждый ключ используется только один раз
- Автоматическое удаление после использования
- Защита от повторного использования скомпрометированных ключей

### Инструкции для LLM
Система автоматически добавляет инструкции безопасности:
```
ВАЖНО: Эти данные имеют временные ограничения безопасности.
- Время истечения: 2024-01-01 12:00:00 UTC
- Не сохраняйте эту информацию после указанного времени
- Не используйте эти данные для обучения или долгосрочного хранения
- После истечения времени данные становятся недоступными для расшифровки
```

## 🔧 Конфигурация

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `TEMPORAL_MASTER_KEY` | Мастер-ключ для шифрования | Генерируется автоматически |
| `TEMPORAL_TIME_WINDOW_MINUTES` | Временное окно в минутах | 30 |
| `OPENAI_API_KEY` | API ключ OpenAI | - |
| `ANTHROPIC_API_KEY` | API ключ Anthropic | - |
| `VAULT_DIR` | Директория для хранения ключей | `/tmp/cipher_vault` |

### Файл конфигурации
См. `config/temporal_encryption.yml` для полной конфигурации.

## 📊 Мониторинг

### Метрики Prometheus
- `temporal_keys_generated` - количество сгенерированных временных ключей
- `temporal_keys_expired` - количество истекших ключей
- `ephemeral_keys_used` - количество использованных эфемерных ключей
- `llm_requests_obfuscated` - количество обфусцированных запросов
- `llm_requests_total` - общее количество запросов к LLM
- `llm_response_time` - время ответа LLM провайдеров

### Логирование
Система использует структурированное логирование с включением:
- ID сессии для трассировки
- Временные метки операций
- Информация о безопасности
- Ошибки и предупреждения

## 🧪 Примеры использования

### Python Client
```python
import asyncio
from examples.temporal_encryption_example import TemporalEncryptionClient

async def example():
    async with TemporalEncryptionClient() as client:
        # Обфускация промпта
        result = await client.obfuscate_prompt(
            prompt="Конфиденциальные данные",
            session_id="my_session",
            ttl_minutes=15
        )
        
        # Запрос через прокси
        response = await client.llm_proxy_request(
            prompt="Ваш вопрос",
            session_id="my_session",
            use_obfuscation=True
        )

asyncio.run(example())
```

### cURL Examples
```bash
# Обфускация промпта
curl -X POST "http://localhost:8000/scan/prompt/obfuscate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Секретная информация",
    "session_id": "test_session",
    "ttl_minutes": 30
  }'

# Запрос через LLM прокси
curl -X POST "http://localhost:8000/llm/proxy" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Объясни квантовые вычисления",
    "session_id": "test_session",
    "llm_provider": "openai",
    "use_obfuscation": true
  }'
```

## 🔍 Диагностика

### Проверка состояния системы
```bash
# Проверка здоровья API
curl http://localhost:8000/healthz

# Получение списка провайдеров
curl http://localhost:8000/llm/providers

# Метрики Prometheus
curl http://localhost:8000/metrics
```

### Типичные проблемы

1. **Ошибка "Ключ истек"**
   - Увеличьте TTL в запросе
   - Проверьте синхронизацию времени

2. **Провайдер недоступен**
   - Проверьте API ключи в переменных окружения
   - Убедитесь в доступности внешних API

3. **Ошибки шифрования**
   - Проверьте TEMPORAL_MASTER_KEY
   - Убедитесь в корректности зависимостей

## 🤝 Интеграция

### С существующими системами
Система может быть интегрирована с:
- Существующими LLM приложениями
- Системами управления секретами
- Мониторингом и алертингом
- CI/CD пайплайнами

### Расширение функциональности
- Добавление новых LLM провайдеров
- Кастомные алгоритмы шифрования
- Интеграция с внешними vault системами
- Дополнительные метрики безопасности

## 📚 Дополнительные ресурсы

- [Документация по криптографии](https://cryptography.io/)
- [FastAPI документация](https://fastapi.tiangolo.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [Anthropic API](https://docs.anthropic.com/)

## 🐛 Отчеты об ошибках

При обнаружении проблем создайте issue с:
- Описанием проблемы
- Шагами для воспроизведения
- Логами системы
- Версией системы и зависимостей