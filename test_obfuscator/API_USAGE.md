# Использование API LLM Guard для обфускации кода

В этом документе описывается, как использовать API LLM Guard для обфускации и деобфускации кода в запросах к языковым моделям (LLM).

## Доступные скрипты

1. `run_api_service.py` - запускает API-сервис LLM Guard с поддержкой CodeCipherObfuscator
2. `api_client_example.py` - демонстрирует базовое использование API с примерами кода
3. `test_api_llm_workflow.py` - полный рабочий процесс с обфускацией/деобфускацией через API и интеграцией с LLM
4. `test_dataset_api.py` - тестирование API на наборе данных из директории `test_data`

## 1. Запуск API-сервиса

API-сервис предоставляет два основных эндпоинта:
- `/v1/scan_prompt` - обфускация кода в промпте
- `/v1/deobfuscate` - деобфускация кода в ответе LLM

Для запуска API-сервиса выполните:

```bash
# Базовый запуск на порту 8000
python run_api_service.py

# Запуск на другом порту
python run_api_service.py --port 8080

# Запуск с другой моделью
python run_api_service.py --model "gpt2-medium"
```

## 2. Использование клиента API

Пример использования API через клиент:

```python
from test_obfuscator.api_client_example import LLMGuardClient

# Инициализация клиента
client = LLMGuardClient(api_url="http://localhost:8000")

# Обфускация промпта
prompt = """Пожалуйста, улучши этот код:

```python
def calculate_sum(a, b):
    return a + b
```

Добавь проверку входных параметров."""

# Отправка запроса на обфускацию
response = client.scan_prompt(prompt)
obfuscated_prompt = response["prompt"]

# ... отправка обфусцированного промпта в LLM ...
# llm_response = call_llm(obfuscated_prompt)

# Деобфускация ответа
# deobfuscated_response = client.deobfuscate(llm_response)
```

## 3. Полный рабочий процесс с использованием API

Скрипт `test_api_llm_workflow.py` демонстрирует полный рабочий процесс:
1. Обфускация кода через API LLM Guard
2. Отправка обфусцированного кода в DeepSeek
3. Получение ответа от DeepSeek
4. Деобфускация ответа через API LLM Guard

```bash
# Базовый запуск
python test_api_llm_workflow.py

# С указанием API URL и ключа DeepSeek
python test_api_llm_workflow.py --guard-url "http://localhost:8000" --deepseek-key "your-api-key"
```

## 4. Тестирование на наборе данных

Скрипт `test_dataset_api.py` позволяет протестировать API на наборе данных из директории `test_data`:

```bash
# Базовый запуск (все тесты)
python test_dataset_api.py

# Ограничение количества тестов
python test_dataset_api.py --limit 5

# Случайная выборка тестов
python test_dataset_api.py --sample 10

# Использование другого URL API
python test_dataset_api.py --api "http://localhost:8080"
```

## Структура API запросов и ответов

### Обфускация кода (`/v1/scan_prompt`)

**Запрос:**
```json
{
  "prompt": "Текст с блоками кода для обфускации",
  "session_id": "уникальный_идентификатор_сессии"
}
```

**Ответ:**
```json
{
  "prompt": "Текст с обфусцированным кодом",
  "valid": true,
  "metadata": {
    "stats": {
      "blocks_found": 2,
      "blocks_obfuscated": 2
    }
  }
}
```

### Деобфускация кода (`/v1/deobfuscate`)

**Запрос:**
```json
{
  "text": "Текст с обфусцированным кодом (ответ LLM)",
  "session_id": "тот_же_идентификатор_сессии",
  "scanner": "CodeCipherObfuscator"
}
```

**Ответ:**
```json
{
  "deobfuscated_text": "Текст с деобфусцированным кодом",
  "is_valid": true,
  "error": null
}
```

## Советы по использованию

1. Всегда используйте один и тот же `session_id` для обфускации и деобфускации в рамках одного диалога.
2. Перед отправкой в LLM проверяйте, что код был корректно обфусцирован (поле `valid` в ответе).
3. При возникновении ошибок в деобфускации, проверьте сохраненные файлы в директории `vault/session_id`.
4. Для обработки больших объемов данных рекомендуется запускать API-сервис в режиме с несколькими воркерами:
   ```bash
   uvicorn app:app --workers 4 --host 0.0.0.0 --port 8000
   ```

## Примеры кода для разных языков

API поддерживает обфускацию кода на различных языках программирования, включая:
- Python
- JavaScript
- Java
- C++
- Ruby
- Go
- и другие

Примеры этих языков можно найти в директории `test_data/code_samples/`. 