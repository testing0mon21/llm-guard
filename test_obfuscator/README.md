# Тестирование обфускации кода с помощью CodeCipherObfuscator

Этот модуль предоставляет инструменты для тестирования функционала обфускации кода с использованием CodeCipherObfuscator, реализованного на основе статьи [CodeCipher: Learning to Obfuscate Source Code Against LLMs](https://arxiv.org/html/2410.05797v1).

## Структура проекта

```
test_obfuscator/
│
├── generate_tests.py         # Генерирует тестовые данные с помощью DeepSeek API
├── test_code_cipher.py       # Тестирует функциональность обфускации и деобфускации
├── run_api_service.py        # Запускает API-сервис LLM Guard с CodeCipherObfuscator
├── api_client_example.py     # Пример использования API для обфускации кода
│
├── test_data/                # Директория для хранения тестовых данных
│   ├── test_cases.json       # Тестовые примеры в формате JSON
│   └── code_samples/         # Примеры кода на разных языках
│
├── results/                  # Результаты тестирования обфускации
│   ├── obfuscation_results.json     # Метрики обфускации
│   ├── summary.json                 # Сводная статистика
│   ├── language_stats.csv           # Статистика по языкам
│   ├── task_stats.csv               # Статистика по типам задач
│   ├── obfuscation_visualizations.png   # Визуализации метрик
│   └── sample_*/                    # Примеры обфускации
│
├── api_results/              # Результаты тестирования API
│   └── api_test_results.json        # Результаты API-тестов
│
├── vault/                   # Хранилище для обфусцированного кода
└── config/                  # Конфигурация для API-сервиса
```

## Использование

### 1. Генерация тестовых данных

Генерирует тестовые данные с использованием DeepSeek API:

```bash
# Генерация 10 тестовых примеров (по умолчанию)
python generate_tests.py

# Генерация определенного количества тестовых примеров
python generate_tests.py --num 20

# Генерация 1000 тестовых примеров
python generate_tests.py --full
```

### 2. Тестирование обфускации и деобфускации

Тестирует работу CodeCipherObfuscator на сгенерированных данных:

```bash
# Тестирование с использованием всех доступных тестовых данных
python test_code_cipher.py

# Тестирование с использованием другой модели
python test_code_cipher.py --model "gpt2-medium"

# Тестирование с ограничением количества тестов
python test_code_cipher.py --limit 50

# Тестирование на выборке из доступных тестов
python test_code_cipher.py --sample 100
```

### 3. Запуск API-сервиса

Запускает API-сервис LLM Guard с CodeCipherObfuscator:

```bash
# Запуск API на порту 8000 (по умолчанию)
python run_api_service.py

# Запуск API на другом порту
python run_api_service.py --port 8080

# Запуск API с другой моделью
python run_api_service.py --model "gpt2-medium"
```

### 4. Тестирование API

Тестирует API-сервис с использованием примеров кода:

```bash
# Тестирование API на localhost:8000 (по умолчанию)
python api_client_example.py

# Тестирование API по другому URL
python api_client_example.py --api "http://localhost:8080"

# Тестирование API только для определенного языка программирования
python api_client_example.py --language python
```

## Зависимости

Для работы с тестами требуются следующие зависимости:

```
torch
transformers
tqdm
requests
pandas
matplotlib
seaborn
python-levenshtein
pyyaml
uvicorn
numpy
```

## Пример рабочего процесса

1. Сгенерировать тестовые данные:
   ```bash
   python generate_tests.py --num 20
   ```

2. Запустить тестирование обфускации:
   ```bash
   python test_code_cipher.py
   ```

3. Запустить API-сервис:
   ```bash
   python run_api_service.py
   ```

4. В отдельном терминале протестировать API:
   ```bash
   python api_client_example.py
   ```

5. Изучить результаты в директориях `results/` и `api_results/`. 