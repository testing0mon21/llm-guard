# LLM Guard - The Security Toolkit for LLM Interactions

LLM Guard by [Protect AI](https://protectai.com/llm-guard) is a comprehensive tool designed to fortify the security of Large Language Models (LLMs).

[**Documentation**](https://llm-guard.com/) | [**Playground**](https://huggingface.co/spaces/ProtectAI/llm-guard-playground) | [**Changelog**](https://llm-guard.com/changelog/)

[![GitHub
stars](https://img.shields.io/github/stars/protectai/llm-guard.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/protectai/llm-guard/stargazers/)
[![MIT license](https://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI - Python Version](https://img.shields.io/pypi/v/llm-guard)](https://pypi.org/project/llm-guard)
[![Downloads](https://static.pepy.tech/badge/llm-guard)](https://pepy.tech/project/llm-guard)
[![Downloads](https://static.pepy.tech/badge/llm-guard/month)](https://pepy.tech/project/llm-guard)

<a href="https://join.slack.com/t/laiyerai/shared_invite/zt-28jv3ci39-sVxXrLs3rQdaN3mIl9IT~w"><img src="https://github.com/protectai/llm-guard/blob/main/docs/assets/join-our-slack-community.png?raw=true" width="200" alt="Join Our Slack Community"></a>

## What is LLM Guard?

![LLM-Guard](https://github.com/protectai/llm-guard/blob/main/docs/assets/flow.png?raw=true)

By offering sanitization, detection of harmful language, prevention of data leakage, and resistance against prompt
injection attacks, LLM-Guard ensures that your interactions with LLMs remain safe and secure.

## Installation

Begin your journey with LLM Guard by downloading the package:

```sh
pip install llm-guard
```

## Getting Started

**Important Notes**:

- LLM Guard is designed for easy integration and deployment in production environments. While it's ready to use
  out-of-the-box, please be informed that we're constantly improving and updating the repository.
- Base functionality requires a limited number of libraries. As you explore more advanced features, necessary libraries
  will be automatically installed.
- Ensure you're using Python version 3.9 or higher. Confirm with: `python --version`.
- Library installation issues? Consider upgrading pip: `python -m pip install --upgrade pip`.

**Examples**:

- Get started with [ChatGPT and LLM Guard](./examples/openai_api.py).
- Deploy LLM Guard as [API](https://llm-guard.com/api/overview/)

## Supported scanners

### Prompt scanners

- [Anonymize](https://llm-guard.com/input_scanners/anonymize/)
- [BanCode](./docs/input_scanners/ban_code.md)
- [BanCompetitors](https://llm-guard.com/input_scanners/ban_competitors/)
- [BanSubstrings](https://llm-guard.com/input_scanners/ban_substrings/)
- [BanTopics](https://llm-guard.com/input_scanners/ban_topics/)
- [Code](https://llm-guard.com/input_scanners/code/)
- [CodeCipherObfuscator](./docs/input_scanners/code_cipher.md)
- [Gibberish](https://llm-guard.com/input_scanners/gibberish/)
- [InvisibleText](https://llm-guard.com/input_scanners/invisible_text/)
- [Language](https://llm-guard.com/input_scanners/language/)
- [PromptInjection](https://llm-guard.com/input_scanners/prompt_injection/)
- [Regex](https://llm-guard.com/input_scanners/regex/)
- [Secrets](https://llm-guard.com/input_scanners/secrets/)
- [Sentiment](https://llm-guard.com/input_scanners/sentiment/)
- [TokenLimit](https://llm-guard.com/input_scanners/token_limit/)
- [Toxicity](https://llm-guard.com/input_scanners/toxicity/)

### Output scanners

- [BanCode](./docs/output_scanners/ban_code.md)
- [BanCompetitors](https://llm-guard.com/output_scanners/ban_competitors/)
- [BanSubstrings](https://llm-guard.com/output_scanners/ban_substrings/)
- [BanTopics](https://llm-guard.com/output_scanners/ban_topics/)
- [Bias](https://llm-guard.com/output_scanners/bias/)
- [Code](https://llm-guard.com/output_scanners/code/)
- [Deanonymize](https://llm-guard.com/output_scanners/deanonymize/)
- [JSON](https://llm-guard.com/output_scanners/json/)
- [Language](https://llm-guard.com/output_scanners/language/)
- [LanguageSame](https://llm-guard.com/output_scanners/language_same/)
- [MaliciousURLs](https://llm-guard.com/output_scanners/malicious_urls/)
- [NoRefusal](https://llm-guard.com/output_scanners/no_refusal/)
- [ReadingTime](https://llm-guard.com/output_scanners/reading_time/)
- [FactualConsistency](https://llm-guard.com/output_scanners/factual_consistency/)
- [Gibberish](https://llm-guard.com/output_scanners/gibberish/)
- [Regex](https://llm-guard.com/output_scanners/regex/)
- [Relevance](https://llm-guard.com/output_scanners/relevance/)
- [Sensitive](https://llm-guard.com/output_scanners/sensitive/)
- [Sentiment](https://llm-guard.com/output_scanners/sentiment/)
- [Toxicity](https://llm-guard.com/output_scanners/toxicity/)
- [URLReachability](https://llm-guard.com/output_scanners/url_reachability/)

## Community, Contributing, Docs & Support

LLM Guard is an open source solution.
We are committed to a transparent development process and highly appreciate any contributions.
Whether you are helping us fix bugs, propose new features, improve our documentation or spread the word,
we would love to have you as part of our community.

- Give us a ⭐️ github star ⭐️ on the top of this page to support what we're doing,
  it means a lot for open source projects!
- Read our
  [docs](https://llm-guard.com/)
  for more info about how to use and customize LLM Guard, and for step-by-step tutorials.
- Post a [Github
  Issue](https://github.com/protectai/llm-guard/issues) to submit a bug report, feature request, or suggest an improvement.
- To contribute to the package, check out our [contribution guidelines](CONTRIBUTING.md), and open a PR.

Join our Slack to give us feedback, connect with the maintainers and fellow users, ask questions,
get help for package usage or contributions, or engage in discussions about LLM security!

<a href="https://join.slack.com/t/laiyerai/shared_invite/zt-28jv3ci39-sVxXrLs3rQdaN3mIl9IT~w"><img src="https://github.com/protectai/llm-guard/blob/main/docs/assets/join-our-slack-community.png?raw=true" width="200" alt="Join Our Slack Community"></a>

### Production Support

We're eager to provide personalized assistance when deploying your LLM Guard to a production environment.

- [Send Email ✉️](mailto:community@protectai.com)

# CodeCipher для LLM Guard

Сервис для обфускации и деобфускации кода при взаимодействии с LLM, основанный на подходе CodeCipher и использующий механизм vault LLM Guard.

## Описание

Данное решение позволяет защитить конфиденциальный код при взаимодействии с языковыми моделями, такими как GPT-4, Claude, и другими. Основные возможности:

- **Обфускация кода**: Преобразование исходного кода в нечитаемую форму при сохранении функциональности
- **Хранение соответствий в vault**: Сохранение оригинального и обфусцированного кода для последующей деобфускации
- **Деобфускация ответов**: Восстановление оригинального кода в ответах LLM

## Структура проекта

- `docker-compose.yml` - Конфигурация для запуска LLM Guard API с поддержкой CodeCipher
- `llm_guard_api/` - Директория с исходным кодом LLM Guard API
- `llm_guard_api/config/scanners.yml` - Конфигурация сканеров, включая CodeCipherObfuscator
- `code_cipher_client.py` - Клиент для демонстрации работы с сервисом
- `test_prompt.txt` - Пример промпта с кодом для тестирования

## Установка и запуск

### Предварительные требования

- Docker и Docker Compose
- Python 3.9+
- pip (для установки зависимостей клиента)

### Запуск LLM Guard API

1. Настройте параметры в файле `docker-compose.yml`, особенно `AUTH_TOKEN` для безопасности:

```yaml
environment:
  - AUTH_TOKEN=your_secure_token_here # Замените на безопасный токен
```

2. Запустите сервис с помощью Docker Compose:

```bash
docker-compose up -d
```

3. Проверьте, что сервис успешно запущен:

```bash
curl http://localhost:8000/healthcheck
```

### Установка зависимостей для клиента

```bash
pip install requests argparse
```

## Использование

### Обфускация кода через клиент

```bash
python code_cipher_client.py --file test_prompt.txt --api-key your_secure_token_here
```

или

```bash
python code_cipher_client.py --prompt "Помоги оптимизировать этот код: \`\`\`python\ndef hello():\n    print('Hello world')\n\`\`\`" --api-key your_secure_token_here
```

### Проверка хранилища vault

Данные хранилища vault сохраняются в томе Docker `cipher_vault_data`. Для просмотра их содержимого можно подключиться к контейнеру:

```bash
docker exec -it llm_guard_api_1 /bin/bash
ls -la /home/user/app/cipher_vault
```

## Ускорение обфускации с помощью предварительно обученных моделей

Для ускорения работы CodeCipherObfuscator можно использовать предварительно обученные модели, что позволяет избежать повторного обучения при каждом запросе:

### 1. Предварительное обучение моделей

Используйте скрипт `code_cipher_pretraining.py` для обучения моделей на примерах кода:

```bash
python code_cipher_pretraining.py --models-dir ./pretrained_models --languages python javascript java cpp
```

Для обучения на собственных примерах кода подготовьте JSON-файл и передайте его в параметре `--code-examples`:

```bash
python code_cipher_pretraining.py --models-dir ./pretrained_models --code-examples ./my_code_examples.json
```

Формат JSON-файла с примерами:
```json
{
    "python": ["def example():\n    return 42", "class Test:\n    pass"],
    "javascript": ["function example() {\n    return 42;\n}"]
}
```

### 2. Использование предварительно обученных моделей

Для использования предварительно обученных моделей в LLM Guard API:

1. Раскомментируйте и настройте `PretrainedCodeCipherObfuscator` в `llm_guard_api/config/scanners.yml`:
```yaml
- type: PretrainedCodeCipherObfuscator
  params:
    pretrained_models_dir: "/home/user/app/pretrained_models"
    vault_dir: "/home/user/app/cipher_vault"
    skip_patterns: ["# COPYRIGHT", "# DO NOT OBFUSCATE"]
```

2. Закомментируйте или удалите стандартный `CodeCipherObfuscator`

3. Перезапустите LLM Guard API:
```bash
docker-compose restart
```

### 3. Тестирование предварительно обученных моделей

Для тестирования скорости и качества обфускации с предварительно обученными моделями используйте скрипт `pretrained_code_cipher_example.py`:

```bash
python pretrained_code_cipher_example.py --models-dir ./pretrained_models
```

## Расширение функциональности

### Добавление эндпоинта деобфускации

В текущей версии LLM Guard API отсутствует эндпоинт для деобфускации ответов. Для полной функциональности необходимо:

1. Добавить новый эндпоинт `/deobfuscate` в `llm_guard_api/app/app.py`
2. Реализовать логику доступа к хранилищу vault и вызова метода `deobfuscate()` 
3. Обновить клиент для использования этого эндпоинта

## Безопасность

- Всегда используйте токен аутентификации для защиты API
- Контролируйте доступ к хранилищу vault, так как оно содержит оригинальный код
- Периодически очищайте устаревшие данные из хранилища

## Ограничения

- CodeCipher может замедлить обработку запросов из-за дополнительных вычислений
- В некоторых случаях обфускация может быть недостаточной для сложного кода
- LLM может генерировать код, отличающийся от обфусцированного, что затруднит деобфускацию

## Дополнительная информация

Подробности о методе CodeCipher можно найти в исследовательской работе [CodeCipher: Learning to Obfuscate Source Code Against LLMs](https://arxiv.org/html/2410.05797v1).

## Лицензия

Данное решение распространяется под лицензией MIT, как и LLM Guard.

# CodeCipher - Система обфускации и деобфускации кода

## Описание
CodeCipher - это система для безопасного обмена кодом с LLM, которая автоматически обфусцирует чувствительный код перед отправкой и деобфусцирует ответы от модели после получения.

Основные компоненты системы:
1. **LLM Guard API** с точкой доступа `/deobfuscate` для деобфускации кода
2. **CodeCipherObfuscator** - сканер для обфускации кода в запросах
3. **Хранилище (vault)** - для сохранения данных, необходимых для деобфускации
4. **Клиентский скрипт** - для простого использования функциональности
5. **Интеграция с DeepSeek API** - для полного цикла тестирования с реальной LLM

## Установка и запуск

### Запуск LLM Guard API

```bash
# Запуск API в Docker контейнере
docker-compose up -d
```

Сервис доступен по адресу http://localhost:8000

### Проверка работоспособности

```bash
# Проверка статуса API
curl http://localhost:8000/healthcheck
```

## Использование

### Обфускация и деобфускация кода

Клиентский скрипт автоматически обфусцирует код, отправит его в LLM и деобфусцирует ответ:

```bash
python code_cipher_client.py --file path/to/your/code.py --api-key your_secret_token_here
```

### Тестирование системы

Система включает скрипты для генерации тестовых примеров кода и проведения комплексного тестирования:

1. Генерация примеров кода в различных языках программирования:

```bash
python generate_code_examples.py --count 100 --output ./code_examples
```

2. Запуск тестов обфускации, вызова реальной LLM и деобфускации:

```bash
python test_code_cipher.py --examples ./code_examples --api-key your_secret_token_here \
    --llm-api-key your_secret_token_here \
    --llm-model deepseek-coder-v2
```

3. Быстрый запуск всего цикла тестирования:

```bash
./run_test.sh --test-files 5
```

## Параметры тестирования

### Генерация примеров кода

```
--count NUM      Количество примеров для генерации (по умолчанию: 1000)
--output DIR     Директория для сохранения примеров (по умолчанию: ./code_examples)
--workers NUM    Количество рабочих потоков (по умолчанию: 4)
```

### Тестирование с использованием DeepSeek API

```
--examples DIR     Директория с примерами кода (по умолчанию: ./code_examples)
--api-url URL      URL API LLM Guard (по умолчанию: http://localhost:8000)
--api-key KEY      Ключ API для LLM Guard (по умолчанию: your_secret_token_here)
--llm-api-url URL  URL API DeepSeek (по умолчанию: https://api.deepseek.com/v1)
--llm-api-key KEY  Ключ API для DeepSeek (по умолчанию: )
--llm-model MODEL  Модель для DeepSeek API (по умолчанию: deepseek-coder-v2)
--workers NUM      Количество рабочих потоков (по умолчанию: 4)
--limit NUM        Ограничение количества тестируемых файлов
--report FILE      Файл для сохранения отчета (по умолчанию: code_cipher_report.csv)
```

### Быстрое тестирование через run_test.sh

```
--api-key KEY      Ключ API для LLM Guard
--llm-api-key KEY  Ключ API для DeepSeek
--llm-api-url URL  URL API DeepSeek
--llm-model MODEL  Модель для DeepSeek API
--examples NUM     Количество генерируемых примеров
--test-files NUM   Количество файлов для тестирования
--workers NUM      Количество рабочих потоков
```

## Структура хранилища

Хранилище (vault) организовано следующим образом:

```
cipher_vault/
  ├── session_id_1/
  │    ├── mapping.json  # Словарь соответствия оригинальных и обфусцированных имен
  │    └── original.py   # Оригинальный код (для проверки)
  ├── session_id_2/
  │    ├── ...
  ...
```

## Отчеты о тестировании

После выполнения тестов создается CSV-файл с подробной информацией о каждом тесте и журнал тестирования в файле `test_code_cipher.log`. Отчет включает:

- Успешность обфускации кода
- Успешность вызова DeepSeek API
- Успешность деобфускации ответа LLM
- Соответствие исходного и деобфусцированного кода
- Время выполнения каждой операции
- Соотношение размеров обфусцированного и оригинального кода

## Характеристики производительности

- **Время обфускации**: Зависит от размера и сложности кода (обычно менее 1 секунды)
- **Время вызова DeepSeek API**: Обычно 2-5 секунд в зависимости от размера кода
- **Время деобфускации**: Зависит от размера и сложности кода (обычно менее 1 секунды)
- **Сохранение функциональности**: Обфусцированный код сохраняет синтаксическую корректность
- **Коэффициент увеличения размера**: Обычно 1.1-1.5x от исходного размера

## Интеграция с DeepSeek API

Система использует DeepSeek API для тестирования полного цикла обфускации и деобфускации:

1. Код обфусцируется с помощью CodeCipherObfuscator
2. Обфусцированный код отправляется в DeepSeek API для анализа и улучшения
3. Ответ от DeepSeek API деобфусцируется для восстановления оригинальных имен переменных и функций
4. Выполняется сравнение для проверки корректности деобфускации

Это позволяет проводить реалистичное тестирование системы с реальной LLM вместо имитации ответов.

## Безопасность

- API защищен токеном аутентификации
- Оригинальный код никогда не передается в исходном виде
- Данные для деобфускации хранятся в безопасном хранилище
- API ключи хранятся локально и могут быть заменены пользователем

## Ограничения

- Некоторые сложные конструкции кода могут обфусцироваться с ошибками
- Производительность может снижаться при обработке очень больших файлов
- При сбоях деобфускации возвращается исходный ответ LLM
- Тестирование с DeepSeek API требует валидный API ключ

## Дальнейшее развитие

- Улучшение обработки сложных конструкций кода
- Поддержка дополнительных языков программирования
- Интеграция с IDE и текстовыми редакторами
- Автоматическая ротация сессий для повышения безопасности
- Поддержка других LLM API (OpenAI, Anthropic, Cohere и т.д.)
