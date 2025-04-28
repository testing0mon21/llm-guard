# CodeCipherObfuscator

Сканер `CodeCipherObfuscator` предназначен для обфускации исходного кода в промтах, чтобы защитить конфиденциальный код от языковых моделей, сохраняя при этом его функциональность. Реализация основана на методе из исследовательской работы [CodeCipher: Learning to Obfuscate Source Code Against LLMs](https://arxiv.org/pdf/2402.09155.pdf).

## Методика обфускации

Сканер использует следующий подход:
1. Обнаружение блоков кода в промте с помощью регулярных выражений
2. Токенизация кода с использованием модели LLM
3. Создание и оптимизация отображения токен-в-токен для обфускации:
   - Инициализация исходной матрицы эмбеддингов
   - Обучение отображения с помощью оптимизационного алгоритма
   - Минимизация функции потерь для сохранения семантики кода
4. Замена исходных токенов на токены из обфусцированного словаря
5. Сохранение оригинального и обфусцированного кода в защищенном хранилище для последующей деобфускации

## Использование

```python
from llm_guard.input_scanners import CodeCipherObfuscator

# Инициализация сканера
scanner = CodeCipherObfuscator(
    model_name="gpt2",  # Модель для обучения отображения токенов
    max_training_iterations=100,  # Максимальное количество итераций для обучения
    learning_rate=0.01,  # Скорость обучения
    perplexity_threshold=50.0,  # Порог перплексии для ранней остановки
    early_stopping_patience=5,  # Количество итераций без улучшения для ранней остановки
    vault_dir="./obfuscation_vault",  # Директория для хранения обфусцированного кода
    skip_patterns=["test", "example"]  # Паттерны для пропуска обфускации
)

prompt = """Here is some private code:

```python
def calculate_salary(base, bonus):
    return base + bonus * 0.85  # After tax
```

Can you help me understand this function?
"""

# Обфускация исходного кода в промте
sanitized_prompt, scan_metadata = scanner.scan(prompt)
print(sanitized_prompt)
# Output:
# Here is some private code:
#
# ```python
# def zxc_ydl(qrt, wmn):
#     return qrt + wmn * 0.85  # After tax
# ```
#
# Can you help me understand this function?

print(scan_metadata)
# {
#   "is_valid": true,
#   "stats": {
#     "code_blocks_found": 1,
#     "code_blocks_obfuscated": 1
#   }
# }

# Деобфускация (при необходимости)
original_prompt = scanner.deobfuscate(sanitized_prompt)
print(original_prompt)
# Output: [исходный промт с оригинальным кодом]
```

## Параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| model_name | str | "gpt2" | Имя модели для обучения отображения токенов |
| max_training_iterations | int | 100 | Максимальное количество итераций обучения |
| learning_rate | float | 0.01 | Скорость обучения для оптимизации |
| perplexity_threshold | float | 50.0 | Порог перплексии для ранней остановки |
| early_stopping_patience | int | 5 | Количество итераций без улучшения для ранней остановки |
| task_specific_dataset | list | None | Пользовательский набор данных для обучения отображения токенов |
| vault_dir | str | "./obfuscation_vault" | Директория для хранения обфусцированного кода |
| skip_patterns | list | [] | Список регулярных выражений для пропуска обфускации |

## Возвращаемое значение

Сканер возвращает кортеж из двух элементов:
1. Обфусцированный промт (str)
2. Метаданные сканирования (dict) содержащие:
   - is_valid: Булево значение, всегда True для этого сканера
   - stats: Словарь статистики, включающий:
     - code_blocks_found: Количество найденных блоков кода
     - code_blocks_obfuscated: Количество успешно обфусцированных блоков
     - skipped_blocks: Количество пропущенных блоков (если указаны skip_patterns)

## Требования

Для работы сканера требуются следующие пакеты:
- transformers
- torch
- tqdm
- numpy 