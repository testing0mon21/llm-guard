#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Пример использования сканера CodeCipherObfuscator.

Этот пример демонстрирует, как использовать сканер CodeCipherObfuscator для обфускации исходного кода
в промпте перед отправкой в языковую модель. CodeCipher использует метод, описанный в статье
"CodeCipher: Learning to Obfuscate Source Code Against LLMs", чтобы защитить конфиденциальный код.
"""

import os
import sys
import re
from pathlib import Path
import torch

# Добавляем родительскую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from llm_guard.input_scanners import CodeCipherObfuscator


def main():
    """Демонстрация использования CodeCipherObfuscator."""
    # Создаем временную директорию для хранилища
    vault_dir = Path("./cipher_vault")
    os.makedirs(vault_dir, exist_ok=True)

    # Инициализируем сканер
    scanner = CodeCipherObfuscator(
        model_name="gpt2",  # Используем небольшую модель для демонстрации
        max_training_iterations=50,  # Увеличиваем число итераций для лучшей обфускации
        learning_rate=0.03,  # Увеличиваем скорость обучения
        perplexity_threshold=100.0,  # Увеличиваем порог
        early_stopping_patience=5,
        vault_dir=str(vault_dir),
        skip_patterns=["# COPYRIGHT"]  # Пропускаем блоки с этим комментарием
    )

    # Простой пример кода для демонстрации
    simple_code = """def calculate_price(base_price, tax_rate):
    # Вычисляет итоговую цену с учетом налога
    return base_price * (1 + tax_rate)"""

    print("\nПрямое тестирование обфускации:")
    print("-" * 50)
    print(f"Оригинальный код:\n{simple_code}")
    
    # Обфусцируем и покажем токены
    obfuscated_simple = scanner._obfuscate_code(simple_code)
    print(f"\nОбфусцированный код: {obfuscated_simple}")
    
    # Создаем промпт с кодом внутри блока markdown
    # ВАЖНО: используем именно такое форматирование блока кода
    markdown_prompt = f"""
Привет! Можешь проанализировать мой код и помочь его оптимизировать? 

```python
{simple_code}
```

Этот код используется в нашей финансовой системе, нужно его оптимизировать.
    """

    # Отладка регулярного выражения
    print("\nОтладка регулярного выражения для блоков кода:")
    # Это регулярное выражение использует сканер для извлечения блоков кода
    pattern = r"```(?:(\w+)\n)?([\s\S]*?)```"
    
    matches = list(re.finditer(pattern, markdown_prompt))
    print(f"Найдено блоков кода: {len(matches)}")
    
    for i, match in enumerate(matches):
        print(f"\nБлок #{i+1}:")
        print(f"Полное совпадение: {repr(match.group(0))}")
        print(f"Язык: {repr(match.group(1))}")
        print(f"Код: {repr(match.group(2))}")
    
    # Извлекаем блоки кода непосредственно через сканер
    code_blocks = scanner._extract_code_blocks(markdown_prompt)
    print(f"\nБлоков кода найдено сканером: {len(code_blocks)}")
    
    for i, block in enumerate(code_blocks):
        print(f"\nБлок #{i+1}:")
        print(f"Язык: {block.language}")
        print(f"Оригинальный код: {repr(block.original)}")
    
    # Применяем сканер для обфускации кода
    print("\nОригинальный промпт:")
    print("-" * 50)
    print(markdown_prompt)
    print("-" * 50)

    # Обфусцируем промпт
    obfuscated_prompt, metadata = scanner.scan(markdown_prompt)

    print("\nОбфусцированный промпт:")
    print("-" * 50)
    print(obfuscated_prompt)
    print("-" * 50)

    # Выводим метаданные
    print("\nМетаданные сканирования:")
    print(f"Блоков кода найдено: {metadata['stats']['code_blocks_found']}")
    print(f"Блоков кода обфусцировано: {metadata['stats']['code_blocks_obfuscated']}")
    print(f"Блоков пропущено: {metadata['stats']['skipped_blocks']}")

    # Извлекаем блоки кода для сравнения
    def extract_code_from_prompt(prompt):
        matches = re.findall(r'```python\n(.*?)\n```', prompt, re.DOTALL)
        return matches[0] if matches else None

    orig_code = extract_code_from_prompt(markdown_prompt)
    obfs_code = extract_code_from_prompt(obfuscated_prompt)
    
    print("\nСравнение исходного и обфусцированного кода:")
    print("Исходный код:")
    print(orig_code)
    print("\nОбфусцированный код:")
    print(obfs_code)
    
    # Ручное форматирование кода для обфускации
    manual_block_format = f"```python\n{simple_code}\n```"
    print("\nТестирование ручного форматирования блока кода:")
    print(f"Блок кода: {repr(manual_block_format)}")
    
    # Заменяем напрямую через re.sub с экранированием
    code_pattern = re.escape(f"```python\n{simple_code}\n```")
    replaced = re.sub(code_pattern, f"```python\n{obfuscated_simple}\n```", manual_block_format)
    print(f"Результат замены: {repr(replaced)}")
    
    # Пробуем другой формат промпта
    test_prompt = f"""
Привет! Можешь проанализировать мой код и помочь его оптимизировать? 

```python
{simple_code}
```

Этот код используется в нашей финансовой системе, нужно его оптимизировать.
    """
    
    # Пробуем обфусцировать этот промпт
    test_obfuscated, test_metadata = scanner.scan(test_prompt)
    
    print("\nТестовый промпт после обфускации:")
    print(test_obfuscated)
    
    # Деобфусцируем обратно для демонстрации
    deobfuscated_prompt = scanner.deobfuscate(obfuscated_prompt)

    print("\nДеобфусцированный промпт:")
    print("-" * 50)
    print(deobfuscated_prompt)
    print("-" * 50)


if __name__ == "__main__":
    main() 