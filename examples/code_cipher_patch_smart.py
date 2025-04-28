#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Интеллектуальная обфускация кода с помощью CodeCipherObfuscator.

Этот пример демонстрирует, как правильно обфусцировать код с помощью метода CodeCipher:
- Код остается читаемым для LLM
- Код становится трудным для понимания человеком
- Сохраняется синтаксис и структура кода
- Имена переменных и функций заменяются на запутанные версии
"""

import sys
import re
import random
import string
from pathlib import Path

# Добавляем родительскую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from llm_guard.input_scanners import CodeCipherObfuscator

# Создаем словарь для консистентного переименования
name_mapping = {}

# Патчим метод _obfuscate_code для интеллектуального преобразования
original_obfuscate_code = CodeCipherObfuscator._obfuscate_code

def generate_obfuscated_name(name, preserve_hint=True):
    """
    Генерирует обфусцированное имя для идентификатора.
    Если preserve_hint=True, пытается сохранить часть оригинального имени.
    """
    if name in name_mapping:
        return name_mapping[name]
    
    if preserve_hint and len(name) > 3:
        # Сохраняем первую букву и одну характерную часть из середины
        first_char = name[0]
        mid_idx = len(name) // 2
        mid_char = name[mid_idx:mid_idx+2]
        
        # Генерируем случайные символы
        random_part1 = ''.join(random.choices(string.ascii_lowercase, k=3))
        random_part2 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        
        # Формируем запутанное имя
        obf_name = f"{first_char}{random_part1}_{mid_char}{random_part2}"
        
        # Для переменных с подчеркиванием, добавляем больше запутывания
        if '_' in name:
            parts = name.split('_')
            if len(parts) >= 2:
                prefix = parts[0][0]
                suffix = parts[-1][0]
                random_mid = ''.join(random.choices(string.ascii_lowercase, k=5))
                obf_name = f"{prefix}{random_mid}{suffix}_{random.randint(0, 99)}"
    else:
        # Для коротких имен или когда не нужно сохранять подсказки
        prefix = name[0] if name else 'x'
        obf_name = f"{prefix}{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"
    
    name_mapping[name] = obf_name
    return obf_name

def smart_obfuscate_code(self, code: str) -> str:
    """
    Интеллектуальная обфускация кода:
    1. Сохраняет структуру и синтаксис
    2. Заменяет имена на запутанные версии
    3. Сохраняет ключевые слова языка
    """
    # Применим оригинальный метод для статистики
    orig_obfuscated = original_obfuscate_code(self, code)
    
    # Список ключевых слов Python для сохранения
    keywords = ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 
                'import', 'from', 'as', 'with', 'try', 'except', 'finally',
                'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is']
    
    # Сначала находим все идентификаторы (имена функций и переменных)
    # Регулярное выражение для идентификаторов в Python
    identifier_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
    
    # Находим все идентификаторы
    identifiers = set(re.findall(identifier_pattern, code))
    
    # Исключаем ключевые слова и встроенные функции
    identifiers = [ident for ident in identifiers if ident not in keywords and not ident.startswith('__')]
    
    # Создаем обфусцированную версию кода
    obfuscated_code = code
    
    # Для каждого идентификатора генерируем и применяем запутанное имя
    for ident in sorted(identifiers, key=len, reverse=True):  # Сортируем по длине, чтобы избежать частичных замен
        if ident in name_mapping:
            # Используем существующее отображение для консистентности
            obf_name = name_mapping[ident]
        else:
            # Генерируем новое запутанное имя
            obf_name = generate_obfuscated_name(ident)
        
        # Заменяем идентификатор в коде
        # Используем word boundary (\b) для замены только целых слов
        obfuscated_code = re.sub(r'\b' + re.escape(ident) + r'\b', obf_name, obfuscated_code)
    
    # Добавляем комментарий для наглядности
    obfuscated_code = "# [OBFUSCATED BY CodeCipher]\n" + obfuscated_code
    
    return obfuscated_code

# Патчим метод обфускации
CodeCipherObfuscator._obfuscate_code = smart_obfuscate_code

def main():
    """Демонстрация интеллектуальной обфускации кода."""
    # Создаем сканер
    scanner = CodeCipherObfuscator(
        model_name="gpt2",
        max_training_iterations=1  # Минимальное значение для скорости
    )
    
    # Примеры кода для обфускации
    sample_codes = [
        # Простая функция
        """def calculate_price(base_price, tax_rate):
    # Вычисляет итоговую цену с учетом налога
    return base_price * (1 + tax_rate)""",
        
        # Более сложная функция
        """def calculate_monthly_payment(principal, annual_rate, years):
    # Вычисляет ежемесячный платеж по кредиту
    monthly_rate = annual_rate / 12 / 100
    months = years * 12
    
    # Формула аннуитета
    if monthly_rate == 0:
        return principal / months
    
    x = (1 + monthly_rate) ** months
    return principal * monthly_rate * x / (x - 1)""",
        
        # Пример класса
        """class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.balance = balance
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            return True
        return False
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False"""
    ]
    
    # Обфусцируем каждый пример кода
    for i, code in enumerate(sample_codes):
        print(f"\n\n===== Пример #{i+1} =====")
        print("\nИсходный код:")
        print("-" * 50)
        print(code)
        
        # Обфусцируем код
        obfuscated_code = scanner._obfuscate_code(code)
        print("\nОбфусцированный код:")
        print("-" * 50)
        print(obfuscated_code)
    
    # Создаем промпт с кодом
    prompt = f"""
Проанализируй следующий код и помоги его оптимизировать:

```python
{sample_codes[0]}
```

Этот код используется в финансовой системе для расчета цен.
"""
    
    # Обфусцируем промпт
    obfuscated_prompt, metadata = scanner.scan(prompt)
    
    print("\n\n===== Обфускация промпта =====")
    print("\nОригинальный промпт:")
    print("-" * 50)
    print(prompt)
    
    print("\nОбфусцированный промпт:")
    print("-" * 50)
    print(obfuscated_prompt)
    
    print("\nМетаданные сканирования:")
    print(f"Блоков кода найдено: {metadata['stats']['code_blocks_found']}")
    print(f"Блоков кода обфусцировано: {metadata['stats']['code_blocks_obfuscated']}")

if __name__ == "__main__":
    main() 