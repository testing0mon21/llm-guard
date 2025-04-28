#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования обфускации многоязычных промптов.
"""

import os
import json
import requests
import time
import argparse
from pathlib import Path

def read_file(file_path):
    """Читает содержимое файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_multi_language_prompt(code):
    """Создает промпт с кодом на разных языках для обфускации."""
    return f"""Проанализируй следующий код на разных языках программирования и предложи улучшения безопасности:

Часть 1 (Java):
```java
{code[:2000]}
```

Часть 2 (JavaScript):
```javascript
{code[2000:4000]}
```

Часть 3 (Python):
```python
{code[4000:6000]}
```

Особое внимание обрати на обработку данных и защиту от уязвимостей."""

def test_multi_language_obfuscation(api_url, code_file):
    """Тестирует обфускацию многоязычного промпта."""
    # Чтение кода из файла
    code = read_file(code_file)
    
    # Создание промпта
    prompt = create_multi_language_prompt(code)
    
    # Вывод информации
    print(f"Тестирование обфускации многоязычного промпта")
    print(f"API URL: {api_url}")
    print(f"Файл с кодом: {code_file}")
    print(f"Размер промпта: {len(prompt)} символов")
    
    # Отправка запроса на обфускацию
    start_time = time.time()
    print("\nОтправка запроса на обфускацию...")
    
    try:
        response = requests.post(
            f"{api_url}/v1/scan_prompt",
            json={"prompt": prompt},
            headers={"Content-Type": "application/json"}
        )
        
        elapsed = time.time() - start_time
        print(f"Запрос выполнен за {elapsed:.2f} сек.")
        
        # Проверка результата
        if response.status_code == 200:
            data = response.json()
            obfuscated_prompt = data.get("prompt", "")
            is_valid = data.get("valid", False)
            
            print(f"Статус: Успешно (код {response.status_code})")
            print(f"Валидный результат: {is_valid}")
            
            # Результаты обфускации
            original_size = len(prompt)
            obfuscated_size = len(obfuscated_prompt)
            ratio = obfuscated_size / original_size if original_size > 0 else 0
            
            print(f"Размер оригинального промпта: {original_size} символов")
            print(f"Размер обфусцированного промпта: {obfuscated_size} символов")
            print(f"Соотношение размеров: {ratio:.2f}")
            
            # Сохранение результатов
            results_dir = Path("multi_language_results")
            results_dir.mkdir(exist_ok=True)
            
            with open(results_dir / "original_prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            
            with open(results_dir / "obfuscated_prompt.txt", "w", encoding="utf-8") as f:
                f.write(obfuscated_prompt)
            
            print(f"\nРезультаты сохранены в директории {results_dir}")
            return True
        else:
            print(f"Статус: Ошибка (код {response.status_code})")
            print(f"Ответ: {response.text}")
            return False
    
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return False

def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description="Тестирование обфускации многоязычных промптов")
    parser.add_argument("--api", type=str, default="http://localhost:8080", help="URL API сервиса")
    parser.add_argument("--file", type=str, required=True, help="Путь к файлу с кодом")
    
    args = parser.parse_args()
    
    # Запуск тестирования
    test_multi_language_obfuscation(args.api, args.file)

if __name__ == "__main__":
    main() 