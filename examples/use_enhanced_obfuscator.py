#!/usr/bin/env python3
"""
Простой пример использования CodeCipherObfuscator.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict

# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Добавляем путь к llm_guard, если нужно
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем обфускатор
from llm_guard.input_scanners.code_cipher import CodeCipherObfuscator

# Пример кода для тестирования
TEST_CODE = """
def check_password(username, password):
    # Проверяет пароль пользователя.
    #
    # Args:
    #     username: Имя пользователя
    #     password: Пароль
    #        
    # Returns:
    #     bool: True, если пароль верный, иначе False
    
    # Загружаем секретные данные
    secrets = {
        "admin": "super_secret_password123",
        "user": "qwerty123",
        "developer": "c0d3r_p@$$w0rd"
    }
    
    # Проверяем пароль
    if username in secrets:
        return secrets[username] == password
    
    return False

# Пример использования
if __name__ == "__main__":
    result = check_password("admin", "super_secret_password123")
    print(f"Authentication result: {result}")
    
    result = check_password("user", "wrong_password")
    print(f"Authentication result: {result}")
"""

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Пример использования CodeCipherObfuscator")
    parser.add_argument("--use-antlr", action="store_true", help="Использовать ANTLR для анализа кода")
    parser.add_argument("--code-file", type=str, help="Путь к файлу с кодом для обфускации")
    
    args = parser.parse_args()
    
    # Создаем обфускатор
    logger.info("Инициализация обфускатора...")
    obfuscator = CodeCipherObfuscator(
        model_name="gpt2",
        use_antlr=args.use_antlr
    )
    
    # Загружаем код из файла, если указан
    code = TEST_CODE
    if args.code_file and os.path.exists(args.code_file):
        with open(args.code_file, 'r') as f:
            code = f.read()
    
    # Обфусцируем код
    logger.info("Обфускация кода...")
    obfuscated_code = obfuscator._obfuscate_code(code)
    
    # Выводим результаты
    logger.info("\nОригинальный код:\n%s\n", code)
    logger.info("\nОбфусцированный код:\n%s\n", obfuscated_code)

if __name__ == "__main__":
    main() 