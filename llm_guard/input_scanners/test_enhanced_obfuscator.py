#!/usr/bin/env python3
"""
Тестирование EnhancedCodeCipherObfuscator с комбинированным подходом ANTLR + CodeCipher.
"""

import os
import sys
import logging
import numpy as np
from typing import List, Dict, Tuple

# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Исправляем импорт на абсолютный путь
from llm_guard.input_scanners.enhanced_code_cipher import EnhancedCodeCipherObfuscator

# Примеры кода для тестирования
PYTHON_TEST_CODE = """
def calculate_factorial(n):
    \"\"\"Вычисляет факториал числа n.\"\"\"
    if n == 0 or n == 1:
        return 1
    else:
        return n * calculate_factorial(n - 1)

# Тестовые данные
test_cases = [5, 7, 10]
for test in test_cases:
    result = calculate_factorial(test)
    print(f"Факториал {test} равен {result}")
"""

JAVA_TEST_CODE = """
public class PasswordValidator {
    private static final String PASSWORD_PATTERN = "^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=])(?=\\S+$).{8,}$";
    
    /**
     * Validates if a password meets the security requirements.
     * @param password The password to validate
     * @return true if the password is valid, false otherwise
     */
    public static boolean isValidPassword(String password) {
        if (password == null) {
            return false;
        }
        return password.matches(PASSWORD_PATTERN);
    }
    
    public static void main(String[] args) {
        String password = "SecureP@ss123";
        System.out.println("Password is valid: " + isValidPassword(password));
    }
}
"""

CSHARP_TEST_CODE = """
using System;
using System.Net;
using System.Collections.Generic;

namespace SecretManager {
    public class CredentialStore {
        private Dictionary<string, string> credentials = new Dictionary<string, string>();
        
        public void AddCredential(string username, string password) {
            credentials[username] = password;
            Console.WriteLine($"Credentials for {username} stored securely");
        }
        
        public bool VerifyCredential(string username, string password) {
            if (!credentials.ContainsKey(username)) {
                return false;
            }
            return credentials[username] == password;
        }
    }
    
    class Program {
        static void Main(string[] args) {
            var store = new CredentialStore();
            store.AddCredential("admin", "super_secret_password123");
            Console.WriteLine("Verification result: " + store.VerifyCredential("admin", "super_secret_password123"));
        }
    }
}
"""

# Функция для оценки качества обфускации
def evaluate_obfuscation(original_code: str, obfuscated_code: str) -> Dict[str, float]:
    """
    Оценивает качество обфускации путем сравнения оригинального и обфусцированного кода.
    
    Args:
        original_code: Исходный код
        obfuscated_code: Обфусцированный код
        
    Returns:
        Словарь с метриками качества
    """
    # Разбиваем код на токены
    original_tokens = set(original_code.replace('(', ' ( ').replace(')', ' ) ').split())
    obfuscated_tokens = set(obfuscated_code.replace('(', ' ( ').replace(')', ' ) ').split())
    
    # Вычисляем базовые метрики
    token_change_ratio = 1.0 - len(original_tokens.intersection(obfuscated_tokens)) / max(len(original_tokens), 1)
    character_change_ratio = 1.0 - len(set(original_code).intersection(set(obfuscated_code))) / max(len(set(original_code)), 1)
    
    # Более сложная метрика для оценки семантического сходства
    # (в реальной реализации можно использовать эмбеддинги)
    line_count_ratio = abs(obfuscated_code.count('\n') - original_code.count('\n')) / max(original_code.count('\n'), 1)
    
    return {
        "token_change_ratio": token_change_ratio,
        "character_change_ratio": character_change_ratio,
        "line_count_ratio": line_count_ratio,
        "avg_obfuscation_score": (token_change_ratio + character_change_ratio) / 2
    }

# Функция для тестирования обфускатора
def test_obfuscator(test_codes: Dict[str, str], use_antlr: bool = True, use_confusion: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Тестирует обфускатор на различных примерах кода.
    
    Args:
        test_codes: Словарь с тестовыми примерами кода
        use_antlr: Использовать ли ANTLR
        use_confusion: Использовать ли матрицу конфузии
        
    Returns:
        Словарь с результатами тестирования
    """
    results = {}
    
    # Создаем обфускатор
    obfuscator = EnhancedCodeCipherObfuscator(
        model_name="distilbert-base-uncased",
        use_antlr=use_antlr,
        embedding_dim=768,
        similarity_threshold=0.7
    )
    
    # Если используем матрицу конфузии, обучаем ее на тестовых данных
    if use_confusion:
        train_dataset = list(test_codes.values())
        obfuscator.train_confusion_mapping(train_dataset)
    
    # Обфусцируем каждый тестовый пример и оцениваем результаты
    for language, code in test_codes.items():
        logger.info(f"Тестирование обфускации для языка: {language}")
        
        # Обфусцируем код
        if use_confusion:
            obfuscated_code = obfuscator.obfuscate_with_confusion(code)
        else:
            obfuscated_code = obfuscator._obfuscate_code(code)
        
        # Оцениваем качество обфускации
        evaluation = evaluate_obfuscation(code, obfuscated_code)
        results[language] = evaluation
        
        # Выводим оригинальный и обфусцированный код
        logger.info(f"\nОригинальный код ({language}):\n{code[:200]}...\n")
        logger.info(f"Обфусцированный код ({language}):\n{obfuscated_code[:200]}...\n")
        logger.info(f"Оценка обфускации: {evaluation}\n")
    
    return results

# Главная функция
def main():
    """Главная функция для тестирования."""
    logger.info("Начало тестирования EnhancedCodeCipherObfuscator")
    
    # Подготавливаем тестовые примеры
    test_codes = {
        "python": PYTHON_TEST_CODE,
        "java": JAVA_TEST_CODE,
        "csharp": CSHARP_TEST_CODE
    }
    
    # Тестируем разные конфигурации
    logger.info("\n\n===== Тестирование только ANTLR =====")
    antlr_results = test_obfuscator(test_codes, use_antlr=True, use_confusion=False)
    
    logger.info("\n\n===== Тестирование комбинированного подхода ANTLR + CodeCipher =====")
    combined_results = test_obfuscator(test_codes, use_antlr=True, use_confusion=True)
    
    # Сравниваем результаты
    logger.info("\n\n===== Сравнение результатов =====")
    for language in test_codes.keys():
        antlr_score = antlr_results[language]["avg_obfuscation_score"]
        combined_score = combined_results[language]["avg_obfuscation_score"]
        improvement = (combined_score - antlr_score) / max(antlr_score, 0.001) * 100
        
        logger.info(f"Язык: {language}")
        logger.info(f"  ANTLR: {antlr_score:.2f}")
        logger.info(f"  ANTLR + CodeCipher: {combined_score:.2f}")
        logger.info(f"  Улучшение: {improvement:.1f}%")
    
    logger.info("\nТестирование завершено")

if __name__ == "__main__":
    main() 