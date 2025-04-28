#!/usr/bin/env python3
"""
Тестирование исправленного EnhancedCodeCipherObfuscator с проверкой работы как с ANTLR, так и без него.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple

# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем ANTLR_AVAILABLE перед всеми остальными импортами
from llm_guard.input_scanners.antlr_grammars import ANTLR_AVAILABLE

# Примеры кода для тестирования
PYTHON_TEST_CODE = """
def check_authentication(username, password):
    # Это просто тестовая функция
    if username == "admin" and password == "secure_password123":
        return True
    return False

# Тестируем функцию
result = check_authentication("admin", "secure_password123")
print(f"Authentication result: {result}")
"""

JAVA_TEST_CODE = """
public class PasswordVerifier {
    private static final String MASTER_PASSWORD = "super_secure_123!";
    
    public static boolean verifyPassword(String password) {
        return MASTER_PASSWORD.equals(password);
    }
    
    public static void main(String[] args) {
        System.out.println("Password verification: " + verifyPassword("super_secure_123!"));
    }
}
"""

def test_with_antlr_available():
    """Тестирование обфускатора с доступным ANTLR"""
    logger.info("=== Тестирование с доступным ANTLR ===")
    
    try:
        # Проверяем, доступен ли ANTLR
        if not ANTLR_AVAILABLE:
            logger.warning("ANTLR не доступен. Пропускаю тест.")
            return False
        
        from llm_guard.input_scanners.enhanced_code_cipher import EnhancedCodeCipherObfuscator
        
        # Создаем обфускатор с использованием ANTLR
        obfuscator = EnhancedCodeCipherObfuscator(
            model_name="gpt2",
            use_antlr=True,
            embedding_dim=768,
            similarity_threshold=0.7
        )
        
        # Тестируем обфускацию Python-кода
        logger.info("Обфускация Python-кода с ANTLR:")
        obfuscated_python = obfuscator._obfuscate_code(PYTHON_TEST_CODE)
        logger.info(f"Оригинальный код:\n{PYTHON_TEST_CODE}")
        logger.info(f"Обфусцированный код:\n{obfuscated_python}")
        
        # Тестируем обфускацию Java-кода
        logger.info("\nОбфускация Java-кода с ANTLR:")
        obfuscated_java = obfuscator._obfuscate_code(JAVA_TEST_CODE)
        logger.info(f"Оригинальный код:\n{JAVA_TEST_CODE}")
        logger.info(f"Обфусцированный код:\n{obfuscated_java}")
        
        # Тестируем обучение конфузионного отображения
        logger.info("\nОбучение конфузионного отображения:")
        obfuscator.train_confusion_mapping([PYTHON_TEST_CODE, JAVA_TEST_CODE])
        
        # Тестируем обфускацию с конфузионным отображением
        logger.info("\nОбфускация Python-кода с конфузионным отображением:")
        confused_python = obfuscator.obfuscate_with_confusion(PYTHON_TEST_CODE)
        logger.info(f"Оригинальный код:\n{PYTHON_TEST_CODE}")
        logger.info(f"Обфусцированный код с конфузией:\n{confused_python}")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при тестировании с ANTLR: {e}")
        return False

def test_without_antlr():
    """Тестирование обфускатора без ANTLR"""
    logger.info("\n=== Тестирование без ANTLR ===")
    
    try:
        # Сохраняем оригинальное значение и устанавливаем ANTLR_AVAILABLE в False на время теста
        original_antlr_available = ANTLR_AVAILABLE
        import llm_guard.input_scanners.antlr_grammars
        llm_guard.input_scanners.antlr_grammars.ANTLR_AVAILABLE = False
        
        # Повторно импортируем модуль, чтобы изменения вступили в силу
        import importlib
        importlib.reload(llm_guard.input_scanners.antlr_grammars)
        
        # Теперь импортируем EnhancedCodeCipherObfuscator
        from llm_guard.input_scanners.enhanced_code_cipher import EnhancedCodeCipherObfuscator
        importlib.reload(llm_guard.input_scanners.enhanced_code_cipher)
        
        # Создаем обфускатор без использования ANTLR
        obfuscator = EnhancedCodeCipherObfuscator(
            model_name="gpt2",
            use_antlr=False,
            embedding_dim=768,
            similarity_threshold=0.7
        )
        
        # Тестируем обфускацию Python-кода
        logger.info("Обфускация Python-кода без ANTLR:")
        obfuscated_python = obfuscator._obfuscate_code(PYTHON_TEST_CODE)
        logger.info(f"Оригинальный код:\n{PYTHON_TEST_CODE}")
        logger.info(f"Обфусцированный код:\n{obfuscated_python}")
        
        # Тестируем обфускацию Java-кода
        logger.info("\nОбфускация Java-кода без ANTLR:")
        obfuscated_java = obfuscator._obfuscate_code(JAVA_TEST_CODE)
        logger.info(f"Оригинальный код:\n{JAVA_TEST_CODE}")
        logger.info(f"Обфусцированный код:\n{obfuscated_java}")
        
        # Тестируем обучение конфузионного отображения
        logger.info("\nОбучение конфузионного отображения без ANTLR:")
        obfuscator.train_confusion_mapping([PYTHON_TEST_CODE, JAVA_TEST_CODE])
        
        # Тестируем обфускацию с конфузионным отображением
        logger.info("\nОбфускация Python-кода с конфузионным отображением без ANTLR:")
        confused_python = obfuscator.obfuscate_with_confusion(PYTHON_TEST_CODE)
        logger.info(f"Оригинальный код:\n{PYTHON_TEST_CODE}")
        logger.info(f"Обфусцированный код с конфузией:\n{confused_python}")
        
        # Восстанавливаем оригинальное значение ANTLR_AVAILABLE
        llm_guard.input_scanners.antlr_grammars.ANTLR_AVAILABLE = original_antlr_available
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при тестировании без ANTLR: {e}")
        return False

def test_error_handling():
    """Тестирование обработки ошибок в обфускаторе"""
    logger.info("\n=== Тестирование обработки ошибок ===")
    
    try:
        from llm_guard.input_scanners.enhanced_code_cipher import EnhancedCodeCipherObfuscator
        
        # Создаем обфускатор
        obfuscator = EnhancedCodeCipherObfuscator(
            model_name="gpt2",
            use_antlr=True,
            embedding_dim=768,
            similarity_threshold=0.7
        )
        
        # Тест обработки пустого кода
        logger.info("Тестирование обработки пустого кода:")
        empty_code = ""
        obfuscated_empty = obfuscator._obfuscate_code(empty_code)
        logger.info(f"Результат: {obfuscated_empty}")
        
        # Тест обработки некорректного кода
        logger.info("\nТестирование обработки некорректного кода:")
        invalid_code = "this is not valid code @#$%^&*"
        obfuscated_invalid = obfuscator._obfuscate_code(invalid_code)
        logger.info(f"Результат: {obfuscated_invalid}")
        
        # Тест метода scan
        logger.info("\nТестирование метода scan с кодом и без кода:")
        
        # Промпт без кода
        prompt_without_code = "Explain how to implement a sorting algorithm."
        result1, metadata1 = obfuscator.scan(prompt_without_code)
        logger.info(f"Результат промпта без кода: {metadata1}")
        
        # Промпт с кодом
        prompt_with_code = f"Please help me understand this code:\n```python\n{PYTHON_TEST_CODE}\n```"
        result2, metadata2 = obfuscator.scan(prompt_with_code)
        logger.info(f"Результат промпта с кодом: {metadata2}")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при тестировании обработки ошибок: {e}")
        return False

def main():
    """Главная функция для тестирования"""
    logger.info("Начало тестирования EnhancedCodeCipherObfuscator")
    
    # Тестируем с ANTLR
    antlr_success = test_with_antlr_available()
    
    # Тестируем без ANTLR
    no_antlr_success = test_without_antlr()
    
    # Тестируем обработку ошибок
    error_success = test_error_handling()
    
    # Выводим итоговый результат
    logger.info("\n=== Итоговые результаты тестирования ===")
    logger.info(f"Тест с ANTLR: {'УСПЕХ' if antlr_success else 'ПРОВАЛ'}")
    logger.info(f"Тест без ANTLR: {'УСПЕХ' if no_antlr_success else 'ПРОВАЛ'}")
    logger.info(f"Тест обработки ошибок: {'УСПЕХ' if error_success else 'ПРОВАЛ'}")
    
    overall_success = antlr_success and no_antlr_success and error_success
    logger.info(f"Общий результат: {'УСПЕХ' if overall_success else 'ПРОВАЛ'}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main()) 