#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Пример клиента API для тестирования функционала обфускации кода.

Этот скрипт демонстрирует, как использовать API-сервис LLM Guard для обфускации кода
перед отправкой запросов к LLM моделям.
"""

import json
import time
import requests
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import re

# Директория для сохранения результатов
TEST_DIR = Path(__file__).parent
RESULTS_DIR = TEST_DIR / "api_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Примеры кода на разных языках программирования
CODE_EXAMPLES = {
    "python": """
def calculate_hash(password, salt=None):
    \"\"\"Calculate a secure hash for the given password.\"\"\"
    import hashlib
    import os
    if salt is None:
        salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return {'salt': salt, 'key': key}

def verify_password(stored_password, provided_password):
    \"\"\"Verify a stored password against a provided password.\"\"\"
    salt = stored_password['salt']
    key = stored_password['key']
    new_hash = calculate_hash(provided_password, salt)
    return new_hash['key'] == key
""",
    "javascript": """
// Authentication utility functions for a web application
async function generateSecureToken(userId, secretKey) {
  const crypto = require('crypto');
  const payload = {
    user: userId,
    timestamp: Date.now(),
    expires: Date.now() + 3600000 // 1 hour
  };
  
  const data = JSON.stringify(payload);
  const hmac = crypto.createHmac('sha256', secretKey);
  const signature = hmac.update(data).digest('hex');
  
  return {
    data: Buffer.from(data).toString('base64'),
    signature: signature
  };
}

function verifyToken(token, secretKey) {
  try {
    const crypto = require('crypto');
    const data = Buffer.from(token.data, 'base64').toString();
    const payload = JSON.parse(data);
    
    // Check if token is expired
    if (payload.expires < Date.now()) {
      return { valid: false, reason: 'Token expired' };
    }
    
    // Verify signature
    const hmac = crypto.createHmac('sha256', secretKey);
    const expectedSignature = hmac.update(data).digest('hex');
    
    if (token.signature !== expectedSignature) {
      return { valid: false, reason: 'Invalid signature' };
    }
    
    return { valid: true, payload: payload };
  } catch (error) {
    return { valid: false, reason: 'Error parsing token' };
  }
}
""",
    "java": """
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.Base64;

public class SecurityUtil {
    private static final int SALT_LENGTH = 16;
    private static final String HASH_ALGORITHM = "SHA-256";
    
    /**
     * Generates a secure random salt
     * @return Base64 encoded salt
     */
    public static String generateSalt() {
        SecureRandom random = new SecureRandom();
        byte[] salt = new byte[SALT_LENGTH];
        random.nextBytes(salt);
        return Base64.getEncoder().encodeToString(salt);
    }
    
    /**
     * Hashes a password with a provided salt
     * @param password The password to hash
     * @param salt The salt to use (Base64 encoded)
     * @return Base64 encoded hash
     * @throws NoSuchAlgorithmException If the hash algorithm is not available
     */
    public static String hashPassword(String password, String salt) throws NoSuchAlgorithmException {
        byte[] saltBytes = Base64.getDecoder().decode(salt);
        
        MessageDigest md = MessageDigest.getInstance(HASH_ALGORITHM);
        md.update(saltBytes);
        byte[] hashedPassword = md.digest(password.getBytes());
        
        return Base64.getEncoder().encodeToString(hashedPassword);
    }
    
    /**
     * Verifies a password against a stored hash and salt
     * @param password The password to verify
     * @param storedHash The stored hash (Base64 encoded)
     * @param salt The salt used for hashing (Base64 encoded)
     * @return true if the password matches, false otherwise
     * @throws NoSuchAlgorithmException If the hash algorithm is not available
     */
    public static boolean verifyPassword(String password, String storedHash, String salt) 
            throws NoSuchAlgorithmException {
        String computedHash = hashPassword(password, salt);
        return computedHash.equals(storedHash);
    }
}
"""
}

class LLMGuardClient:
    """Клиент для API-сервиса LLM Guard."""
    
    def __init__(self, api_url: str):
        """
        Инициализация клиента.
        
        Args:
            api_url: URL API-сервиса
        """
        self.api_url = api_url.rstrip('/')
    
    def scan_prompt(self, prompt: str, session_id: str = None) -> Dict[str, Any]:
        """
        Обфусцирует код в промпте.
        
        Args:
            prompt: Исходный промпт
            session_id: ID сессии для отслеживания обфускации
            
        Returns:
            Dict: Результат обфускации
        """
        url = f"{self.api_url}/v1/scan_prompt"
        payload = {"prompt": prompt}
        
        if session_id:
            payload["session_id"] = session_id
            
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Error scanning prompt: {response.status_code} - {response.text}")
        
        return response.json()
    
    def deobfuscate(self, text: str, session_id: str, scanner: str = "CodeCipherObfuscator") -> Dict[str, Any]:
        """
        Деобфусцирует текст (ответ LLM) с кодом.
        
        Args:
            text: Текст с обфусцированным кодом
            session_id: ID сессии, использованный при обфускации
            scanner: Имя сканера, который выполнил обфускацию
            
        Returns:
            Dict: Результат деобфускации
        """
        url = f"{self.api_url}/v1/deobfuscate"
        payload = {
            "text": text,
            "session_id": session_id,
            "scanner": scanner
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Error deobfuscating text: {response.status_code} - {response.text}")
        
        return response.json()
    
    def scan_output(self, prompt: str, output: str) -> Dict[str, Any]:
        """
        Сканирует вывод LLM.
        
        Args:
            prompt: Исходный промпт
            output: Вывод LLM
            
        Returns:
            Dict: Результат сканирования
        """
        url = f"{self.api_url}/v1/scan_output"
        payload = {"prompt": prompt, "output": output}
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Error scanning output: {response.status_code} - {response.text}")
        
        return response.json()


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Извлекает блоки кода из текста.
    
    Args:
        text: Текст с блоками кода
        
    Returns:
        List[Dict[str, str]]: Список блоков кода с языком и содержимым
    """
    pattern = r"```(\w*)\n([\s\S]*?)\n```"
    matches = re.finditer(pattern, text, re.MULTILINE)
    
    blocks = []
    for match in matches:
        language = match.group(1) or "unknown"
        code = match.group(2)
        blocks.append({
            "language": language,
            "code": code
        })
    
    return blocks


def test_api(api_url: str, language: Optional[str] = None):
    """
    Тестирует API-сервис на примерах кода.
    
    Args:
        api_url: URL API-сервиса
        language: Язык программирования для тестирования (опционально)
    """
    client = LLMGuardClient(api_url)
    
    # Фильтруем примеры кода по языку, если указан
    examples = {}
    if language:
        if language in CODE_EXAMPLES:
            examples[language] = CODE_EXAMPLES[language]
        else:
            print(f"Язык {language} не найден. Доступные языки: {', '.join(CODE_EXAMPLES.keys())}")
            return
    else:
        examples = CODE_EXAMPLES
    
    results = []
    
    for lang, code in examples.items():
        print(f"\nТестирование для языка: {lang}")
        print("-" * 50)
        
        # Создаем промпт с кодом
        prompt = f"""Пожалуйста, проанализируй следующий код и предложи улучшения безопасности:

```{lang}
{code}
```

Особое внимание обрати на обработку паролей и защиту от уязвимостей."""
        
        print("Отправка промпта на обфускацию...")
        
        try:
            # Замеряем время выполнения
            start_time = time.time()
            response = client.scan_prompt(prompt)
            elapsed_time = time.time() - start_time
            
            # Сохраняем результаты
            obfuscated_prompt = response.get("prompt", "")
            valid = response.get("valid", False)
            
            # Извлекаем блоки кода
            original_blocks = extract_code_blocks(prompt)
            obfuscated_blocks = extract_code_blocks(obfuscated_prompt)
            
            print(f"Обфускация завершена за {elapsed_time:.2f} сек.")
            print(f"Валидный результат: {valid}")
            print(f"Найдено блоков кода: {len(original_blocks)} оригинальных, {len(obfuscated_blocks)} обфусцированных")
            
            # Сохраняем для сравнения
            result = {
                "language": lang,
                "original_prompt": prompt,
                "obfuscated_prompt": obfuscated_prompt,
                "original_blocks": original_blocks,
                "obfuscated_blocks": obfuscated_blocks,
                "elapsed_time": elapsed_time,
                "valid": valid,
                "metadata": response.get("metadata", {})
            }
            
            results.append(result)
            
            # Сохраняем в отдельные файлы
            lang_dir = RESULTS_DIR / lang
            lang_dir.mkdir(exist_ok=True)
            
            with open(lang_dir / "original.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            
            with open(lang_dir / "obfuscated.txt", "w", encoding="utf-8") as f:
                f.write(obfuscated_prompt)
            
            # Выводим первые строки для сравнения
            if obfuscated_blocks:
                print("\nСравнение (первые 3 строки):")
                orig_lines = original_blocks[0]["code"].split("\n")[:3]
                obfs_lines = obfuscated_blocks[0]["code"].split("\n")[:3]
                
                print("Оригинал:")
                for line in orig_lines:
                    print(f"  {line}")
                
                print("Обфусцировано:")
                for line in obfs_lines:
                    print(f"  {line}")
            
        except Exception as e:
            print(f"Ошибка при обфускации: {e}")
    
    # Сохраняем все результаты
    with open(RESULTS_DIR / "api_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nВсе результаты сохранены в {RESULTS_DIR}")


def main():
    """Основная функция для запуска примера."""
    parser = argparse.ArgumentParser(description="Тестирование API для обфускации кода")
    parser.add_argument("--api", type=str, default="http://localhost:8000", help="URL API-сервиса")
    parser.add_argument("--language", type=str, default=None, help="Язык программирования для тестирования")
    
    args = parser.parse_args()
    
    print(f"Тестирование API-сервиса LLM Guard для обфускации кода")
    print(f"API URL: {args.api}")
    print(f"Выбранный язык: {args.language or 'все'}")
    
    test_api(args.api, args.language)


if __name__ == "__main__":
    main() 