#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Пример использования API LLM Guard с поддержкой сессий.

Этот скрипт демонстрирует, как использовать API-сервис LLM Guard с сессиями 
для корректной обфускации и деобфускации кода в диалоге с LLM.
"""

import os
import time
import uuid
import requests
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Импортируем клиент LLM Guard
from api_client_example import LLMGuardClient

# Примеры промптов с кодом
PROMPTS = [
    {
        "title": "Python email validator",
        "content": """Улучши этот код для валидации email:

```python
import re

def validate_email(email):
    """Validate an email address using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True
    else:
        return False
```

Добавь проверку длины email и улучши обработку ошибок."""
    },
    {
        "title": "JavaScript password checker",
        "content": """Проверь этот код для проверки безопасности пароля и предложи улучшения:

```javascript
function checkPasswordStrength(password) {
  let score = 0;
  const feedback = [];
  
  if (password.length < 8) {
    feedback.push('Password is too short');
  } else {
    score += 1;
  }
  
  if (!/[a-z]/.test(password)) {
    feedback.push('Add lowercase letters');
  } else {
    score += 1;
  }
  
  if (!/[A-Z]/.test(password)) {
    feedback.push('Add uppercase letters');
  } else {
    score += 1;
  }
  
  return { score, feedback };
}
```

Что еще нужно проверять в пароле для безопасности?"""
    }
]

# Симуляция ответов LLM
LLM_RESPONSES = [
    """Вот улучшенная версия функции валидации email:

```python
import re

def validate_email(email):
    """
    Validate an email address using regex and additional checks.
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if email is None or empty
    if not email or not isinstance(email, str):
        return False
    
    # Check email length
    if len(email) > 320:  # Max email length according to standards
        return False
    
    # Check for basic email pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    
    # Simple return without unnecessary if-else
    return bool(re.match(pattern, email))
```

Это улучшение включает:
1. Проверку на None и пустую строку
2. Проверку типа данных
3. Ограничение длины email (320 символов согласно стандартам)
4. Упрощенный возврат значения без лишних ветвлений
""",
    """Для улучшения функции проверки безопасности пароля рекомендую добавить следующие проверки:

```javascript
function checkPasswordStrength(password) {
  let score = 0;
  const feedback = [];
  
  // Check password type
  if (typeof password !== 'string') {
    return { score: 0, feedback: ['Invalid password format'] };
  }
  
  // Check length
  if (password.length < 8) {
    feedback.push('Password is too short (minimum 8 characters)');
  } else {
    score += 1;
  }
  
  // Check for lowercase letters
  if (!/[a-z]/.test(password)) {
    feedback.push('Add lowercase letters');
  } else {
    score += 1;
  }
  
  // Check for uppercase letters
  if (!/[A-Z]/.test(password)) {
    feedback.push('Add uppercase letters');
  } else {
    score += 1;
  }
  
  // Check for numbers
  if (!/\\d/.test(password)) {
    feedback.push('Add numbers');
  } else {
    score += 1;
  }
  
  // Check for special characters
  if (!/[^\\w\\s]/.test(password)) {
    feedback.push('Add special characters (!@#$%^&*)');
  } else {
    score += 1;
  }
  
  // Check for common patterns and passwords
  const commonPasswords = ['password', 'admin', '123456', 'qwerty'];
  if (commonPasswords.includes(password.toLowerCase())) {
    feedback.push('Using a common password is insecure');
    score = Math.max(0, score - 2);  // Penalize common passwords heavily
  }
  
  // Determine strength
  let strength = 'Weak';
  if (score >= 4) {
    strength = 'Strong';
  } else if (score >= 2) {
    strength = 'Moderate';
  }
  
  return { 
    score, 
    strength,
    feedback: feedback.length > 0 ? feedback : ['Password is secure'] 
  };
}
```

Улучшения включают:
1. Проверку на тип входных данных
2. Проверку на наличие цифр
3. Проверку на наличие специальных символов
4. Проверку на распространенные пароли
5. Добавление уровня надежности (слабый/средний/сильный)
6. Улучшенный вывод информации о надежности пароля
"""
]

class SessionDemo:
    """Демонстрация использования сессий в API LLM Guard."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Инициализация демонстрации.
        
        Args:
            api_url: URL API-сервиса LLM Guard
        """
        self.api_url = api_url
        self.client = LLMGuardClient(api_url)
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        print(f"Инициализирована сессия: {self.session_id}")
        print(f"API URL: {self.api_url}")
    
    def mock_llm_call(self, prompt: str, index: int) -> str:
        """
        Имитирует вызов LLM.
        
        Args:
            prompt: Запрос к LLM
            index: Индекс примера
            
        Returns:
            str: Ответ LLM
        """
        # В реальном сценарии здесь был бы вызов к реальной LLM
        print(f"[Имитация] Отправка запроса к LLM...")
        time.sleep(1)  # Имитация задержки сети
        
        # Используем предопределенные ответы
        if index < len(LLM_RESPONSES):
            return LLM_RESPONSES[index]
        else:
            return "Извините, ответ не готов."
    
    def run_demo(self):
        """Запускает демонстрацию использования API с сессиями."""
        print("\n" + "=" * 80)
        print("Демонстрация API LLM Guard с использованием сессий")
        print("=" * 80)
        
        results = []
        
        for i, prompt_data in enumerate(PROMPTS):
            print(f"\n\n--- Пример {i+1}: {prompt_data['title']} ---\n")
            
            prompt = prompt_data["content"]
            print("Исходный промпт:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)
            
            try:
                # 1. Обфускация промпта
                print("\nШаг 1: Обфускация промпта...")
                obfuscation_response = self.client.scan_prompt(prompt, session_id=self.session_id)
                obfuscated_prompt = obfuscation_response.get("prompt", "")
                
                print("Обфусцированный промпт:")
                print("-" * 40)
                print(obfuscated_prompt)
                print("-" * 40)
                
                # 2. Отправка запроса к LLM (имитация)
                print("\nШаг 2: Отправка запроса к LLM...")
                llm_response = self.mock_llm_call(obfuscated_prompt, i)
                
                print("\nОтвет LLM (все еще с обфусцированным кодом):")
                print("-" * 40)
                print(llm_response)
                print("-" * 40)
                
                # 3. Деобфускация ответа
                print("\nШаг 3: Деобфускация ответа...")
                deobfuscation_response = self.client.deobfuscate(
                    llm_response, 
                    session_id=self.session_id
                )
                deobfuscated_text = deobfuscation_response.get("deobfuscated_text", llm_response)
                
                print("\nДеобфусцированный ответ:")
                print("-" * 40)
                print(deobfuscated_text)
                print("-" * 40)
                
                results.append({
                    "example": i,
                    "title": prompt_data["title"],
                    "original_prompt": prompt,
                    "obfuscated_prompt": obfuscated_prompt,
                    "llm_response": llm_response,
                    "deobfuscated_response": deobfuscated_text,
                    "success": True
                })
                
            except Exception as e:
                print(f"\nОшибка при обработке примера {i+1}: {e}")
                results.append({
                    "example": i,
                    "title": prompt_data["title"],
                    "error": str(e),
                    "success": False
                })
            
            print("\n" + "-" * 80)
        
        # Итоги
        print("\n\nИтоги демонстрации:")
        print("=" * 40)
        
        success_count = sum(1 for r in results if r.get("success", False))
        print(f"Успешно обработано: {success_count}/{len(results)} примеров")
        
        for i, result in enumerate(results):
            status = "✓" if result.get("success", False) else "✗"
            print(f"{status} Пример {i+1}: {result['title']}")
        
        print("\nСессия ID: " + self.session_id)
        print("=" * 40)


def main():
    """Главная функция для запуска демонстрации."""
    parser = argparse.ArgumentParser(description="Демонстрация API LLM Guard с сессиями")
    parser.add_argument("--api", type=str, default="http://localhost:8000", help="URL API LLM Guard")
    
    args = parser.parse_args()
    
    demo = SessionDemo(api_url=args.api)
    demo.run_demo()


if __name__ == "__main__":
    main() 