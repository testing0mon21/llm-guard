#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования полного рабочего процесса с использованием API-сервиса:
1. Обфускация кода через API LLM Guard
2. Отправка обфусцированного кода в DeepSeek
3. Получение ответа от DeepSeek
4. Деобфускация ответа через API LLM Guard

Этот скрипт использует API-сервис LLM Guard для обфускации/деобфускации
и интегрируется с DeepSeek для полного сквозного тестирования.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
import argparse
from tqdm import tqdm

# Директории
TEST_DIR = Path(__file__).parent
RESULTS_DIR = TEST_DIR / "api_llm_workflow_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Примеры запросов к LLM с кодом
CODE_PROMPTS = [
    {
        "prompt": "Можешь ли ты улучшить следующую функцию в Python для валидации email?\n\n```python\nimport re\n\ndef validate_email(email):\n    \"\"\"Validate an email address using regex.\"\"\"\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    if re.match(pattern, email):\n        return True\n    else:\n        return False\n```\n\nНужно добавить проверку длины email и улучшить обработку ошибок.",
        "language": "python"
    },
    {
        "prompt": "Проанализируй эту JavaScript функцию для проверки безопасности паролей:\n\n```javascript\nfunction checkPasswordStrength(password) {\n  let score = 0;\n  const feedback = [];\n  \n  if (password.length < 8) {\n    feedback.push('Password is too short');\n  } else {\n    score += 1;\n  }\n  \n  if (!/[a-z]/.test(password)) {\n    feedback.push('Add lowercase letters');\n  } else {\n    score += 1;\n  }\n  \n  if (!/[A-Z]/.test(password)) {\n    feedback.push('Add uppercase letters');\n  } else {\n    score += 1;\n  }\n  \n  return { score, feedback };\n}\n```\n\nЧто можно улучшить с точки зрения безопасности?",
        "language": "javascript"
    }
]

class APIConfig:
    """Конфигурация для API-сервисов"""
    def __init__(
        self, 
        guard_api_url="http://localhost:8000",
        deepseek_api_key=None, 
        deepseek_model="deepseek-chat", 
        deepseek_api_base="https://api.deepseek.com/v1"
    ):
        # LLM Guard API
        self.guard_api_url = guard_api_url
        
        # DeepSeek API
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_model = deepseek_model
        self.deepseek_api_base = deepseek_api_base
        self.deepseek_api_url = f"{deepseek_api_base}/chat/completions"
        
        print(f"LLM Guard API URL: {self.guard_api_url}")
        print(f"DeepSeek API URL: {self.deepseek_api_url}")
        
    def is_deepseek_configured(self):
        """Проверяет, настроена ли конфигурация DeepSeek API"""
        return bool(self.deepseek_api_key)


class ApiWorkflowTester:
    """Тестер рабочего процесса с использованием API"""
    
    def __init__(self, api_config):
        """
        Инициализация тестера.
        
        Args:
            api_config: Конфигурация API
        """
        self.api_config = api_config
        
    def obfuscate_via_api(self, prompt):
        """
        Обфусцирует код в промпте, используя API LLM Guard.
        
        Args:
            prompt: Исходный промпт
            
        Returns:
            str: Обфусцированный промпт
            dict: Метаданные
        """
        url = f"{self.api_config.guard_api_url}/v1/scan_prompt"
        payload = {"prompt": prompt}
        headers = {"Content-Type": "application/json"}
        
        try:
            print(f"Отправка запроса на обфускацию к {url}")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return data["prompt"], data.get("metadata", {})
        except Exception as e:
            print(f"Ошибка при обфускации: {e}")
            return prompt, {"error": str(e)}
    
    def deobfuscate_via_api(self, prompt):
        """
        Деобфусцирует код в промпте, используя API LLM Guard.
        
        Args:
            prompt: Обфусцированный промпт
            
        Returns:
            str: Деобфусцированный промпт
        """
        url = f"{self.api_config.guard_api_url}/v1/deobfuscate"
        payload = {"prompt": prompt}
        headers = {"Content-Type": "application/json"}
        
        try:
            print(f"Отправка запроса на деобфускацию к {url}")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return data["prompt"]
        except Exception as e:
            print(f"Ошибка при деобфускации: {e}")
            return prompt
    
    def call_deepseek(self, prompt):
        """
        Отправляет запрос к DeepSeek API и получает ответ.
        
        Args:
            prompt: Текст запроса
            
        Returns:
            str: Ответ от DeepSeek
        """
        if not self.api_config.is_deepseek_configured():
            print("DeepSeek API не сконфигурирован. Возвращается тестовый ответ.")
            return "Вот улучшенная версия функции:\n\n```python\nimport re\n\ndef validate_email(email):\n    \"\"\"Validate an email address using regex and additional checks.\"\"\"\n    if not email or len(email) > 320:  # Max email length\n        return False\n        \n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))\n```"
        
        headers = {
            "Authorization": f"Bearer {self.api_config.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.api_config.deepseek_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            print(f"Отправка запроса к DeepSeek API: {self.api_config.deepseek_api_url}")
            response = requests.post(self.api_config.deepseek_api_url, json=data, headers=headers)
            response.raise_for_status()
            
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"Ошибка при вызове DeepSeek API: {e}")
            return f"Ошибка при вызове DeepSeek API: {e}"
    
    def test_workflow(self, prompts=None):
        """
        Тестирует полный рабочий процесс с обфускацией через API и деобфускацией.
        
        Args:
            prompts: Список запросов для тестирования
            
        Returns:
            list: Результаты тестирования
        """
        if prompts is None:
            prompts = CODE_PROMPTS
        
        results = []
        
        for i, prompt_data in enumerate(prompts):
            print(f"Тестирование рабочего процесса: {i+1}/{len(prompts)}")
            prompt = prompt_data["prompt"]
            language = prompt_data.get("language", "unknown")
            
            try:
                # 1. Обфускация запроса через API
                print("Шаг 1: Обфускация запроса через API")
                start_time = time.time()
                obfuscated_prompt, metadata = self.obfuscate_via_api(prompt)
                obfuscation_time = time.time() - start_time
                
                # 2. Отправка обфусцированного запроса в DeepSeek
                print("Шаг 2: Отправка обфусцированного запроса в DeepSeek")
                deepseek_start_time = time.time()
                deepseek_response = self.call_deepseek(obfuscated_prompt)
                deepseek_time = time.time() - deepseek_start_time
                
                # 3. Деобфускация ответа через API
                print("Шаг 3: Деобфускация ответа через API")
                deobfuscation_start_time = time.time()
                deobfuscated_response = self.deobfuscate_via_api(deepseek_response)
                deobfuscation_time = time.time() - deobfuscation_start_time
                
                # 4. Сохранение результатов
                result = {
                    "id": i,
                    "language": language,
                    "original_prompt": prompt,
                    "obfuscated_prompt": obfuscated_prompt,
                    "deepseek_response": deepseek_response,
                    "deobfuscated_response": deobfuscated_response,
                    "obfuscation_time": obfuscation_time,
                    "deepseek_time": deepseek_time,
                    "deobfuscation_time": deobfuscation_time,
                    "metadata": metadata,
                    "success": True
                }
                
                # Сохраняем в отдельные файлы
                result_dir = RESULTS_DIR / f"test_{i}"
                result_dir.mkdir(exist_ok=True)
                
                with open(result_dir / "original_prompt.txt", "w", encoding="utf-8") as f:
                    f.write(prompt)
                
                with open(result_dir / "obfuscated_prompt.txt", "w", encoding="utf-8") as f:
                    f.write(obfuscated_prompt)
                
                with open(result_dir / "deepseek_response.txt", "w", encoding="utf-8") as f:
                    f.write(deepseek_response)
                
                with open(result_dir / "deobfuscated_response.txt", "w", encoding="utf-8") as f:
                    f.write(deobfuscated_response)
                
            except Exception as e:
                print(f"Ошибка при обработке теста {i}: {e}")
                result = {
                    "id": i,
                    "language": language,
                    "original_prompt": prompt,
                    "error": str(e),
                    "success": False
                }
            
            results.append(result)
            
            # Небольшая пауза между тестами
            time.sleep(0.5)
        
        # Сохраняем все результаты
        with open(RESULTS_DIR / "api_workflow_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results


def main():
    """Основная функция для запуска тестирования."""
    parser = argparse.ArgumentParser(description="Тестирование API-интеграции с LLM Guard и DeepSeek")
    parser.add_argument("--guard-url", type=str, default="http://localhost:8000", help="URL API LLM Guard")
    parser.add_argument("--deepseek-key", type=str, default=None, help="API ключ для DeepSeek")
    parser.add_argument("--deepseek-model", type=str, default="deepseek-chat", help="Модель DeepSeek")
    parser.add_argument("--deepseek-api", type=str, default="https://api.deepseek.com/v1", help="Базовый URL DeepSeek API")
    
    args = parser.parse_args()
    
    # Инициализируем конфигурацию API
    api_config = APIConfig(
        guard_api_url=args.guard_url,
        deepseek_api_key=args.deepseek_key,
        deepseek_model=args.deepseek_model,
        deepseek_api_base=args.deepseek_api
    )
    
    # Инициализируем тестер
    tester = ApiWorkflowTester(api_config)
    
    # Запускаем тестирование
    results = tester.test_workflow()
    
    # Выводим статистику
    success_count = sum(1 for r in results if r.get("success", False))
    total_count = len(results)
    
    print(f"\nТестирование рабочего процесса API завершено")
    print(f"Успешно обработано: {success_count}/{total_count} примеров")
    
    if success_count > 0:
        successful_results = [r for r in results if r.get("success", False)]
        avg_obfuscation_time = sum(r["obfuscation_time"] for r in successful_results) / success_count
        avg_deepseek_time = sum(r["deepseek_time"] for r in successful_results) / success_count
        avg_deobfuscation_time = sum(r["deobfuscation_time"] for r in successful_results) / success_count
        
        print(f"Среднее время обфускации через API: {avg_obfuscation_time:.4f} сек.")
        print(f"Среднее время запроса к DeepSeek: {avg_deepseek_time:.4f} сек.")
        print(f"Среднее время деобфускации через API: {avg_deobfuscation_time:.4f} сек.")
    
    print(f"Результаты сохранены в {RESULTS_DIR}")


if __name__ == "__main__":
    main() 