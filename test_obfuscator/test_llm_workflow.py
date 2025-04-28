#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования полного рабочего процесса:
1. Обфускация кода
2. Отправка обфусцированного кода в LLM
3. Получение ответа от LLM
4. Деобфускация ответа

Этот скрипт использует CodeCipherObfuscator для обфускации/деобфускации
и интегрируется с LLM для полного сквозного тестирования.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
import argparse
from tqdm import tqdm

# Добавляем корневую директорию проекта в sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Импортируем CodeCipherObfuscator
from llm_guard.input_scanners import CodeCipherObfuscator
from llm_guard.util import get_logger

# Директории
TEST_DIR = Path(__file__).parent
RESULTS_DIR = TEST_DIR / "llm_workflow_results"
RESULTS_DIR.mkdir(exist_ok=True)
VAULT_DIR = TEST_DIR / "vault"
VAULT_DIR.mkdir(exist_ok=True)

# Настройка логгера
logger = get_logger()

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

# Конфигурация LLM
class LLMConfig:
    def __init__(self, api_key=None, model="deepseek-chat", api_base="https://api.deepseek.com/v1", provider="deepseek"):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.api_base = api_base
        
        if self.provider == "deepseek":
            self.api_url = f"{self.api_base}/chat/completions"
        elif self.provider == "openai":
            self.api_url = "https://api.openai.com/v1/chat/completions"
        else:
            raise ValueError(f"Неподдерживаемый провайдер: {self.provider}")
        
        if not self.api_key:
            logger.warning("API ключ не найден. Укажите api_key в аргументах.")
    
    def is_configured(self):
        return bool(self.api_key)


class WorkflowTester:
    def __init__(self, model_name="gpt2", llm_config=None):
        """
        Инициализация тестера рабочего процесса.
        
        Args:
            model_name: Имя модели для CodeCipherObfuscator
            llm_config: Конфигурация LLM
        """
        self.model_name = model_name
        self.llm_config = llm_config or LLMConfig()
        self.obfuscator = None
        
        self.initialize_obfuscator()
    
    def initialize_obfuscator(self):
        """Инициализирует обфускатор кода."""
        try:
            self.obfuscator = CodeCipherObfuscator(
                model_name=self.model_name,
                max_training_iterations=30,
                learning_rate=0.03,
                perplexity_threshold=100.0,
                early_stopping_patience=3,
                vault_dir=str(VAULT_DIR),
                skip_patterns=["# COPYRIGHT", "# DO NOT OBFUSCATE"]
            )
            logger.info(f"Инициализирован обфускатор с моделью {self.model_name}")
        except Exception as e:
            logger.error(f"Ошибка при инициализации обфускатора: {e}")
            raise
    
    def call_llm(self, prompt):
        """
        Отправляет запрос к LLM и получает ответ.
        
        Args:
            prompt: Текст запроса
            
        Returns:
            str: Ответ от LLM
        """
        if not self.llm_config.is_configured():
            logger.warning("LLM не сконфигурирован. Возвращается тестовый ответ.")
            return "Вот улучшенная версия функции:\n\n```python\nimport re\n\ndef validate_email(email):\n    \"\"\"Validate an email address using regex and additional checks.\"\"\"\n    if not email or len(email) > 320:  # Max email length\n        return False\n        \n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))\n```"
        
        headers = {
            "Authorization": f"Bearer {self.llm_config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.llm_config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            logger.info(f"Отправка запроса к {self.llm_config.provider} API: {self.llm_config.api_url}")
            response = requests.post(self.llm_config.api_url, json=data, headers=headers)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Обработка ответа в зависимости от провайдера
            if self.llm_config.provider == "deepseek":
                return response_data["choices"][0]["message"]["content"]
            elif self.llm_config.provider == "openai":
                return response_data["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Неподдерживаемый провайдер: {self.llm_config.provider}")
            
        except Exception as e:
            logger.error(f"Ошибка при вызове LLM: {e}")
            return f"Ошибка при вызове LLM: {e}"
    
    def test_workflow(self, prompts=None):
        """
        Тестирует полный рабочий процесс с обфускацией и деобфускацией.
        
        Args:
            prompts: Список запросов для тестирования
            
        Returns:
            list: Результаты тестирования
        """
        if prompts is None:
            prompts = CODE_PROMPTS
        
        results = []
        
        for i, prompt_data in enumerate(tqdm(prompts, desc="Тестирование рабочего процесса")):
            prompt = prompt_data["prompt"]
            language = prompt_data.get("language", "unknown")
            
            try:
                # 1. Обфускация запроса
                start_time = time.time()
                obfuscated_prompt, metadata = self.obfuscator.scan(prompt)
                obfuscation_time = time.time() - start_time
                
                # 2. Отправка обфусцированного запроса в LLM
                llm_start_time = time.time()
                llm_response = self.call_llm(obfuscated_prompt)
                llm_time = time.time() - llm_start_time
                
                # 3. Деобфускация ответа
                deobfuscation_start_time = time.time()
                deobfuscated_response = self.obfuscator.deobfuscate(llm_response)
                deobfuscation_time = time.time() - deobfuscation_start_time
                
                # 4. Сохранение результатов
                result = {
                    "id": i,
                    "language": language,
                    "original_prompt": prompt,
                    "obfuscated_prompt": obfuscated_prompt,
                    "llm_response": llm_response,
                    "deobfuscated_response": deobfuscated_response,
                    "obfuscation_time": obfuscation_time,
                    "llm_time": llm_time,
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
                
                with open(result_dir / "llm_response.txt", "w", encoding="utf-8") as f:
                    f.write(llm_response)
                
                with open(result_dir / "deobfuscated_response.txt", "w", encoding="utf-8") as f:
                    f.write(deobfuscated_response)
                
            except Exception as e:
                logger.error(f"Ошибка при обработке теста {i}: {e}")
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
        with open(RESULTS_DIR / "workflow_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results


def main():
    """Основная функция для запуска тестирования."""
    parser = argparse.ArgumentParser(description="Тестирование полного рабочего процесса обфускации")
    parser.add_argument("--model", type=str, default="gpt2", help="Имя модели для CodeCipherObfuscator")
    parser.add_argument("--api-key", type=str, default=None, help="API ключ для LLM")
    parser.add_argument("--llm-model", type=str, default="deepseek-chat", help="Модель LLM")
    parser.add_argument("--api-base", type=str, default="https://api.deepseek.com/v1", help="Базовый URL API")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "openai"], help="Провайдер LLM API")
    
    args = parser.parse_args()
    
    # Инициализируем конфигурацию LLM
    llm_config = LLMConfig(
        api_key=args.api_key,
        model=args.llm_model,
        api_base=args.api_base,
        provider=args.provider
    )
    
    # Инициализируем тестер
    tester = WorkflowTester(model_name=args.model, llm_config=llm_config)
    
    # Запускаем тестирование
    results = tester.test_workflow()
    
    # Выводим статистику
    success_count = sum(1 for r in results if r.get("success", False))
    total_count = len(results)
    
    print(f"\nТестирование рабочего процесса завершено")
    print(f"Успешно обработано: {success_count}/{total_count} примеров")
    
    if success_count > 0:
        successful_results = [r for r in results if r.get("success", False)]
        avg_obfuscation_time = sum(r["obfuscation_time"] for r in successful_results) / success_count
        avg_llm_time = sum(r["llm_time"] for r in successful_results) / success_count
        avg_deobfuscation_time = sum(r["deobfuscation_time"] for r in successful_results) / success_count
        
        print(f"Среднее время обфускации: {avg_obfuscation_time:.4f} сек.")
        print(f"Среднее время запроса к LLM: {avg_llm_time:.4f} сек.")
        print(f"Среднее время деобфускации: {avg_deobfuscation_time:.4f} сек.")
    
    print(f"Результаты сохранены в {RESULTS_DIR}")


if __name__ == "__main__":
    main() 