#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования API LLM Guard с использованием тестовых данных.

Этот скрипт загружает тестовые данные из директории test_data и проводит
тестирование API LLM Guard для обфускации и деобфускации кода.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
import argparse
from tqdm import tqdm
import random

# Директории
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "test_data"
RESULTS_DIR = TEST_DIR / "api_dataset_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Конфигурация API
class APIConfig:
    """Конфигурация API LLM Guard."""
    
    def __init__(self, api_url="http://localhost:8000", session_id=None):
        """
        Инициализация конфигурации API.
        
        Args:
            api_url: URL API-сервиса LLM Guard
            session_id: ID сессии для отслеживания обфускации (опционально)
        """
        self.api_url = api_url.rstrip('/')
        self.session_id = session_id or f"test_session_{int(time.time())}"
        
        print(f"API URL: {self.api_url}")
        print(f"Session ID: {self.session_id}")


class APIClient:
    """Клиент для API LLM Guard."""
    
    def __init__(self, api_config):
        """
        Инициализация клиента API.
        
        Args:
            api_config: Конфигурация API
        """
        self.api_config = api_config
    
    def obfuscate(self, prompt):
        """
        Обфусцирует код в промпте через API.
        
        Args:
            prompt: Исходный промпт
            
        Returns:
            tuple: (обфусцированный_промпт, метаданные)
        """
        url = f"{self.api_config.api_url}/v1/scan_prompt"
        payload = {
            "prompt": prompt,
            "session_id": self.api_config.session_id
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return data.get("prompt", prompt), data.get("metadata", {})
        except Exception as e:
            print(f"Ошибка при обфускации: {e}")
            return prompt, {"error": str(e)}
    
    def deobfuscate(self, text):
        """
        Деобфусцирует текст через API.
        
        Args:
            text: Текст с обфусцированным кодом
            
        Returns:
            str: Деобфусцированный текст
        """
        url = f"{self.api_config.api_url}/v1/deobfuscate"
        payload = {
            "text": text,
            "session_id": self.api_config.session_id,
            "scanner": "CodeCipherObfuscator"
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return data.get("deobfuscated_text", text)
        except Exception as e:
            print(f"Ошибка при деобфускации: {e}")
            return text


class TestDatasetHandler:
    """Обработчик тестовых данных."""
    
    def __init__(self, api_client):
        """
        Инициализация обработчика.
        
        Args:
            api_client: Клиент API
        """
        self.api_client = api_client
        self.test_cases = []
    
    def load_test_data(self, limit=None, sample_size=None):
        """
        Загружает тестовые данные из JSON-файла.
        
        Args:
            limit: Ограничение на количество тестов
            sample_size: Размер случайной выборки
            
        Returns:
            list: Загруженные тестовые данные
        """
        test_cases_file = TEST_DATA_DIR / "test_cases.json"
        
        if not test_cases_file.exists():
            raise FileNotFoundError(f"Файл тестовых данных не найден: {test_cases_file}")
        
        with open(test_cases_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        
        print(f"Загружено {len(test_cases)} тестовых случаев")
        
        # Применяем ограничения
        if limit is not None:
            test_cases = test_cases[:limit]
            print(f"Применено ограничение: {limit} тестов")
        
        if sample_size is not None and sample_size < len(test_cases):
            test_cases = random.sample(test_cases, sample_size)
            print(f"Случайная выборка: {sample_size} тестов")
        
        self.test_cases = test_cases
        return test_cases
    
    def create_prompt_with_code(self, test_case):
        """
        Создает промпт с кодом для конкретного тестового случая.
        
        Args:
            test_case: Тестовый случай
            
        Returns:
            str: Промпт с кодом
        """
        language = test_case["language"]
        code = test_case["code"]
        prompt_text = test_case["prompt"]
        
        prompt = f"{prompt_text}\n\n```{language}\n{code}\n```\n\nПожалуйста, проанализируй этот код и предложи улучшения."
        return prompt
    
    def test_on_dataset(self, limit=None, sample_size=None):
        """
        Проводит тестирование на наборе данных.
        
        Args:
            limit: Ограничение на количество тестов
            sample_size: Размер случайной выборки
            
        Returns:
            list: Результаты тестирования
        """
        # Загружаем тестовые данные
        self.load_test_data(limit, sample_size)
        
        results = []
        
        for i, test_case in enumerate(tqdm(self.test_cases, desc="Тестирование API")):
            # Создаем промпт с кодом
            prompt = self.create_prompt_with_code(test_case)
            language = test_case["language"]
            
            try:
                # 1. Обфускация
                start_time = time.time()
                obfuscated_prompt, metadata = self.api_client.obfuscate(prompt)
                obfuscation_time = time.time() - start_time
                
                # 2. Генерируем искусственный "ответ LLM" (просто для демонстрации)
                mock_llm_response = f"Вот улучшенная версия кода:\n\n```{language}\n{test_case['code']}\n\n# Добавлены улучшения\ndef improved_function():\n    pass\n```"
                
                # 3. Деобфускация
                deobfuscation_start_time = time.time()
                deobfuscated_response = self.api_client.deobfuscate(mock_llm_response)
                deobfuscation_time = time.time() - deobfuscation_start_time
                
                # 4. Сохранение результатов
                result = {
                    "id": test_case["id"],
                    "language": language,
                    "task_type": test_case.get("task_type", "unknown"),
                    "original_prompt": prompt,
                    "obfuscated_prompt": obfuscated_prompt,
                    "mock_llm_response": mock_llm_response,
                    "deobfuscated_response": deobfuscated_response,
                    "obfuscation_time": obfuscation_time,
                    "deobfuscation_time": deobfuscation_time,
                    "metadata": metadata,
                    "success": True
                }
                
                # Сохраняем в отдельные файлы
                result_dir = RESULTS_DIR / f"{language}_test_{test_case['id']}"
                result_dir.mkdir(exist_ok=True)
                
                with open(result_dir / "original_prompt.txt", "w", encoding="utf-8") as f:
                    f.write(prompt)
                
                with open(result_dir / "obfuscated_prompt.txt", "w", encoding="utf-8") as f:
                    f.write(obfuscated_prompt)
                
                with open(result_dir / "mock_llm_response.txt", "w", encoding="utf-8") as f:
                    f.write(mock_llm_response)
                
                with open(result_dir / "deobfuscated_response.txt", "w", encoding="utf-8") as f:
                    f.write(deobfuscated_response)
                
            except Exception as e:
                print(f"Ошибка при обработке теста {test_case['id']}: {e}")
                result = {
                    "id": test_case["id"],
                    "language": language,
                    "original_prompt": prompt,
                    "error": str(e),
                    "success": False
                }
            
            results.append(result)
            
            # Небольшая пауза между тестами
            time.sleep(0.5)
        
        # Сохраняем все результаты
        with open(RESULTS_DIR / "api_dataset_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def analyze_results(self, results):
        """
        Анализирует результаты тестирования.
        
        Args:
            results: Результаты тестирования
            
        Returns:
            dict: Статистика по результатам
        """
        # Общая статистика
        total_count = len(results)
        success_count = sum(1 for r in results if r.get("success", False))
        
        # Статистика по языкам
        language_stats = {}
        for result in results:
            language = result.get("language", "unknown")
            success = result.get("success", False)
            
            if language not in language_stats:
                language_stats[language] = {"total": 0, "success": 0}
            
            language_stats[language]["total"] += 1
            if success:
                language_stats[language]["success"] += 1
        
        # Расчет средних времен
        avg_obfuscation_time = 0
        avg_deobfuscation_time = 0
        
        if success_count > 0:
            successful_results = [r for r in results if r.get("success", False)]
            avg_obfuscation_time = sum(r.get("obfuscation_time", 0) for r in successful_results) / success_count
            avg_deobfuscation_time = sum(r.get("deobfuscation_time", 0) for r in successful_results) / success_count
        
        # Формируем статистику
        stats = {
            "total_count": total_count,
            "success_count": success_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "language_stats": language_stats,
            "avg_obfuscation_time": avg_obfuscation_time,
            "avg_deobfuscation_time": avg_deobfuscation_time
        }
        
        # Сохраняем статистику
        with open(RESULTS_DIR / "api_dataset_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        return stats


def main():
    """Главная функция для запуска тестирования."""
    parser = argparse.ArgumentParser(description="Тестирование API LLM Guard на наборе данных")
    parser.add_argument("--api", type=str, default="http://localhost:8000", help="URL API LLM Guard")
    parser.add_argument("--session", type=str, default=None, help="ID сессии (опционально)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничение на количество тестов")
    parser.add_argument("--sample", type=int, default=None, help="Размер случайной выборки")
    
    args = parser.parse_args()
    
    # Инициализируем конфигурацию API
    api_config = APIConfig(api_url=args.api, session_id=args.session)
    
    # Инициализируем клиент API
    api_client = APIClient(api_config)
    
    # Инициализируем обработчик тестовых данных
    test_handler = TestDatasetHandler(api_client)
    
    # Запускаем тестирование
    results = test_handler.test_on_dataset(limit=args.limit, sample_size=args.sample)
    
    # Анализируем результаты
    stats = test_handler.analyze_results(results)
    
    # Выводим статистику
    print("\nРезультаты тестирования API на наборе данных:")
    print(f"Всего тестов: {stats['total_count']}")
    print(f"Успешно: {stats['success_count']} ({stats['success_rate']*100:.2f}%)")
    print(f"Среднее время обфускации: {stats['avg_obfuscation_time']:.4f} сек.")
    print(f"Среднее время деобфускации: {stats['avg_deobfuscation_time']:.4f} сек.")
    
    print("\nСтатистика по языкам:")
    for language, lang_stats in stats["language_stats"].items():
        success_rate = lang_stats["success"] / lang_stats["total"] if lang_stats["total"] > 0 else 0
        print(f"- {language}: {lang_stats['success']}/{lang_stats['total']} ({success_rate*100:.2f}%)")
    
    print(f"\nРезультаты сохранены в {RESULTS_DIR}")


if __name__ == "__main__":
    main() 