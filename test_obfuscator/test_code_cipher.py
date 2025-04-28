#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования функциональности обфускации и деобфускации
с использованием CodeCipherObfuscator.

Этот скрипт загружает тестовые данные, созданные generate_tests.py, 
и проверяет, как работает обфускация и деобфускация на различных языках программирования.
"""

import os
import json
import sys
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем корневую директорию проекта в sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from llm_guard.input_scanners import CodeCipherObfuscator
from llm_guard.util import get_logger

# Настройка логгера
logger = get_logger()

# Директории для тестирования
TEST_DIR = Path(__file__).parent
TESTS_DATA_DIR = TEST_DIR / "test_data"
RESULTS_DIR = TEST_DIR / "results"
VAULT_DIR = TEST_DIR / "vault"

# Создаем директории, если они не существуют
TESTS_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
VAULT_DIR.mkdir(exist_ok=True)

class ObfuscationTester:
    """Класс для тестирования обфускации и деобфускации кода."""
    
    def __init__(self, model_name="gpt2"):
        """
        Инициализация тестера.
        
        Args:
            model_name: Имя модели для использования в CodeCipherObfuscator
        """
        self.model_name = model_name
        self.scanner = None
        self.initialize_scanner()
        
    def initialize_scanner(self):
        """Инициализирует сканер CodeCipherObfuscator."""
        try:
            self.scanner = CodeCipherObfuscator(
                model_name=self.model_name,
                max_training_iterations=30,  # Уменьшаем для быстрого тестирования
                learning_rate=0.03,
                perplexity_threshold=100.0,
                early_stopping_patience=3,
                vault_dir=str(VAULT_DIR),
                skip_patterns=["# COPYRIGHT", "# DO NOT OBFUSCATE"]
            )
            logger.info(f"Инициализирован сканер с моделью {self.model_name}")
        except Exception as e:
            logger.error(f"Ошибка при инициализации сканера: {e}")
            raise
    
    def load_test_data(self, limit=None):
        """
        Загружает тестовые данные.
        
        Args:
            limit: Максимальное количество тестов для загрузки
            
        Returns:
            list: Список тестовых примеров
        """
        test_data_path = TESTS_DATA_DIR / "test_cases.json"
        
        if not test_data_path.exists():
            logger.warning(f"Файл с тестовыми данными не найден: {test_data_path}")
            return []
        
        try:
            with open(test_data_path, "r", encoding="utf-8") as f:
                test_cases = json.load(f)
            
            if limit:
                test_cases = test_cases[:limit]
                
            logger.info(f"Загружено {len(test_cases)} тестовых примеров")
            return test_cases
        except Exception as e:
            logger.error(f"Ошибка при загрузке тестовых данных: {e}")
            return []
    
    def test_obfuscation(self, test_cases, sample_size=None):
        """
        Тестирует обфускацию на предоставленных тестовых примерах.
        
        Args:
            test_cases: Список тестовых примеров
            sample_size: Размер выборки для тестирования
            
        Returns:
            dict: Результаты тестирования
        """
        if not test_cases:
            logger.warning("Нет тестовых примеров для тестирования")
            return {}
        
        # Если указан размер выборки, берем случайную выборку
        if sample_size and sample_size < len(test_cases):
            import random
            test_cases = random.sample(test_cases, sample_size)
        
        results = []
        
        for test_case in tqdm(test_cases, desc="Тестирование обфускации"):
            # Создаем входной промпт с кодом в блоке markdown
            code = test_case["code"]
            language = test_case["language"]
            
            # Обрабатываем пустые значения
            if not code or not code.strip():
                logger.warning(f"Пустой код в тесте {test_case['id']}, пропускаем")
                continue
            
            # Формируем промпт с блоком кода
            prompt = f"```{language}\n{code}\n```"
            
            try:
                # Замеряем время обфускации
                start_time = time.time()
                obfuscated_prompt, metadata = self.scanner.scan(prompt)
                obfuscation_time = time.time() - start_time
                
                # Извлекаем обфусцированный код из промпта
                import re
                obfuscated_code_match = re.search(r"```.*?\n(.*?)\n```", obfuscated_prompt, re.DOTALL)
                obfuscated_code = obfuscated_code_match.group(1) if obfuscated_code_match else ""
                
                # Замеряем время деобфускации
                start_time = time.time()
                deobfuscated_prompt = self.scanner.deobfuscate(obfuscated_prompt)
                deobfuscation_time = time.time() - start_time
                
                # Извлекаем деобфусцированный код из промпта
                deobfuscated_code_match = re.search(r"```.*?\n(.*?)\n```", deobfuscated_prompt, re.DOTALL)
                deobfuscated_code = deobfuscated_code_match.group(1) if deobfuscated_code_match else ""
                
                # Вычисляем метрики
                original_length = len(code)
                obfuscated_length = len(obfuscated_code)
                
                # Расчет метрики различия (расстояние Левенштейна нормализованное)
                from Levenshtein import distance as levenshtein_distance
                
                obfuscation_diff = levenshtein_distance(code, obfuscated_code) / max(1, original_length)
                deobfuscation_diff = levenshtein_distance(code, deobfuscated_code) / max(1, original_length)
                
                # Сохраняем результаты
                result = {
                    "id": test_case["id"],
                    "language": language,
                    "task_type": test_case.get("task_type", "unknown"),
                    "original_length": original_length,
                    "obfuscated_length": obfuscated_length,
                    "obfuscation_diff": obfuscation_diff,
                    "deobfuscation_diff": deobfuscation_diff,
                    "obfuscation_time": obfuscation_time,
                    "deobfuscation_time": deobfuscation_time,
                    "blocks_found": metadata["stats"]["code_blocks_found"],
                    "blocks_obfuscated": metadata["stats"]["code_blocks_obfuscated"],
                    "blocks_skipped": metadata["stats"]["skipped_blocks"],
                    "success": True
                }
                
                # Сохраняем примеры кода для проверки
                if test_case["id"] < 5:  # Сохраняем только первые 5 примеров для наглядности
                    sample_dir = RESULTS_DIR / f"sample_{test_case['id']}"
                    sample_dir.mkdir(exist_ok=True)
                    
                    with open(sample_dir / "original.txt", "w", encoding="utf-8") as f:
                        f.write(code)
                    
                    with open(sample_dir / "obfuscated.txt", "w", encoding="utf-8") as f:
                        f.write(obfuscated_code)
                    
                    with open(sample_dir / "deobfuscated.txt", "w", encoding="utf-8") as f:
                        f.write(deobfuscated_code)
                
            except Exception as e:
                logger.error(f"Ошибка при обработке теста {test_case['id']}: {e}")
                result = {
                    "id": test_case["id"],
                    "language": language,
                    "task_type": test_case.get("task_type", "unknown"),
                    "error": str(e),
                    "success": False
                }
            
            results.append(result)
            
            # Небольшая пауза между тестами
            time.sleep(0.1)
        
        # Сохраняем результаты в JSON
        with open(RESULTS_DIR / "obfuscation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def analyze_results(self, results):
        """
        Анализирует результаты тестирования и создает отчеты.
        
        Args:
            results: Результаты тестирования
            
        Returns:
            dict: Статистика по результатам
        """
        if not results:
            logger.warning("Нет результатов для анализа")
            return {}
        
        # Фильтруем успешные результаты
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        logger.info(f"Успешных тестов: {len(successful_results)}/{len(results)} ({len(successful_results)/max(1, len(results))*100:.2f}%)")
        
        if not successful_results:
            logger.warning("Нет успешных результатов для анализа")
            return {
                "total": len(results),
                "success_rate": 0,
                "failed": len(failed_results)
            }
        
        # Создаем DataFrame для анализа
        df = pd.DataFrame(successful_results)
        
        # Базовая статистика
        stats = {
            "total": len(results),
            "successful": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "failed": len(failed_results),
            "avg_obfuscation_diff": df["obfuscation_diff"].mean(),
            "avg_deobfuscation_diff": df["deobfuscation_diff"].mean(),
            "avg_obfuscation_time": df["obfuscation_time"].mean(),
            "avg_deobfuscation_time": df["deobfuscation_time"].mean()
        }
        
        # Статистика по языкам
        language_stats = df.groupby("language").agg({
            "obfuscation_diff": "mean",
            "deobfuscation_diff": "mean",
            "obfuscation_time": "mean",
            "deobfuscation_time": "mean",
            "id": "count"
        }).rename(columns={"id": "count"}).reset_index()
        
        # Сохраняем статистику
        language_stats.to_csv(RESULTS_DIR / "language_stats.csv", index=False)
        
        # Статистика по типам задач
        task_stats = df.groupby("task_type").agg({
            "obfuscation_diff": "mean",
            "deobfuscation_diff": "mean",
            "id": "count"
        }).rename(columns={"id": "count"}).reset_index()
        
        task_stats.to_csv(RESULTS_DIR / "task_stats.csv", index=False)
        
        # Создаем визуализации
        self.create_visualizations(df)
        
        # Сводная информация
        with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Средняя разница при обфускации: {stats['avg_obfuscation_diff']:.4f}")
        logger.info(f"Средняя разница при деобфускации: {stats['avg_deobfuscation_diff']:.4f}")
        logger.info(f"Среднее время обфускации: {stats['avg_obfuscation_time']:.4f} сек.")
        logger.info(f"Среднее время деобфускации: {stats['avg_deobfuscation_time']:.4f} сек.")
        
        return stats
    
    def create_visualizations(self, df):
        """
        Создает визуализации результатов.
        
        Args:
            df: DataFrame с результатами
        """
        # Устанавливаем стиль
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 10))
        
        # 1. Распределение разницы при обфускации по языкам
        plt.subplot(2, 2, 1)
        sns.boxplot(x="language", y="obfuscation_diff", data=df)
        plt.title("Разница при обфускации по языкам")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 2. Распределение времени обфускации по языкам
        plt.subplot(2, 2, 2)
        sns.boxplot(x="language", y="obfuscation_time", data=df)
        plt.title("Время обфускации по языкам")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 3. Сравнение обфускации и деобфускации
        plt.subplot(2, 2, 3)
        
        # Создаем данные для графика
        comparison_data = pd.melt(
            df[["obfuscation_diff", "deobfuscation_diff"]],
            var_name="Тип", value_name="Разница"
        )
        
        sns.boxplot(x="Тип", y="Разница", data=comparison_data)
        plt.title("Сравнение обфускации и деобфускации")
        plt.tight_layout()
        
        # 4. Распределение по типам задач
        plt.subplot(2, 2, 4)
        task_means = df.groupby("task_type")["obfuscation_diff"].mean().sort_values(ascending=False)
        
        # Берем топ-10 типов задач для наглядности
        top_tasks = task_means.head(10)
        
        sns.barplot(x=top_tasks.values, y=top_tasks.index)
        plt.title("Средняя разница при обфускации по типам задач (топ-10)")
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig(RESULTS_DIR / "obfuscation_visualizations.png", dpi=300, bbox_inches="tight")
        
        # Создаем диаграмму рассеяния для размера кода и разницы обфускации
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="original_length", 
            y="obfuscation_diff", 
            hue="language", 
            data=df, 
            alpha=0.7
        )
        plt.title("Зависимость качества обфускации от размера кода")
        plt.xlabel("Размер оригинального кода (символы)")
        plt.ylabel("Разница при обфускации")
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig(RESULTS_DIR / "size_vs_obfuscation.png", dpi=300, bbox_inches="tight")
        
        logger.info("Визуализации сохранены в директорию results")


def main():
    """Основная функция для запуска тестирования."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование обфускации и деобфускации кода")
    parser.add_argument("--model", type=str, default="gpt2", help="Имя модели для CodeCipherObfuscator")
    parser.add_argument("--limit", type=int, default=None, help="Максимальное количество тестов для загрузки")
    parser.add_argument("--sample", type=int, default=None, help="Размер выборки для тестирования")
    
    args = parser.parse_args()
    
    # Инициализируем тестер
    tester = ObfuscationTester(model_name=args.model)
    
    # Загружаем тестовые данные
    test_cases = tester.load_test_data(limit=args.limit)
    
    if not test_cases:
        print("Нет тестовых данных. Сначала запустите generate_tests.py")
        return
    
    # Запускаем тестирование обфускации
    results = tester.test_obfuscation(test_cases, sample_size=args.sample)
    
    # Анализируем результаты
    stats = tester.analyze_results(results)
    
    print(f"\nТестирование завершено. Результаты сохранены в {RESULTS_DIR}")
    print(f"Успешно обработано: {stats.get('successful', 0)}/{stats.get('total', 0)} примеров")
    print(f"Средняя разница при обфускации: {stats.get('avg_obfuscation_diff', 0):.4f}")
    print(f"Средняя разница при деобфускации: {stats.get('avg_deobfuscation_diff', 0):.4f}")


if __name__ == "__main__":
    main() 