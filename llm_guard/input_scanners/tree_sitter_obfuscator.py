#!/usr/bin/env python3
"""
Обфускатор кода с использованием Tree-sitter для анализа кода и CodeCipher для обфускации.
"""

import logging
import os
import re
from typing import List, Dict, Tuple, Set, Optional, Any, Union

from llm_guard.input_scanners.code_cipher import CodeCipherObfuscator, CodeCipherConfig
from llm_guard.input_scanners.tree_sitter_base import TREE_SITTER_AVAILABLE
from llm_guard.input_scanners.tree_sitter_collectors import get_collector_for_language
from llm_guard.util import detect_language

logger = logging.getLogger(__name__)

class TreeSitterCodeCipherObfuscator(CodeCipherObfuscator):
    """
    Обфускатор кода с использованием Tree-sitter для анализа структуры кода.
    
    Расширяет возможности CodeCipherObfuscator путем анализа структуры кода
    с помощью Tree-sitter и более точного определения идентификаторов,
    которые следует обфусцировать.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        training_iterations: int = 1,
        learning_rate: float = 0.001,
        use_tree_sitter: bool = True,
        **kwargs
    ):
        """
        Инициализирует обфускатор кода с использованием Tree-sitter.
        
        Args:
            model_name: Название модели для обфускации
            training_iterations: Количество итераций обучения
            learning_rate: Скорость обучения
            use_tree_sitter: Использовать ли Tree-sitter для анализа кода
            **kwargs: Дополнительные параметры
        """
        super().__init__(
            model_name=model_name,
            training_iterations=training_iterations,
            learning_rate=learning_rate,
            use_antlr=False,  # Tree-sitter заменяет ANTLR
            **kwargs
        )
        self.use_tree_sitter = use_tree_sitter and TREE_SITTER_AVAILABLE
        
        if self.use_tree_sitter:
            logger.info("Tree-sitter доступен и будет использоваться для анализа кода")
        else:
            logger.warning(
                "Tree-sitter недоступен. Будет использоваться простой анализ на основе регулярных выражений. "
                "Установите Tree-sitter с помощью 'setup_tree_sitter.py' для полноценного анализа кода."
            )
    
    def _obfuscate_with_tree_sitter(self, code: str, language: str) -> Tuple[str, Set[str]]:
        """
        Обфусцирует код с использованием Tree-sitter для анализа структуры.
        
        Args:
            code: Исходный код
            language: Язык программирования
            
        Returns:
            Tuple[str, Set[str]]: Кортеж из исходного кода и множества элементов, которые не должны обфусцироваться
        """
        collector = get_collector_for_language(language)
        if not collector:
            logger.warning(f"Коллектор для языка {language} не найден. Использую стандартный метод обфускации.")
            return code, set()
        
        # Анализируем код с помощью Tree-sitter
        collector.collect(code)
        
        # Получаем множества различных элементов кода
        non_obfuscatable = set()
        
        # Добавляем ключевые слова языка
        non_obfuscatable.update(collector.get_keywords())
        
        # Добавляем импорты
        non_obfuscatable.update(collector.get_imports())
        
        # Добавляем строковые литералы
        non_obfuscatable.update(collector.get_string_literals())
        
        # Добавляем числовые литералы
        non_obfuscatable.update(collector.get_numeric_literals())
        
        # Получаем статистику для логирования
        stats = collector.get_statistics()
        logger.info(f"Tree-sitter анализ кода на языке {language}:")
        logger.info(f"  Всего идентификаторов: {stats['total_identifiers']}")
        logger.info(f"  Классы: {stats['classes']}")
        logger.info(f"  Функции: {stats['functions']}")
        logger.info(f"  Переменные: {stats['variables']}")
        logger.info(f"  Импорты: {stats['imports']}")
        logger.info(f"  Строковые литералы: {stats['string_literals']}")
        logger.info(f"  Числовые литералы: {stats['numeric_literals']}")
        
        # Возвращаем исходный код и множество элементов, которые не должны обфусцироваться
        return code, non_obfuscatable
    
    def _obfuscate_code(self, code: str) -> Tuple[str, Dict, Dict]:
        """
        Обфусцирует код, используя Tree-sitter для анализа структуры (если доступен).
        
        Args:
            code: Исходный код
            
        Returns:
            Tuple[str, Dict, Dict]: Кортеж из обфусцированного кода, метаданных и маппинга обфускации
        """
        # Определяем язык кода
        language = detect_language(code)
        
        # Если Tree-sitter доступен и язык поддерживается, используем его для анализа кода
        non_obfuscatable = set()
        if self.use_tree_sitter and language:
            _, non_obfuscatable = self._obfuscate_with_tree_sitter(code, language)
        
        # Используем родительский метод для обфускации, передавая множество non_obfuscatable
        self.config = CodeCipherConfig(non_obfuscatable=non_obfuscatable)
        return super()._obfuscate_code(code)
    
    def scan(self, prompt: str) -> Tuple[str, Dict]:
        """
        Сканирует и обфусцирует код в промпте.
        
        Args:
            prompt: Текст промпта
            
        Returns:
            Tuple[str, Dict]: Обфусцированный промпт и метаданные
        """
        return super().scan(prompt) 