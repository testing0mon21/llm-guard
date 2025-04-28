#!/usr/bin/env python3
"""
Базовые классы для работы с Tree-sitter в CodeCipherObfuscator.
"""

import os
from pathlib import Path
import logging
from typing import Set, Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

# Путь к скомпилированной библиотеке с грамматиками
GRAMMARS_DIR = Path("llm_guard/input_scanners/tree_sitter_grammars")
LANGUAGES_LIB = GRAMMARS_DIR / "languages.so"

# Проверяем доступность Tree-sitter
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_IMPORT_AVAILABLE = True
except ImportError:
    logger.warning("Не удалось импортировать tree_sitter. Установите модуль: pip install tree-sitter")
    TREE_SITTER_IMPORT_AVAILABLE = False

# Проверяем, доступна ли библиотека
TREE_SITTER_AVAILABLE = TREE_SITTER_IMPORT_AVAILABLE and LANGUAGES_LIB.exists()

if not TREE_SITTER_AVAILABLE:
    if not LANGUAGES_LIB.exists():
        logger.warning("Библиотека с грамматиками Tree-sitter не найдена. Запустите скрипт установки.")
    
    # Создаем заглушки для классов
    class DummyParser:
        def set_language(self, *args, **kwargs):
            pass
        
        def parse(self, *args, **kwargs):
            return None
    
    class DummyLanguage:
        pass
    
    # Если Tree-sitter недоступен, создаем заглушки
    Language = DummyLanguage
    Parser = DummyParser
    TREE_SITTER_LANGUAGES = {}
else:
    # Загружаем языки
    TREE_SITTER_LANGUAGES = {}
    try:
        for lang in ["python", "javascript", "java", "csharp", "kotlin", "php", "ruby", "swift", "go", "rust"]:
            try:
                TREE_SITTER_LANGUAGES[lang] = Language(str(LANGUAGES_LIB), lang)
            except Exception as e:
                logger.warning(f"Не удалось загрузить грамматику для языка {lang}: {e}")
        
        if TREE_SITTER_LANGUAGES:
            logger.info(f"Загружены грамматики Tree-sitter для языков: {', '.join(TREE_SITTER_LANGUAGES.keys())}")
        else:
            logger.warning("Не удалось загрузить ни одной грамматики Tree-sitter")
    except Exception as e:
        logger.error(f"Ошибка при загрузке грамматик Tree-sitter: {e}")
        TREE_SITTER_AVAILABLE = False

class BaseTreeSitterCollector:
    """Базовый класс для сбора идентификаторов с помощью Tree-sitter"""
    
    def __init__(self, language: str):
        """
        Инициализация коллектора.
        
        Args:
            language: Название языка программирования
        """
        self.language = language
        self.parser = Parser()
        
        if TREE_SITTER_AVAILABLE and language in TREE_SITTER_LANGUAGES:
            self.parser.set_language(TREE_SITTER_LANGUAGES[language])
        
        # Множества для хранения информации о коде
        self.identifiers = set()  # Все идентификаторы
        self.functions = set()    # Имена функций
        self.classes = set()      # Имена классов
        self.variables = set()    # Имена переменных
        self.imports = set()      # Импорты
        self.string_literals = set()  # Строковые литералы
        self.numeric_literals = set() # Числовые литералы
        self.keywords = self._get_language_keywords()  # Ключевые слова языка
    
    def _get_language_keywords(self) -> Set[str]:
        """
        Возвращает множество ключевых слов для данного языка.
        
        Returns:
            Set[str]: Множество ключевых слов
        """
        # Базовая реализация, переопределяется в конкретных классах
        return set()
    
    def parse(self, code: str) -> Any:
        """
        Парсит код с помощью Tree-sitter.
        
        Args:
            code: Исходный код
            
        Returns:
            Дерево разбора
        """
        if not TREE_SITTER_AVAILABLE:
            return None
        
        try:
            return self.parser.parse(bytes(code, 'utf8'))
        except Exception as e:
            logger.error(f"Ошибка при парсинге кода с помощью Tree-sitter: {e}")
            return None
    
    def collect(self, code: str) -> None:
        """
        Собирает информацию о коде.
        
        Args:
            code: Исходный код
        """
        tree = self.parse(code)
        if tree:
            self._walk_tree(tree.root_node)
    
    def _walk_tree(self, node: Any) -> None:
        """
        Рекурсивный обход дерева разбора.
        
        Args:
            node: Узел дерева
        """
        # Базовая реализация, переопределяется в конкретных классах
        pass
    
    def get_all_non_obfuscatable(self) -> Set[str]:
        """
        Возвращает множество элементов, которые не должны быть обфусцированы.
        
        Returns:
            Set[str]: Множество необфусцируемых элементов
        """
        non_obfuscatable = set()
        non_obfuscatable.update(self.keywords)
        non_obfuscatable.update(self.imports)
        non_obfuscatable.update(self.string_literals)
        non_obfuscatable.update(self.numeric_literals)
        
        return non_obfuscatable
        
    def get_keywords(self) -> Set[str]:
        """
        Возвращает множество ключевых слов языка.
        
        Returns:
            Set[str]: Множество ключевых слов
        """
        return self.keywords
    
    def get_imports(self) -> Set[str]:
        """
        Возвращает множество импортов в коде.
        
        Returns:
            Set[str]: Множество импортов
        """
        return self.imports
    
    def get_string_literals(self) -> Set[str]:
        """
        Возвращает множество строковых литералов в коде.
        
        Returns:
            Set[str]: Множество строковых литералов
        """
        return self.string_literals
    
    def get_numeric_literals(self) -> Set[str]:
        """
        Возвращает множество числовых литералов в коде.
        
        Returns:
            Set[str]: Множество числовых литералов
        """
        return self.numeric_literals
    
    def get_classes(self) -> Set[str]:
        """
        Возвращает множество имен классов в коде.
        
        Returns:
            Set[str]: Множество имен классов
        """
        return self.classes
    
    def get_functions(self) -> Set[str]:
        """
        Возвращает множество имен функций в коде.
        
        Returns:
            Set[str]: Множество имен функций
        """
        return self.functions
    
    def get_variables(self) -> Set[str]:
        """
        Возвращает множество имен переменных в коде.
        
        Returns:
            Set[str]: Множество имен переменных
        """
        return self.variables
    
    def get_all_identifiers(self) -> Set[str]:
        """
        Возвращает множество всех идентификаторов в коде.
        
        Returns:
            Set[str]: Множество всех идентификаторов
        """
        return self.identifiers
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Возвращает статистику анализа кода.
        
        Returns:
            Dict[str, int]: Словарь со статистикой
        """
        return {
            "total_identifiers": len(self.identifiers),
            "classes": len(self.classes),
            "functions": len(self.functions),
            "variables": len(self.variables),
            "imports": len(self.imports),
            "string_literals": len(self.string_literals),
            "numeric_literals": len(self.numeric_literals)
        } 