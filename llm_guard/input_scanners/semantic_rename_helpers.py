"""
Вспомогательные функции и классы для интеллектуального семантического переименования идентификаторов.

Модуль предоставляет инструменты для обфускации кода при сохранении семантических подсказок,
что позволяет LLM-моделям понимать код, но затрудняет его анализ для людей.
"""

import re
import random
from typing import Dict, List, Set, Tuple, Optional
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


class SemanticRenamer:
    """
    Класс для семантического переименования идентификаторов с сохранением смысловых подсказок.
    """

    # Стандартные шаблоны для разных стилей именования
    NAME_STYLES = {
        "snake_case": r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*",
        "camelCase": r"[a-z][a-zA-Z0-9]*",
        "PascalCase": r"[A-Z][a-zA-Z0-9]*",
        "UPPER_CASE": r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*",
    }

    # Суффиксы для типов данных
    TYPE_SUFFIXES = {
        "int": "_i",
        "integer": "_i",
        "float": "_f",
        "string": "_s",
        "str": "_s",
        "list": "_lst",
        "array": "_arr",
        "dict": "_dict",
        "dictionary": "_dict",
        "bool": "_b",
        "boolean": "_b",
        "object": "_obj",
        "function": "_fn",
        "class": "_cls",
    }

    # Префиксы для различных категорий идентификаторов
    CATEGORY_PREFIXES = {
        "count": ["cnt", "num", "n"],
        "index": ["idx", "i", "pos"],
        "temporary": ["tmp", "temp", "t"],
        "calculate": ["calc", "compute", "proc"],
        "process": ["proc", "handle", "exec"],
        "value": ["val", "v", "data"],
        "result": ["res", "ret", "output"],
        "parameter": ["param", "arg", "input"],
        "variable": ["var", "v", "x"],
        "collection": ["col", "set", "grp"],
        "manager": ["mgr", "ctrl", "handler"],
        "controller": ["ctrl", "ctl", "handler"],
        "service": ["svc", "srvc", "srv"],
        "repository": ["repo", "store", "db"],
        "utility": ["util", "helper", "tool"],
    }

    # Словарь доменно-специфичных сокращений для разных областей
    DOMAIN_ABBREVIATIONS = {
        "finance": {
            "account": "acc",
            "balance": "bal",
            "transaction": "tx",
            "payment": "pmt",
            "amount": "amt",
            "currency": "curr",
            "interest": "int",
            "deposit": "dep",
            "withdrawal": "wdw",
            "invoice": "inv",
        },
        "medical": {
            "patient": "pt",
            "diagnosis": "dx",
            "treatment": "tx",
            "prescription": "rx",
            "medical": "med",
            "hospital": "hosp",
            "appointment": "appt",
            "insurance": "ins",
            "laboratory": "lab",
            "symptom": "sym",
        },
        "web": {
            "request": "req",
            "response": "resp",
            "session": "sess",
            "cookie": "ck",
            "parameter": "param",
            "authenticate": "auth",
            "authorize": "authz",
            "database": "db",
            "document": "doc",
            "element": "elem",
        },
        "general": {
            "configuration": "cfg",
            "information": "info",
            "message": "msg",
            "error": "err",
            "warning": "warn",
            "success": "succ",
            "directory": "dir",
            "identifier": "id",
            "reference": "ref",
            "attribute": "attr",
        }
    }

    # Сопоставление общих английских слов с сокращениями
    COMMON_WORD_ABBREVIATIONS = {
        "application": "app",
        "button": "btn",
        "command": "cmd",
        "communication": "comm",
        "context": "ctx",
        "control": "ctrl",
        "coordinate": "coord",
        "destination": "dst",
        "directory": "dir",
        "environment": "env",
        "exception": "exc",
        "information": "info",
        "initialize": "init",
        "language": "lang",
        "library": "lib",
        "maximum": "max",
        "message": "msg",
        "minimum": "min",
        "object": "obj",
        "parameter": "param",
        "position": "pos",
        "previous": "prev",
        "reference": "ref",
        "repository": "repo",
        "request": "req",
        "response": "resp",
        "sequence": "seq",
        "source": "src",
        "standard": "std",
        "string": "str",
        "synchronize": "sync",
        "system": "sys",
        "temperature": "temp",
        "temporary": "tmp",
        "utility": "util",
        "value": "val",
        "window": "win",
    }

    def __init__(self, domain: str = "general", semantic_preservation_level: float = 0.7):
        """
        Инициализирует SemanticRenamer.
        
        Args:
            domain: Предметная область кода (finance, medical, web, general)
            semantic_preservation_level: Уровень сохранения семантики (0.0-1.0)
                где 0.0 - полная обфускация, 1.0 - максимальное сохранение семантики
        """
        self.domain = domain
        self.semantic_preservation_level = semantic_preservation_level
        self.identifier_cache = {}  # Кэш для консистентного переименования
        
    def _detect_name_style(self, identifier: str) -> str:
        """Определяет стиль именования идентификатора."""
        for style, pattern in self.NAME_STYLES.items():
            if re.fullmatch(pattern, identifier):
                return style
        return "snake_case"  # По умолчанию
    
    def _split_identifier(self, identifier: str) -> List[str]:
        """Разбивает идентификатор на составные части."""
        # Сначала определяем стиль
        style = self._detect_name_style(identifier)
        
        if style == "snake_case" or style == "UPPER_CASE":
            return identifier.lower().split('_')
        elif style == "camelCase":
            # Разбиваем camelCase
            return re.findall(r'[a-z][a-z0-9]*|[A-Z][a-z0-9]*', identifier)
        elif style == "PascalCase":
            # Разбиваем PascalCase
            words = re.findall(r'[A-Z][a-z0-9]*', identifier)
            return [word.lower() for word in words]
        else:
            return [identifier.lower()]

    def _generate_abbreviation(self, word: str) -> str:
        """Генерирует сокращение для слова."""
        # Проверяем, есть ли слово в словаре общих сокращений
        if word in self.COMMON_WORD_ABBREVIATIONS:
            return self.COMMON_WORD_ABBREVIATIONS[word]
            
        # Проверяем, есть ли слово в доменно-специфичных сокращениях
        if self.domain in self.DOMAIN_ABBREVIATIONS:
            if word in self.DOMAIN_ABBREVIATIONS[self.domain]:
                return self.DOMAIN_ABBREVIATIONS[self.domain][word]
        
        # Проверяем общие сокращения
        if word in self.DOMAIN_ABBREVIATIONS["general"]:
            return self.DOMAIN_ABBREVIATIONS["general"][word]
        
        # Если нет готового сокращения, создаем его
        if len(word) <= 3:
            return word  # Короткие слова не сокращаем
        
        # Используем первые буквы для сокращения
        if len(word) <= 6:
            # Для слов средней длины используем первые 2-3 символа
            length = min(3, len(word) - 1)
            return word[:length]
        else:
            # Для длинных слов используем согласные
            consonants = ''.join([c for c in word if c.lower() not in 'aeiou'])
            if len(consonants) >= 3:
                return consonants[:3]
            else:
                return word[:3]
    
    def _get_prefix_for_category(self, parts: List[str]) -> Optional[str]:
        """Определяет префикс на основе категории идентификатора."""
        for word in parts:
            for category, prefixes in self.CATEGORY_PREFIXES.items():
                if word == category or self._are_semantically_similar(word, category):
                    # Выбираем префикс в зависимости от уровня сохранения семантики
                    if self.semantic_preservation_level > 0.8:
                        return prefixes[0]  # Самый очевидный
                    elif self.semantic_preservation_level > 0.5:
                        return random.choice(prefixes[:2])  # Из первых двух
                    else:
                        return random.choice(prefixes)  # Любой
        return None
    
    def _get_type_suffix(self, identifier: str, context: str = "") -> Optional[str]:
        """Определяет суффикс типа данных на основе контекста использования."""
        for type_name, suffix in self.TYPE_SUFFIXES.items():
            if type_name in context.lower():
                return suffix
            
            # Эвристики для определения типа
            if type_name == "int" and re.search(r'count|index|num|size|len', identifier.lower()):
                return suffix
            elif type_name == "float" and re.search(r'price|rate|amount|value', identifier.lower()):
                return suffix
            elif type_name == "str" and re.search(r'name|title|text|message|description', identifier.lower()):
                return suffix
            elif type_name == "list" and re.search(r'list|array|items|elements', identifier.lower()):
                return suffix
            elif type_name == "dict" and re.search(r'map|dict|table|hash', identifier.lower()):
                return suffix
            elif type_name == "bool" and re.search(r'is|has|can|should|flag', identifier.lower()):
                return suffix
        
        return ""
    
    def _are_semantically_similar(self, word1: str, word2: str) -> bool:
        """Проверяет семантическую схожесть двух слов с помощью WordNet."""
        try:
            # Получаем синсеты для слов
            synsets1 = wordnet.synsets(word1)
            synsets2 = wordnet.synsets(word2)
            
            if not synsets1 or not synsets2:
                return False
            
            # Вычисляем максимальную схожесть
            max_similarity = 0.0
            for s1 in synsets1:
                for s2 in synsets2:
                    similarity = s1.path_similarity(s2)
                    if similarity and similarity > max_similarity:
                        max_similarity = similarity
            
            # Порог схожести
            return max_similarity > 0.5
        except:
            return False
    
    def _combine_parts(self, parts: List[str], style: str) -> str:
        """Комбинирует части идентификатора в соответствии с исходным стилем."""
        if style == "snake_case":
            return "_".join(parts)
        elif style == "UPPER_CASE":
            return "_".join([p.upper() for p in parts])
        elif style == "camelCase":
            return parts[0] + "".join([p.capitalize() for p in parts[1:]])
        elif style == "PascalCase":
            return "".join([p.capitalize() for p in parts])
        else:
            return "_".join(parts)
    
    def rename_identifier(self, identifier: str, context: str = "") -> str:
        """
        Выполняет семантическое переименование идентификатора с учетом контекста.
        
        Args:
            identifier: Исходный идентификатор
            context: Контекст использования (например, тип данных, окружающий код)
            
        Returns:
            Переименованный идентификатор с сохранением семантических подсказок
        """
        # Проверяем кэш для консистентного переименования
        if identifier in self.identifier_cache:
            return self.identifier_cache[identifier]
        
        # Определяем стиль именования
        style = self._detect_name_style(identifier)
        
        # Разбиваем идентификатор на части
        parts = self._split_identifier(identifier)
        
        # Применяем семантическое переименование к каждой части
        new_parts = []
        
        # Определяем префикс на основе категории
        prefix = self._get_prefix_for_category(parts)
        if prefix and random.random() < self.semantic_preservation_level:
            new_parts.append(prefix)
            
        # Для каждой значимой части создаем сокращение
        meaningful_parts = [p for p in parts if len(p) > 1]
        
        for part in meaningful_parts:
            # Шанс сохранить часть без изменений
            if random.random() < self.semantic_preservation_level * 0.5:
                new_parts.append(part)
            else:
                # Генерируем сокращение
                abbr = self._generate_abbreviation(part)
                new_parts.append(abbr)
                
        # Если после всех преобразований у нас пустой список, добавляем хотя бы одну часть
        if not new_parts and meaningful_parts:
            new_parts.append(self._generate_abbreviation(meaningful_parts[0]))
        
        # Добавляем суффикс типа, если можем его определить
        type_suffix = self._get_type_suffix(identifier, context)
        if type_suffix and random.random() < self.semantic_preservation_level:
            new_parts.append(type_suffix.lstrip("_"))
        
        # Соединяем части в соответствии с исходным стилем
        result = self._combine_parts(new_parts, style)
        
        # Если каким-то образом результат пустой, используем первые буквы исходного идентификатора
        if not result:
            if style in ["snake_case", "UPPER_CASE"]:
                result = "_".join([p[0] for p in parts if p])
            else:
                result = parts[0][0] if parts and parts[0] else "x"
                result += "".join([p[0] for p in parts[1:] if p])
                
        # Если результат совпадает с исходным идентификатором, вносим небольшое изменение
        if result == identifier:
            if style in ["snake_case", "UPPER_CASE"]:
                result += "_v"
            else:
                result += "V"
        
        # Сохраняем в кэш
        self.identifier_cache[identifier] = result
        
        return result


class SemanticIdentifierRenamer:
    """
    Класс для переименования идентификаторов в исходном коде с сохранением семантических подсказок.
    """
    
    def __init__(
        self, 
        domain: str = "general", 
        semantic_preservation_level: float = 0.7,
        preserve_keywords: bool = True
    ):
        """
        Инициализирует SemanticIdentifierRenamer.
        
        Args:
            domain: Предметная область кода (finance, medical, web, general)
            semantic_preservation_level: Уровень сохранения семантики (0.0-1.0)
            preserve_keywords: Сохранять ли ключевые слова языков программирования
        """
        self.renamer = SemanticRenamer(domain, semantic_preservation_level)
        self.preserve_keywords = preserve_keywords
        self._setup_keywords()
        
    def _setup_keywords(self):
        """Настраивает список ключевых слов для различных языков программирования."""
        self.keywords = {
            'python': ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 'import', 
                       'from', 'as', 'with', 'try', 'except', 'finally', 'raise', 'assert', 
                       'lambda', 'None', 'True', 'False'],
            'javascript': ['function', 'const', 'let', 'var', 'if', 'else', 'for', 'while', 
                           'return', 'import', 'export', 'class', 'this', 'new', 'async', 
                           'await', 'try', 'catch', 'finally', 'throw', 'null', 'undefined', 
                           'true', 'false'],
            'java': ['public', 'private', 'protected', 'class', 'interface', 'extends', 
                     'implements', 'static', 'final', 'void', 'if', 'else', 'for', 'while', 
                     'return', 'try', 'catch', 'finally', 'throw', 'new', 'null', 'true', 'false'],
            'cpp': ['int', 'float', 'double', 'char', 'bool', 'void', 'class', 'struct', 
                    'enum', 'template', 'namespace', 'using', 'if', 'else', 'for', 'while', 
                    'return', 'try', 'catch', 'throw', 'new', 'delete', 'nullptr', 'true', 'false'],
        }
        
        # Объединяем все ключевые слова в один набор
        self.all_keywords = set()
        for keywords in self.keywords.values():
            self.all_keywords.update(keywords)
    
    def rename_identifiers_in_code(self, code: str) -> str:
        """
        Переименовывает идентификаторы в коде с сохранением семантических подсказок.
        
        Args:
            code: Исходный код
            
        Returns:
            Код с переименованными идентификаторами
        """
        # Паттерн для поиска идентификаторов
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        
        def replace_identifier(match):
            identifier = match.group(0)
            
            # Пропускаем ключевые слова, если установлен соответствующий флаг
            if self.preserve_keywords and identifier in self.all_keywords:
                return identifier
            
            # Получаем контекст для типизированных языков (упрощенно)
            # В реальности здесь нужен более сложный анализ
            context = code[max(0, match.start() - 30):min(len(code), match.end() + 30)]
            
            # Переименовываем идентификатор
            return self.renamer.rename_identifier(identifier, context)
        
        # Применяем переименование ко всем идентификаторам в коде
        return re.sub(pattern, replace_identifier, code)
    
    def clear_cache(self):
        """Очищает кэш переименований."""
        self.renamer.identifier_cache = {} 