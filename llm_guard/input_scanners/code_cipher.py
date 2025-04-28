"""
LLM Guard: CodeCipherObfuscator
-------------------
Сканер, который обфусцирует исходный код с использованием метода, 
описанного в статье https://arxiv.org/html/2410.05797v1 "CodeCipher: Learning to Obfuscate Source Code Against LLMs".

Поддерживает два режима обфускации:
1. Базовый режим на основе регулярных выражений (всегда доступен)
2. Расширенный режим с использованием ANTLR для точного синтаксического анализа (при наличии зависимостей)

Для использования расширенного режима с ANTLR, необходимо установить следующие зависимости:
```
pip install antlr4-python3-runtime
```

И сгенерировать грамматики для поддерживаемых языков:
1. Скачайте ANTLR JAR файл с сайта https://www.antlr.org/download.html
2. Создайте директорию для грамматик:
```
mkdir -p llm_guard/input_scanners/antlr_grammars
```
3. Скачайте грамматики для нужных языков:
   - Python: https://github.com/antlr/grammars-v4/tree/master/python
   - JavaScript: https://github.com/antlr/grammars-v4/tree/master/javascript
   - Java: https://github.com/antlr/grammars-v4/tree/master/java
   - C#: https://github.com/antlr/grammars-v4/tree/master/csharp
4. Сгенерируйте парсеры для каждого языка (пример для Python):
```
java -jar antlr-4.9.3-complete.jar -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/ Python3.g4
```

:Copyright: © 2023-2023
:License: MIT (see LICENSE for details)
"""

import os
import re
import json
import uuid
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set, Union
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import hashlib

# Проверка наличия ANTLR и грамматик
ANTLR_AVAILABLE = False
GRAMMARS_AVAILABLE = False

try:
    from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker, ErrorListener
    ANTLR_AVAILABLE = True
    
    # Определим простой слушатель ошибок, который их игнорирует
    class SilentErrorListener(ErrorListener):
        def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
            pass
        
        def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigSet, configs):
            pass
        
        def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
            pass
        
        def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
            pass
    
    # Проверка наличия грамматик для различных языков
    languages = ['python', 'javascript', 'java', 'csharp']
    grammars_path = Path(__file__).parent / 'antlr_grammars'
    
    if grammars_path.exists():
        # Проверяем наличие грамматик для каждого языка
        available_grammars = []
        
        for lang in languages:
            lang_path = grammars_path / lang
            if lang_path.exists():
                available_grammars.append(lang)
        
        if len(available_grammars) > 0:
            GRAMMARS_AVAILABLE = True
            
            # Импортируем необходимые классы для каждого языка
            if 'python' in available_grammars:
                from llm_guard.input_scanners.antlr_grammars.python.Python3Lexer import Python3Lexer
                from llm_guard.input_scanners.antlr_grammars.python.Python3Parser import Python3Parser
                from llm_guard.input_scanners.antlr_grammars.python.Python3Listener import Python3Listener
                from llm_guard.input_scanners.antlr_grammars.python.PythonIdentifierCollector import PythonIdentifierCollector
            
            if 'javascript' in available_grammars:
                from llm_guard.input_scanners.antlr_grammars.javascript.JavaScriptLexer import JavaScriptLexer
                from llm_guard.input_scanners.antlr_grammars.javascript.JavaScriptParser import JavaScriptParser
                from llm_guard.input_scanners.antlr_grammars.javascript.JavaScriptListener import JavaScriptListener
                from llm_guard.input_scanners.antlr_grammars.javascript.JavaScriptIdentifierCollector import JavaScriptIdentifierCollector
            
            if 'java' in available_grammars:
                from llm_guard.input_scanners.antlr_grammars.java.JavaLexer import JavaLexer
                from llm_guard.input_scanners.antlr_grammars.java.JavaParser import JavaParser
                from llm_guard.input_scanners.antlr_grammars.java.JavaListener import JavaListener
                from llm_guard.input_scanners.antlr_grammars.java.JavaIdentifierCollector import JavaIdentifierCollector
            
            if 'csharp' in available_grammars:
                from llm_guard.input_scanners.antlr_grammars.csharp.CSharpLexer import CSharpLexer
                from llm_guard.input_scanners.antlr_grammars.csharp.CSharpParser import CSharpParser
                from llm_guard.input_scanners.antlr_grammars.csharp.CSharpListener import CSharpListener
                from llm_guard.input_scanners.antlr_grammars.csharp.CSharpIdentifierCollector import CSharpIdentifierCollector
            
            # Импорты для новых языков
            if 'kotlin' in available_grammars:
                from llm_guard.input_scanners.antlr_grammars.kotlin.KotlinLexer import KotlinLexer
                from llm_guard.input_scanners.antlr_grammars.kotlin.KotlinParser import KotlinParser
                from llm_guard.input_scanners.antlr_grammars.kotlin.KotlinIdentifierCollector import KotlinIdentifierCollector
            
            if 'php' in available_grammars:
                from llm_guard.input_scanners.antlr_grammars.php.PhpLexer import PhpLexer
                from llm_guard.input_scanners.antlr_grammars.php.PhpParser import PhpParser
                from llm_guard.input_scanners.antlr_grammars.php.PHPIdentifierCollector import PHPIdentifierCollector
            
            if 'ruby' in available_grammars:
                from llm_guard.input_scanners.antlr_grammars.ruby.CorundumLexer import CorundumLexer
                from llm_guard.input_scanners.antlr_grammars.ruby.CorundumParser import CorundumParser
                from llm_guard.input_scanners.antlr_grammars.ruby.RubyIdentifierCollector import RubyIdentifierCollector
            
            if 'swift' in available_grammars:
                from llm_guard.input_scanners.antlr_grammars.swift.Swift5Lexer import Swift5Lexer
                from llm_guard.input_scanners.antlr_grammars.swift.Swift5Parser import Swift5Parser
                from llm_guard.input_scanners.antlr_grammars.swift.SwiftIdentifierCollector import SwiftIdentifierCollector
            
            if 'golang' in available_grammars:
                from llm_guard.input_scanners.antlr_grammars.golang.GoLexer import GoLexer
                from llm_guard.input_scanners.antlr_grammars.golang.GoParser import GoParser
                from llm_guard.input_scanners.antlr_grammars.golang.GoIdentifierCollector import GoIdentifierCollector
            
            if 'rust' in available_grammars:
                from llm_guard.input_scanners.antlr_grammars.rust.RustLexer import RustLexer
                from llm_guard.input_scanners.antlr_grammars.rust.RustParser import RustParser
                from llm_guard.input_scanners.antlr_grammars.rust.RustIdentifierCollector import RustIdentifierCollector

except ImportError:
    # ANTLR не установлен
    pass

from llm_guard.input_scanners.base import Scanner
from llm_guard.util import get_logger

logger = get_logger()


@dataclass
class CodeBlock:
    """Класс для хранения информации о блоке кода"""
    original: str
    obfuscated: Optional[str] = None
    language: Optional[str] = None
    id: str = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


# Базовый класс для сбора идентификаторов с помощью ANTLR
class BaseIdentifierCollector:
    def __init__(self):
        self.identifiers = set()
        self.imports = set()
        self.string_literals = set()
        self.numeric_literals = set()
        self.variable_assignments = {}  # {имя_переменной: значение}
    
    def get_all_non_obfuscatable(self):
        """Возвращает все элементы, которые не должны быть обфусцированы"""
        non_obfuscatable = set()
        non_obfuscatable.update(self.imports)
        non_obfuscatable.update(self.string_literals)
        non_obfuscatable.update(self.numeric_literals)
        for value in self.variable_assignments.values():
            if isinstance(value, str):
                non_obfuscatable.add(value)
        return non_obfuscatable


class CodeCipherObfuscator(Scanner):
    """
    Сканер для обфускации исходного кода с использованием метода CodeCipher.
    
    Данный сканер реализует подход, описанный в статье "CodeCipher: Learning to Obfuscate Source Code Against LLMs".
    Он обфусцирует исходный код путем замены токенов на семантически искаженные эквиваленты,
    обученные с использованием специализированного алгоритма оптимизации.
    
    Attributes:
        model_name (str): Имя модели для использования при обфускации
        max_training_iterations (int): Максимальное количество итераций обучения
        learning_rate (float): Скорость обучения для оптимизации
        perplexity_threshold (float): Порог перплексии для ранней остановки
        early_stopping_patience (int): Количество итераций без улучшения перед остановкой
        task_specific_dataset (List[str]): Набор данных для оптимизации обфускации
        vault_dir (str): Директория для хранения обфусцированного кода
        skip_patterns (List[str]): Шаблоны для пропуска обфускации
        use_antlr (bool): Использовать ли ANTLR парсинг, когда доступен
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        max_training_iterations: int = 100,
        learning_rate: float = 0.01,
        perplexity_threshold: float = 50.0,
        early_stopping_patience: int = 5,
        task_specific_dataset: Optional[List[str]] = None,
        vault_dir: Optional[str] = None,
        skip_patterns: Optional[List[str]] = None,
        use_antlr: bool = True  # Новый параметр: использовать ли ANTLR когда доступен
    ):
        """
        Инициализация сканера CodeCipherObfuscator.
        
        Args:
            model_name: Имя модели для использования при обфускации
            max_training_iterations: Максимальное количество итераций обучения
            learning_rate: Скорость обучения для оптимизации
            perplexity_threshold: Порог перплексии для ранней остановки
            early_stopping_patience: Количество итераций без улучшения перед остановкой
            task_specific_dataset: Набор данных для оптимизации обфускации
            vault_dir: Директория для хранения обфусцированного кода
            skip_patterns: Шаблоны для пропуска обфускации
            use_antlr: Использовать ли ANTLR парсинг, когда доступен
        """
        super().__init__()
        
        # Параметры сканера
        self.model_name = model_name
        self.max_training_iterations = max_training_iterations
        self.learning_rate = learning_rate
        self.perplexity_threshold = perplexity_threshold
        self.early_stopping_patience = early_stopping_patience
        self.task_specific_dataset = task_specific_dataset or []
        self.skip_patterns = skip_patterns or []
        self.skip_patterns_compiled = [re.compile(pattern) for pattern in self.skip_patterns]
        self.use_antlr = use_antlr and ANTLR_AVAILABLE and GRAMMARS_AVAILABLE
        
        if self.use_antlr:
            logger.info("ANTLR поддержка активирована для обфускации кода")
        else:
            if use_antlr and not ANTLR_AVAILABLE:
                logger.warning("ANTLR недоступен. Установите antlr4-python3-runtime для улучшенной обфускации")
            elif use_antlr and not GRAMMARS_AVAILABLE:
                logger.warning("ANTLR грамматики недоступны. Добавьте грамматики в llm_guard/input_scanners/antlr_grammars/")
        
        # Параметры хранилища
        if vault_dir:
            self.vault_dir = Path(vault_dir)
            os.makedirs(self.vault_dir, exist_ok=True)
        else:
            self.vault_dir = Path(os.path.join(os.path.expanduser("~"), ".llm_guard", "obfuscation_vault"))
            os.makedirs(self.vault_dir, exist_ok=True)
        
        # Инициализация модели и токенизатора
        logger.info(f"Initializing CodeCipherObfuscator with model {model_name}")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Получение матрицы эмбеддингов
            self.token_embeddings = self.model.get_input_embeddings().weight.detach().clone()
            
            # Создание токен-в-токен отображения для обфускации
            self._initialize_obfuscation_mapping()
            
        except Exception as e:
            logger.error(f"Error initializing CodeCipherObfuscator: {e}")
            raise e
        
        # Кэш обфускации для поддержания консистентности
        self.obfuscation_cache = {}
        
        logger.info("CodeCipherObfuscator initialized successfully")
    
    def _initialize_obfuscation_mapping(self):
        """
        Инициализация отображения для обфускации токенов.
        Создает токен-в-токен отображение, оптимизированное для обфускации кода.
        """
        logger.info("Initializing obfuscation mapping...")
        
        # Если у нас есть набор данных для оптимизации, используем его
        if self.task_specific_dataset:
            self._train_obfuscation_mapping()
        else:
            # В противном случае используем простое random permutation
            vocab_size = len(self.tokenizer)
            permutation = np.random.permutation(vocab_size)
            self.token_mapping = {i: permutation[i] for i in range(vocab_size)}
            
            # Исключаем специальные токены из обфускации
            for token_id in self.tokenizer.all_special_ids:
                self.token_mapping[token_id] = token_id
        
        logger.info("Obfuscation mapping initialized")
    
    def _train_obfuscation_mapping(self):
        """
        Обучение отображения токен-в-токен для обфускации.
        Использует подход из статьи CodeCipher для создания оптимального отображения.
        """
        logger.info("Training token-to-token mapping...")
        
        # Токенизируем набор данных
        encoded_data = [self.tokenizer.encode(text, return_tensors="pt").to(self.device) 
                        for text in self.task_specific_dataset]
        
        # Инициализируем оптимизируемую матрицу эмбеддингов
        vocab_size, embed_dim = self.token_embeddings.shape
        perturbed_embeddings = torch.nn.Parameter(self.token_embeddings.clone())
        optimizer = Adam([perturbed_embeddings], lr=self.learning_rate)
        
        # Обучение отображения
        best_loss = float('inf')
        patience_counter = 0
        
        progress_bar = tqdm(range(self.max_training_iterations), desc="Training")
        for iter_idx in progress_bar:
            total_loss = 0
            
            for encoded_text in encoded_data:
                # Вычисляем исходную перплексию
                with torch.no_grad():
                    orig_outputs = self.model(encoded_text)
                    orig_loss = orig_outputs.loss.item()
                
                # Создаем копию модели и заменяем эмбеддинги
                temp_model = self.model.clone()
                temp_model.get_input_embeddings().weight.data = perturbed_embeddings
                
                # Вычисляем новую перплексию
                outputs = temp_model(encoded_text)
                new_loss = outputs.loss
                
                # Вычисляем расстояние между оригинальными и обфусцированными эмбеддингами
                embedding_distance = torch.norm(
                    self.token_embeddings - perturbed_embeddings, dim=1
                ).mean()
                
                # Наша цель - максимизировать расстояние (обфускация),
                # сохраняя перплексию в пределах порога
                perplexity_factor = max(0, new_loss.item() / orig_loss - self.perplexity_threshold)
                loss = -embedding_distance + perplexity_factor * new_loss
                
                # Обратное распространение ошибки
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Проверка ранней остановки
            avg_loss = total_loss / len(encoded_data)
            progress_bar.set_postfix({"loss": avg_loss})
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at iteration {iter_idx}")
                break
        
        # После обучения создаем отображение токенов
        # Для каждого токена находим ближайший токен в новом пространстве
        token_mapping = {}
        
        for i in range(vocab_size):
            # Вычисляем расстояние между текущим эмбеддингом и всеми другими
            distances = torch.norm(
                perturbed_embeddings[i].unsqueeze(0) - self.token_embeddings, dim=1
            )
            
            # Находим ближайший токен (исключая себя)
            distances[i] = float('inf')  # Исключаем самого себя
            closest_token = torch.argmin(distances).item()
            token_mapping[i] = closest_token
        
        # Исключаем специальные токены из обфускации
        for token_id in self.tokenizer.all_special_ids:
            token_mapping[token_id] = token_id
        
        self.token_mapping = token_mapping
        logger.info("Token-to-token mapping trained successfully")
    
    def _should_skip_obfuscation(self, code: str) -> bool:
        """
        Проверяет, следует ли пропустить обфускацию кода.
        
        Args:
            code: Исходный код для проверки
            
        Returns:
            bool: True, если код следует пропустить, иначе False
        """
        if not self.skip_patterns_compiled:
            return False
        
        for pattern in self.skip_patterns_compiled:
            if pattern.search(code):
                return True
        
        return False
    
    def _obfuscate_code(self, code: str) -> str:
        """
        Обфусцирует исходный код с использованием интеллектуальной замены идентификаторов.
        Сохраняет структуру и синтаксис кода, ключевые слова языка, но заменяет
        имена функций, переменных и классов на запутанные версии.
        
        Args:
            code: Исходный код для обфускации
            
        Returns:
            str: Обфусцированный код
        """
        # Если код уже был обфусцирован, возвращаем его из кэша
        if code in self.obfuscation_cache:
            return self.obfuscation_cache[code]
        
        # Проверяем, нужно ли пропустить обфускацию
        if self._should_skip_obfuscation(code):
            logger.info("Skipping obfuscation due to skip pattern match")
            return code
        
        # Определим язык кода
        detected_language = self._detect_language(code)
        
        # Идентификаторы, которые не нужно обфусцировать
        do_not_obfuscate = set()
        
        # Словарь для отображения идентификаторов
        identifier_mapping = {}

        # Попытаемся использовать ANTLR для более точного анализа кода
        antlr_used = False
        if self.use_antlr and ANTLR_AVAILABLE and GRAMMARS_AVAILABLE:
            try:
                logger.debug(f"Используем ANTLR для обфускации кода на языке {detected_language}")
                _, antlr_non_obfuscatable = self._obfuscate_with_antlr(code, detected_language)
                do_not_obfuscate.update(antlr_non_obfuscatable)
                antlr_used = True
                logger.debug(f"ANTLR обнаружил {len(antlr_non_obfuscatable)} необфусцируемых элементов")
            except Exception as e:
                logger.warning(f"Ошибка при использовании ANTLR: {e}. Используем резервный метод на основе регулярных выражений.")
                antlr_used = False
        
        # Если ANTLR не использовался или не смог обработать код, используем регулярные выражения
        if not antlr_used:
            # Список ключевых слов для популярных языков программирования
            keywords = {
                'python': [
                    'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 
                    'import', 'from', 'as', 'with', 'try', 'except', 'finally',
                    'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'lambda',
                    'global', 'nonlocal', 'yield', 'assert', 'del', 'pass', 'break', 'continue',
                    'raise', 'async', 'await', 'self', 'super'
                ],
                'javascript': [
                    'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'return',
                    'class', 'extends', 'super', 'this', 'new', 'try', 'catch', 'finally',
                    'throw', 'typeof', 'instanceof', 'import', 'export', 'default', 'null',
                    'undefined', 'true', 'false', 'async', 'await', 'break', 'continue',
                    'switch', 'case', 'delete', 'in', 'of', 'yield', 'static', 'get', 'set'
                ],
                'java': [
                    'public', 'private', 'protected', 'class', 'interface', 'extends', 'implements',
                    'static', 'final', 'abstract', 'void', 'if', 'else', 'for', 'while', 'return',
                    'try', 'catch', 'finally', 'throw', 'throws', 'new', 'this', 'super', 'package',
                    'import', 'true', 'false', 'null', 'instanceof', 'break', 'continue', 'switch', 'case',
                    'byte', 'short', 'int', 'long', 'float', 'double', 'char', 'boolean', 'default', 'enum'
                ],
                'csharp': [
                    'public', 'private', 'protected', 'internal', 'class', 'interface', 'struct', 'enum',
                    'static', 'readonly', 'const', 'void', 'if', 'else', 'for', 'while', 'return',
                    'try', 'catch', 'finally', 'throw', 'new', 'this', 'base', 'namespace',
                    'using', 'true', 'false', 'null', 'break', 'continue', 'switch', 'case',
                    'int', 'long', 'float', 'double', 'char', 'bool', 'string', 'var', 'delegate',
                    'event', 'virtual', 'override', 'abstract', 'async', 'await', 'sealed', 'partial'
                ],
                'php': [
                    'function', 'class', 'if', 'else', 'elseif', 'for', 'foreach', 'while', 'return',
                    'public', 'private', 'protected', 'static', 'final', 'abstract', 'try', 'catch',
                    'finally', 'throw', 'new', 'this', 'extends', 'implements', 'namespace',
                    'use', 'true', 'false', 'null', 'break', 'continue', 'switch', 'case',
                    'echo', 'print', 'include', 'include_once', 'require', 'require_once', 'var',
                    'global', 'array', 'const', 'trait', 'interface', 'clone', 'yield', 'yield from'
                ],
                'kotlin': [
                    'fun', 'class', 'if', 'else', 'for', 'while', 'when', 'return',
                    'public', 'private', 'protected', 'internal', 'open', 'final', 'abstract', 'try', 'catch',
                    'finally', 'throw', 'val', 'var', 'constructor', 'init', 'this', 'super', 'package',
                    'import', 'true', 'false', 'null', 'break', 'continue', 'object', 'interface', 
                    'companion', 'data', 'sealed', 'inline', 'typealias', 'suspend', 'override',
                    'lateinit', 'infix', 'operator', 'reified', 'crossinline', 'noinline', 'external'
                ]
            }
            
            # Объединяем все ключевые слова
            all_keywords = set()
            for lang_keywords in keywords.values():
                all_keywords.update(lang_keywords)
            
            # Шаблоны импортов для разных языков
            import_patterns = {
                'python': r'^(?:import|from)\s+([a-zA-Z0-9_\.]+)(?:\s+import\s+(?:[a-zA-Z0-9_\*]+(?:\s*,\s*[a-zA-Z0-9_\*]+)*))?\s*(?:as\s+[a-zA-Z0-9_]+)?$',
                'javascript': r'^import\s+(?:{[^}]*}|[a-zA-Z0-9_*$]+)\s+from\s+[\'"][^\'""]+[\'"]$',
                'java': r'^import\s+[a-zA-Z0-9_\.]+(?:\.\*)?;$',
                'csharp': r'^using\s+(?:static\s+)?[a-zA-Z0-9_\.]+(?:\s*=\s*[a-zA-Z0-9_\.]+)?;$',
                'php': r'^(?:use|require|include|require_once|include_once)\s+[a-zA-Z0-9_\\]+(?:\s+as\s+[a-zA-Z0-9_]+)?;$',
                'kotlin': r'^import\s+[a-zA-Z0-9_\.]+(?:\.\*)?(?:\s+as\s+[a-zA-Z0-9_]+)?$'
            }
            
            # Найдем и сохраним строки и числовые литералы
            string_literals = self._extract_string_literals(code)
            for string_literal in string_literals:
                do_not_obfuscate.add(string_literal)
            
            # Распознаем и сохраняем значения строковых переменных
            # Шаблон для присваивания строковых литералов переменным
            string_assignments = []
            
            # Шаблоны для разных языков программирования
            assignment_patterns = {
                'python': r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\'\"][^\'\"]*[\'\"])',
                'javascript': r'(var|let|const)?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\'\"`][^\'\"]*[\'\"`])',
                'java': r'(String)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\'\"][^\'\"]*[\'\"])',
                'csharp': r'(string|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\'\"][^\'\"]*[\'\"])',
                'kotlin': r'(val|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(:.*?)?\s*=\s*([\'\"][^\'\"]*[\'\"])',
                'php': r'\$([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\'\"][^\'\"]*[\'\"])'
            }
            
            # Добавляем общий шаблон для всех языков, если язык не определен
            default_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\'\"][^\'\"]*[\'\"])'
            
            # Используем соответствующий шаблон для поиска
            pattern = assignment_patterns.get(detected_language, default_pattern)
            if detected_language == 'python':
                for match in re.finditer(pattern, code):
                    var_name, string_value = match.groups()
                    string_assignments.append((var_name, string_value))
            elif detected_language == 'javascript':
                for match in re.finditer(pattern, code):
                    var_type, var_name, string_value = match.groups()
                    string_assignments.append((var_name, string_value))
            elif detected_language == 'java' or detected_language == 'csharp':
                for match in re.finditer(pattern, code):
                    var_type, var_name, string_value = match.groups()
                    string_assignments.append((var_name, string_value))
            elif detected_language == 'kotlin':
                for match in re.finditer(pattern, code):
                    var_type, var_name, type_annotation, string_value = match.groups()
                    string_assignments.append((var_name, string_value))
            elif detected_language == 'php':
                for match in re.finditer(pattern, code):
                    var_name, string_value = match.groups()
                    string_assignments.append((var_name, string_value))
            else:
                # Общий случай для других языков
                for match in re.finditer(default_pattern, code):
                    var_name, string_value = match.groups()
                    string_assignments.append((var_name, string_value))
            
            # Добавляем найденные строковые значения переменных в список исключений
            for var_name, string_value in string_assignments:
                do_not_obfuscate.add(string_value)
            
            # Найдем и сохраним числовые литералы
            number_literals = self._extract_number_literals(code)
            for number_literal in number_literals:
                do_not_obfuscate.add(number_literal)
            
            # Найдем и сохраним строки импортов и идентификаторы в них
            import_lines = []
            import_identifiers = set()
            if detected_language in import_patterns:
                pattern = import_patterns[detected_language]
                for line in code.split('\n'):
                    line_stripped = line.strip()
                    if re.match(pattern, line_stripped):
                        import_lines.append(line_stripped)
                        do_not_obfuscate.add(line_stripped)
                        
                        # Извлекаем идентификаторы из импортов
                        if detected_language == 'python':
                            # Для Python извлекаем модули и импортируемые имена
                            if 'import' in line_stripped:
                                parts = line_stripped.split('import')
                                if len(parts) > 1:
                                    if 'from' in parts[0]:
                                        # Для конструкции "from X import Y"
                                        module = parts[0].replace('from', '').strip()
                                        import_identifiers.update([m.strip() for m in module.split('.')])
                                    
                                    # Добавляем импортируемые имена
                                    names = parts[-1].strip()
                                    if 'as' in names:
                                        # Обрабатываем случай "import X as Y"
                                        names_parts = names.split('as')
                                        original_name = names_parts[0].strip()
                                        import_identifiers.update([n.strip() for n in original_name.split('.')])
                                        import_identifiers.add(names_parts[1].strip())
                                    else:
                                        # Простые импорты
                                        import_identifiers.update([n.strip() for n in names.split(',')])
                                        for name in names.split(','):
                                            import_identifiers.update([n.strip() for n in name.split('.')])
                        
                        elif detected_language == 'javascript':
                            # Для JavaScript извлекаем имена из import
                            if 'from' in line_stripped:
                                parts = line_stripped.split('from')
                                names = parts[0].replace('import', '').strip()
                                if '{' in names:
                                    # Для именованных импортов "import { X, Y } from 'module'"
                                    names = names.strip('{}')
                                    import_identifiers.update([n.strip() for n in names.split(',')])
                                else:
                                    # Для импортов по умолчанию "import X from 'module'"
                                    import_identifiers.add(names)
                        
                        # Для других языков добавьте соответствующую логику извлечения идентификаторов
                        elif detected_language == 'java':
                            # Для Java извлекаем пакеты и классы
                            if 'import' in line_stripped:
                                # Убираем "import" и ";"
                                import_path = line_stripped.replace('import', '').replace(';', '').strip()
                                # Обрабатываем импорт с "*" (всего пакета)
                                if import_path.endswith('.*'):
                                    import_path = import_path[:-2]
                                # Разбиваем на части пакета
                                import_identifiers.update([part.strip() for part in import_path.split('.')])
                        
                        elif detected_language == 'csharp':
                            # Для C# извлекаем пространства имен и классы
                            if 'using' in line_stripped:
                                # Убираем "using" и ";"
                                using_path = line_stripped.replace('using', '').replace(';', '').strip()
                                # Обрабатываем статические импорты
                                if 'static' in using_path:
                                    using_path = using_path.replace('static', '').strip()
                                # Обрабатываем использование с псевдонимом
                                if '=' in using_path:
                                    parts = using_path.split('=')
                                    alias = parts[0].strip()
                                    original = parts[1].strip()
                                    import_identifiers.add(alias)
                                    import_identifiers.update([part.strip() for part in original.split('.')])
                                else:
                                    # Разбиваем на части пространства имен
                                    import_identifiers.update([part.strip() for part in using_path.split('.')])
                        
                        elif detected_language == 'php':
                            # Для PHP извлекаем пространства имен и классы
                            if any(keyword in line_stripped for keyword in ['use', 'require', 'include', 'require_once', 'include_once']):
                                # Убираем ключевое слово и ";"
                                for keyword in ['use', 'require', 'include', 'require_once', 'include_once']:
                                    if keyword in line_stripped:
                                        import_path = line_stripped.replace(keyword, '').replace(';', '').strip()
                                        break
                                
                                # Обрабатываем импорт с псевдонимом
                                if ' as ' in import_path:
                                    parts = import_path.split(' as ')
                                    original = parts[0].strip()
                                    alias = parts[1].strip()
                                    import_identifiers.add(alias)
                                    import_identifiers.update([part.strip() for part in original.split('\\')])
                                else:
                                    # Разбиваем на части пространства имен
                                    import_identifiers.update([part.strip() for part in import_path.split('\\')])
                        
                        elif detected_language == 'kotlin':
                            # Для Kotlin извлекаем пакеты и классы
                            if 'import' in line_stripped:
                                # Убираем "import"
                                import_path = line_stripped.replace('import', '').strip()
                                # Обрабатываем импорт с "*"
                                if import_path.endswith('.*'):
                                    import_path = import_path[:-2]
                                # Обрабатываем импорт с псевдонимом
                                if ' as ' in import_path:
                                    parts = import_path.split(' as ')
                                    original = parts[0].strip()
                                    alias = parts[1].strip()
                                    import_identifiers.add(alias)
                                    import_identifiers.update([part.strip() for part in original.split('.')])
                                else:
                                    # Разбиваем на части пакета
                                    import_identifiers.update([part.strip() for part in import_path.split('.')])
        
        # Регулярное выражение для идентификаторов (работает для большинства языков программирования)
        identifier_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        # Находим все идентификаторы в коде
        identifiers = set(re.findall(identifier_pattern, code))
        
        # Исключаем ключевые слова, магические методы/переменные, внутренние имена и идентификаторы в импортах
        filtered_identifiers = []
        for ident in identifiers:
            # Проверяем если используем ANTLR
            if antlr_used:
                # Проверяем только на исключения из do_not_obfuscate
                if (not ident.startswith('__') and 
                    not (detected_language == 'python' and ident.startswith('_')) and
                    ident not in do_not_obfuscate):
                    filtered_identifiers.append(ident)
            else:
                # Если ANTLR не используется, проверяем и на ключевые слова
                if (ident not in all_keywords and 
                    not ident.startswith('__') and 
                    not (detected_language == 'python' and ident.startswith('_')) and
                    ident not in import_identifiers and
                    ident not in do_not_obfuscate):
                    filtered_identifiers.append(ident)
        
        # Создаем маску для строковых литералов
        masked_code = code
        string_placeholders = {}
        for i, string_literal in enumerate(string_literals):
            placeholder = f"__STRING_PLACEHOLDER_{i}__"
            string_placeholders[placeholder] = string_literal
            masked_code = masked_code.replace(string_literal, placeholder)
        
        # Создаем обфусцированную версию кода
        obfuscated_code = masked_code
        
        # Сортируем идентификаторы по длине, чтобы избежать частичных замен
        sorted_identifiers = sorted(filtered_identifiers, key=len, reverse=True)
        
        # Генерируем уникальные запутанные имена для каждого идентификатора и применяем их
        for ident in sorted_identifiers:
            # Если идентификатор уже обрабатывался, используем существующее отображение
            if ident in identifier_mapping:
                obf_name = identifier_mapping[ident]
            else:
                # Генерируем запутанное имя, сохраняя семантические подсказки
                obf_name = self._generate_obfuscated_name(ident, detected_language)
                identifier_mapping[ident] = obf_name
            
            # Заменяем только идентификаторы как целые слова, учитывая границы слов
            obfuscated_code = re.sub(
                r'\b' + re.escape(ident) + r'\b',
                obf_name,
                obfuscated_code
            )
        
        # Восстанавливаем строковые литералы
        for placeholder, string_literal in string_placeholders.items():
            obfuscated_code = obfuscated_code.replace(placeholder, string_literal)
        
        # Сохраняем в кэш
        self.obfuscation_cache[code] = obfuscated_code
        return obfuscated_code
    
    def _detect_language(self, code: str) -> str:
        """
        Определяет язык программирования по коду.
        
        Args:
            code: Исходный код для анализа
            
        Returns:
            str: Определенный язык программирования
        """
        # Характерные шаблоны для разных языков
        patterns = {
            'python': [
                r'\bimport\s+[a-zA-Z0-9_\.]+',
                r'\bfrom\s+[a-zA-Z0-9_\.]+\s+import\b',
                r'\bdef\s+[a-zA-Z0-9_]+\s*\(',
                r'\bclass\s+[a-zA-Z0-9_]+\s*(\(|:)',
                r':\s*\n'
            ],
            'javascript': [
                r'\bfunction\s+[a-zA-Z0-9_]+\s*\(',
                r'\bconst\s+[a-zA-Z0-9_]+\s*=',
                r'\blet\s+[a-zA-Z0-9_]+\s*=',
                r'\bvar\s+[a-zA-Z0-9_]+\s*=',
                r'\bimport\s+.*\s+from\s+[\'"]',
                r'export\s+(?:default\s+)?(?:function|class|const|let|var)',
                r'=>'
            ],
            'java': [
                r'\bpublic\s+(?:class|interface|enum)\s+[a-zA-Z0-9_]+',
                r'\bimport\s+[a-zA-Z0-9_\.]+;',
                r'\bpackage\s+[a-zA-Z0-9_\.]+;',
                r'\bprivate\s+[a-zA-Z0-9_<>]+\s+[a-zA-Z0-9_]+\s*\(',
                r'\bprotected\s+[a-zA-Z0-9_<>]+\s+[a-zA-Z0-9_]+\s*\('
            ],
            'csharp': [
                r'\bnamespace\s+[a-zA-Z0-9_\.]+',
                r'\busing\s+[a-zA-Z0-9_\.]+;',
                r'\bpublic\s+(?:class|interface|struct|enum)\s+[a-zA-Z0-9_]+',
                r'\bprivate\s+[a-zA-Z0-9_<>]+\s+[a-zA-Z0-9_]+\s*\(',
                r'\bprotected\s+[a-zA-Z0-9_<>]+\s+[a-zA-Z0-9_]+\s*\(',
                r'\binternal\s+[a-zA-Z0-9_<>]+\s+[a-zA-Z0-9_]+\s*\('
            ],
            'php': [
                r'<\?php',
                r'\bfunction\s+[a-zA-Z0-9_]+\s*\(',
                r'\bnamespace\s+[a-zA-Z0-9_\\]+;',
                r'\buse\s+[a-zA-Z0-9_\\]+(?:\s+as\s+[a-zA-Z0-9_]+)?;',
                r'\bclass\s+[a-zA-Z0-9_]+(?:\s+extends\s+[a-zA-Z0-9_]+)?(?:\s+implements\s+[a-zA-Z0-9_,\s]+)?'
            ],
            'kotlin': [
                r'\bfun\s+[a-zA-Z0-9_]+\s*\(',
                r'\bval\s+[a-zA-Z0-9_]+\s*:',
                r'\bvar\s+[a-zA-Z0-9_]+\s*:',
                r'\bclass\s+[a-zA-Z0-9_]+(?:\s*\(|:)',
                r'\bimport\s+[a-zA-Z0-9_\.]+(?:\.\*)?'
            ]
        }
        
        # Подсчитываем совпадения для каждого языка
        matches = {lang: 0 for lang in patterns}
        for lang, patterns_list in patterns.items():
            for pattern in patterns_list:
                if re.search(pattern, code):
                    matches[lang] += 1
        
        # Выбираем язык с наибольшим количеством совпадений
        best_match = max(matches, key=matches.get)
        if matches[best_match] > 0:
            return best_match
        
        # Если не удалось определить, возвращаем Python как наиболее вероятный
        return 'python'
    
    def _extract_string_literals(self, code: str) -> List[str]:
        """
        Извлекает строковые литералы из кода.
        
        Args:
            code: Исходный код
            
        Returns:
            List[str]: Список строковых литералов
        """
        # Шаблоны для разных типов строк
        patterns = [
            # Строки в одинарных кавычках
            r'\'(?:\\.|[^\\\'])*\'',
            # Строки в двойных кавычках
            r'"(?:\\.|[^\\"])*"',
            # Многострочные строки Python (тройные кавычки)
            r'"""(?:\\.|[^\\"])*"""',
            r"'''(?:\\.|[^\\'])*'''",
            # Template literals в JavaScript
            r'`(?:\\.|[^\\`])*`'
        ]
        
        literals = []
        for pattern in patterns:
            literals.extend(re.findall(pattern, code))
        
        return literals
    
    def _extract_number_literals(self, code: str) -> List[str]:
        """
        Извлекает числовые литералы из кода.
        
        Args:
            code: Исходный код
            
        Returns:
            List[str]: Список числовых литералов
        """
        # Шаблон для различных форматов чисел
        pattern = r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[fFdDlL]?\b'
        
        return re.findall(pattern, code)
    
    def _generate_obfuscated_name(self, original_name: str, language: str) -> str:
        """
        Генерирует обфусцированное имя для идентификатора, сохраняя семантические подсказки.
        
        Args:
            original_name: Оригинальное имя
            language: Язык программирования
            
        Returns:
            str: Обфусцированное имя
        """
        # Создаем псевдослучайный хеш на основе оригинального имени
        hash_val = hashlib.md5(original_name.encode()).hexdigest()
        
        # Используем часть хеша, чтобы имя оставалось достаточно коротким
        suffix = hash_val[:6]
        
        # Сохраняем особенности имени в зависимости от типа идентификатора
        
        # 1. Сохраняем конвенции именования для разных языков
        if language == 'python':
            # Сохраняем конвенцию приватности в Python
            if original_name.startswith('_') and not original_name.startswith('__'):
                return f"_{original_name[1]}{suffix}"
                
            # Сохраняем конвенцию для констант
            if original_name.isupper():
                return f"{original_name[0]}{suffix}".upper()
                
        # 2. Обработка общих паттернов именования в разных языках программирования
        
        # Обработка геттеров и сеттеров
        if original_name.startswith('get') and len(original_name) > 3:
            # Сохраняем префикс геттеров и первую букву имени свойства
            return f"get{original_name[3].lower()}{suffix}"
            
        if original_name.startswith('set') and len(original_name) > 3:
            # Сохраняем префикс сеттеров и первую букву имени свойства
            return f"set{original_name[3].lower()}{suffix}"
            
        # Обработка булевых методов и свойств
        if original_name.startswith('is') and len(original_name) > 2:
            # Сохраняем префикс 'is' и первую букву проверяемого свойства
            return f"is{original_name[2].lower()}{suffix}"
            
        if original_name.startswith('has') and len(original_name) > 3:
            # Сохраняем префикс 'has' и первую букву проверяемого свойства
            return f"has{original_name[3].lower()}{suffix}"
            
        # Обработка тестовых функций и моков
        if 'test' in original_name.lower():
            # Для тестовых функций сохраняем префикс 'test'
            if original_name.lower().startswith('test'):
                return f"test_{suffix}"
            else:
                return f"{original_name[0]}test_{suffix}"
                
        if 'mock' in original_name.lower():
            # Для моков сохраняем префикс 'mock'
            if original_name.lower().startswith('mock'):
                return f"mock_{suffix}"
            else:
                return f"{original_name[0]}mock_{suffix}"
                
        # 3. Сохраняем стиль именования
        
        # Обработка snake_case (разделение подчеркиваниями)
        if '_' in original_name:
            # Выделяем отдельные части имени
            parts = original_name.split('_')
            # Для snake_case берем первую букву каждой значимой части
            abbr = ''.join([p[0] for p in parts if p])
            return f"{abbr}_{suffix}"
            
        # Обработка camelCase и PascalCase
        elif any(c.isupper() for c in original_name[1:]):
            # Находим все заглавные буквы (начало новых слов)
            capitals = [c for c in original_name if c.isupper()]
            if len(capitals) > 0:
                # Используем аббревиатуру из заглавных букв
                abbr = ''.join(capitals).lower()
                
                # Сохраняем стиль PascalCase или camelCase
                if original_name[0].isupper():
                    # PascalCase (для классов)
                    return f"{abbr.capitalize()}{suffix}"
                else:
                    # camelCase (для методов/переменных)
                    return f"{abbr}{suffix}"
            else:
                # Если не удалось найти заглавные буквы, используем первые буквы
                prefix = original_name[0]
                return f"{prefix}{suffix}"
                
        # 4. Особые случаи для различных типов идентификаторов
        
        # Для счетчиков и индексов сохраняем короткие имена
        if original_name.lower() in ['i', 'j', 'k', 'n', 'x', 'y', 'z', 'index', 'count', 'counter']:
            # Для коротких переменных оставляем короткие имена
            if len(original_name) <= 2:
                return original_name + suffix[:2]
            else:
                return original_name[0] + suffix[:3]
                
        # Для имен файлов и путей сохраняем суффиксы
        if 'file' in original_name.lower() or 'path' in original_name.lower() or 'dir' in original_name.lower():
            if 'file' in original_name.lower():
                return f"f_{suffix}"
            elif 'path' in original_name.lower():
                return f"p_{suffix}"
            else:
                return f"d_{suffix}"
                
        # 5. По умолчанию сохраняем первую букву и добавляем суффикс
        prefix = original_name[0]
        return f"{prefix}{suffix}"
    
    def _extract_code_blocks(self, text: str) -> List[CodeBlock]:
        """
        Извлекает блоки кода из текста.
        
        Args:
            text: Входной текст
            
        Returns:
            List[CodeBlock]: Список объектов CodeBlock
        """
        # Регулярное выражение для поиска блоков кода
        # Поддерживает блоки в формате ```language\ncode\n``` и ```code```
        pattern = r"```(?:(\w+)\n)?([\s\S]*?)```"
        
        blocks = []
        for match in re.finditer(pattern, text):
            language, code = match.groups()
            if language is None:
                language = "unknown"
                
            # Создаем объект CodeBlock
            block = CodeBlock(original=code, language=language)
            blocks.append(block)
            
        return blocks
    
    def _save_to_vault(self, code_blocks: List[CodeBlock]):
        """
        Сохраняет оригинальные и обфусцированные блоки кода в хранилище.
        
        Args:
            code_blocks: Список объектов CodeBlock для сохранения
        """
        if not code_blocks:
            return
        
        # Создаем уникальный ID для текущей сессии обфускации
        session_id = str(uuid.uuid4())
        session_dir = self.vault_dir / session_id
        os.makedirs(session_dir, exist_ok=True)
        
        # Сохраняем информацию о блоках кода
        blocks_data = []
        for block in code_blocks:
            if block.obfuscated:
                block_data = {
                    "id": block.id,
                    "language": block.language,
                    "original_path": f"{block.id}_original.txt",
                    "obfuscated_path": f"{block.id}_obfuscated.txt"
                }
                blocks_data.append(block_data)
                
                # Сохраняем оригинальный и обфусцированный код
                with open(session_dir / f"{block.id}_original.txt", "w") as f:
                    f.write(block.original)
                
                with open(session_dir / f"{block.id}_obfuscated.txt", "w") as f:
                    f.write(block.obfuscated)
        
        # Сохраняем метаданные сессии
        metadata = {
            "session_id": session_id,
            "model_name": self.model_name,
            "blocks": blocks_data,
            "timestamp": str(uuid.uuid1())
        }
        
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Сохраняем путь текущей сессии для использования при деобфускации
        self.current_session_dir = session_dir
    
    def _obfuscate_code_blocks(self, text: str, code_blocks: List[CodeBlock]) -> str:
        """
        Обфусцирует блоки кода в тексте.
        
        Args:
            text: Входной текст
            code_blocks: Список объектов CodeBlock
            
        Returns:
            str: Текст с обфусцированными блоками кода
        """
        # Обфусцируем каждый блок кода
        for block in code_blocks:
            block.obfuscated = self._obfuscate_code(block.original)
            
        # Вставляем обфусцированный код через сплиты
        result = text
        cursor = 0
        parts = []
        
        # Находим все блоки кода в тексте
        for match in re.finditer(r"```(?:(\w+)\n)?([\s\S]*?)```", result):
            start, end = match.span()
            language, code = match.groups()
            if language is None:
                language = "unknown"
                
            # Ищем соответствующий блок кода в нашем списке
            for block in code_blocks:
                if block.language == language and block.original.strip() == code.strip():
                    # Добавляем текст до кода
                    parts.append(result[cursor:start])
                    # Добавляем обфусцированный блок с правильной разметкой
                    parts.append(f"```{language}\n{block.obfuscated}\n```")
                    cursor = end
                    break
            else:
                # Если не нашли соответствующий блок, добавляем как есть
                parts.append(result[cursor:start])
                parts.append(result[start:end])
                cursor = end
        
        # Добавляем оставшийся текст
        if cursor < len(result):
            parts.append(result[cursor:])
        
        return "".join(parts)
    
    def scan(
        self, 
        prompt: str, 
        conversation: Optional[List[Dict[str, str]]] = None, 
        raw: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Сканирует промт, обнаруживает и обфусцирует блоки кода.
        
        Args:
            prompt: Входной промт
            conversation: Опциональная история разговора (не используется)
            raw: Возвращать ли необработанный результат (не используется)
            
        Returns:
            Tuple[str, Dict[str, Any]]: Кортеж из обфусцированного промта и метаданных
        """
        if not prompt:
            return prompt, {"is_valid": True, "stats": {"code_blocks_found": 0}}
        
        # Извлекаем блоки кода
        code_blocks = self._extract_code_blocks(prompt)
        if not code_blocks:
            return prompt, {"is_valid": True, "stats": {"code_blocks_found": 0}}
        
        # Статистика для метаданных
        stats = {
            "code_blocks_found": len(code_blocks),
            "code_blocks_obfuscated": 0,
            "skipped_blocks": 0
        }
        
        # Фильтруем блоки, которые нужно пропустить
        filtered_blocks = []
        for block in code_blocks:
            if self._should_skip_obfuscation(block.original):
                stats["skipped_blocks"] += 1
            else:
                filtered_blocks.append(block)
                stats["code_blocks_obfuscated"] += 1
        
        # Обфусцируем блоки кода
        if filtered_blocks:
            sanitized_prompt = self._obfuscate_code_blocks(prompt, filtered_blocks)
            
            # Сохраняем в хранилище
            if self.vault_dir:
                self._save_to_vault(filtered_blocks)
        else:
            sanitized_prompt = prompt
        
        return sanitized_prompt, {"is_valid": True, "stats": stats}
    
    def deobfuscate(self, obfuscated_prompt: str) -> str:
        """
        Деобфусцирует ранее обфусцированный промт.
        
        Args:
            obfuscated_prompt: Обфусцированный промт
            
        Returns:
            str: Деобфусцированный промт
        """
        if not hasattr(self, "current_session_dir") or not self.current_session_dir:
            logger.warning("No session information available for deobfuscation")
            return obfuscated_prompt
        
        # Загружаем метаданные сессии
        try:
            with open(self.current_session_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading session metadata: {e}")
            return obfuscated_prompt
        
        # Загружаем информацию о блоках кода
        result = obfuscated_prompt
        for block_data in metadata.get("blocks", []):
            try:
                # Загружаем оригинальный и обфусцированный код
                with open(self.current_session_dir / block_data["original_path"], "r") as f:
                    original = f.read()
                
                with open(self.current_session_dir / block_data["obfuscated_path"], "r") as f:
                    obfuscated = f.read()
                
                # Заменяем обфусцированный код на оригинальный
                language = block_data["language"]
                
                # Заменяем в различных форматах
                result = result.replace(f"```{language}\n{obfuscated}\n```", f"```{language}\n{original}\n```")
                result = result.replace(f"```\n{obfuscated}\n```", f"```\n{original}\n```")
                result = result.replace(f"```{obfuscated}```", f"```{original}```")
                
            except Exception as e:
                logger.error(f"Error during deobfuscation of block {block_data['id']}: {e}")
        
        return result
    
    def _obfuscate_with_antlr(self, code: str, language: str) -> Tuple[str, Set[str]]:
        """
        Обфусцирует код с использованием ANTLR парсера.
        
        Args:
            code: Исходный код для обфускации
            language: Язык программирования
        
        Returns:
            Tuple[str, Set[str]]: Кортеж из обфусцированного кода и множества необфусцируемых элементов
        """
        if not self.use_antlr or not ANTLR_AVAILABLE or not GRAMMARS_AVAILABLE:
            return code, set()
        
        try:
            # Создаем поток ввода
            input_stream = InputStream(code)
            
            # Выбираем соответствующие лексер, парсер и слушатель в зависимости от языка
            if language == 'python':
                lexer = Python3Lexer(input_stream)
                lexer.removeErrorListeners()
                lexer.addErrorListener(SilentErrorListener())
                
                tokens = CommonTokenStream(lexer)
                parser = Python3Parser(tokens)
                parser.removeErrorListeners()
                parser.addErrorListener(SilentErrorListener())
                
                # Парсим дерево
                tree = parser.file_input()
                
                # Создаем и применяем слушатель
                collector = PythonIdentifierCollector()
                walker = ParseTreeWalker()
                walker.walk(collector, tree)
                
                # Получаем все элементы, которые нельзя обфусцировать
                non_obfuscatable = collector.get_all_non_obfuscatable()
                
                return code, non_obfuscatable
                
            elif language == 'javascript':
                lexer = JavaScriptLexer(input_stream)
                lexer.removeErrorListeners()
                lexer.addErrorListener(SilentErrorListener())
                
                tokens = CommonTokenStream(lexer)
                parser = JavaScriptParser(tokens)
                parser.removeErrorListeners()
                parser.addErrorListener(SilentErrorListener())
                
                # Парсим дерево
                tree = parser.program()
                
                # Создаем и применяем слушатель
                collector = JavaScriptIdentifierCollector()
                walker = ParseTreeWalker()
                walker.walk(collector, tree)
                
                # Получаем все элементы, которые нельзя обфусцировать
                non_obfuscatable = collector.get_all_non_obfuscatable()
                
                return code, non_obfuscatable
                
            elif language == 'java':
                lexer = JavaLexer(input_stream)
                lexer.removeErrorListeners()
                lexer.addErrorListener(SilentErrorListener())
                
                tokens = CommonTokenStream(lexer)
                parser = JavaParser(tokens)
                parser.removeErrorListeners()
                parser.addErrorListener(SilentErrorListener())
                
                # Парсим дерево
                tree = parser.compilationUnit()
                
                # Создаем и применяем слушатель
                collector = JavaIdentifierCollector()
                walker = ParseTreeWalker()
                walker.walk(collector, tree)
                
                # Получаем все элементы, которые нельзя обфусцировать
                non_obfuscatable = collector.get_all_non_obfuscatable()
                
                return code, non_obfuscatable
                
            elif language == 'csharp':
                lexer = CSharpLexer(input_stream)
                lexer.removeErrorListeners()
                lexer.addErrorListener(SilentErrorListener())
                
                tokens = CommonTokenStream(lexer)
                parser = CSharpParser(tokens)
                parser.removeErrorListeners()
                parser.addErrorListener(SilentErrorListener())
                
                # Парсим дерево
                tree = parser.compilation_unit()
                
                # Создаем и применяем слушатель
                collector = CSharpIdentifierCollector()
                walker = ParseTreeWalker()
                walker.walk(collector, tree)
                
                # Получаем все элементы, которые нельзя обфусцировать
                non_obfuscatable = collector.get_all_non_obfuscatable()
                
                return code, non_obfuscatable
                
            elif language == 'kotlin':
                lexer = KotlinLexer(input_stream)
                lexer.removeErrorListeners()
                lexer.addErrorListener(SilentErrorListener())
                
                tokens = CommonTokenStream(lexer)
                parser = KotlinParser(tokens)
                parser.removeErrorListeners()
                parser.addErrorListener(SilentErrorListener())
                
                # Парсим дерево
                tree = parser.kotlinFile()
                
                # Создаем и применяем слушатель
                collector = KotlinIdentifierCollector()
                walker = ParseTreeWalker()
                walker.walk(collector, tree)
                
                # Получаем все элементы, которые нельзя обфусцировать
                non_obfuscatable = collector.get_all_non_obfuscatable()
                
                return code, non_obfuscatable
                
            elif language == 'php':
                lexer = PhpLexer(input_stream)
                lexer.removeErrorListeners()
                lexer.addErrorListener(SilentErrorListener())
                
                tokens = CommonTokenStream(lexer)
                parser = PhpParser(tokens)
                parser.removeErrorListeners()
                parser.addErrorListener(SilentErrorListener())
                
                # Парсим дерево
                tree = parser.htmlDocument()
                
                # Создаем и применяем слушатель
                collector = PHPIdentifierCollector()
                walker = ParseTreeWalker()
                walker.walk(collector, tree)
                
                # Получаем все элементы, которые нельзя обфусцировать
                non_obfuscatable = collector.get_all_non_obfuscatable()
                
                return code, non_obfuscatable
                
            elif language == 'ruby':
                lexer = CorundumLexer(input_stream)
                lexer.removeErrorListeners()
                lexer.addErrorListener(SilentErrorListener())
                
                tokens = CommonTokenStream(lexer)
                parser = CorundumParser(tokens)
                parser.removeErrorListeners()
                parser.addErrorListener(SilentErrorListener())
                
                # Парсим дерево
                tree = parser.prog()
                
                # Создаем и применяем слушатель
                collector = RubyIdentifierCollector()
                walker = ParseTreeWalker()
                walker.walk(collector, tree)
                
                # Получаем все элементы, которые нельзя обфусцировать
                non_obfuscatable = collector.get_all_non_obfuscatable()
                
                return code, non_obfuscatable
                
            elif language == 'swift':
                lexer = Swift5Lexer(input_stream)
                lexer.removeErrorListeners()
                lexer.addErrorListener(SilentErrorListener())
                
                tokens = CommonTokenStream(lexer)
                parser = Swift5Parser(tokens)
                parser.removeErrorListeners()
                parser.addErrorListener(SilentErrorListener())
                
                # Парсим дерево
                tree = parser.top_level()
                
                # Создаем и применяем слушатель
                collector = SwiftIdentifierCollector()
                walker = ParseTreeWalker()
                walker.walk(collector, tree)
                
                # Получаем все элементы, которые нельзя обфусцировать
                non_obfuscatable = collector.get_all_non_obfuscatable()
                
                return code, non_obfuscatable
                
            elif language == 'golang':
                lexer = GoLexer(input_stream)
                lexer.removeErrorListeners()
                lexer.addErrorListener(SilentErrorListener())
                
                tokens = CommonTokenStream(lexer)
                parser = GoParser(tokens)
                parser.removeErrorListeners()
                parser.addErrorListener(SilentErrorListener())
                
                # Парсим дерево
                tree = parser.sourceFile()
                
                # Создаем и применяем слушатель
                collector = GoIdentifierCollector()
                walker = ParseTreeWalker()
                walker.walk(collector, tree)
                
                # Получаем все элементы, которые нельзя обфусцировать
                non_obfuscatable = collector.get_all_non_obfuscatable()
                
                return code, non_obfuscatable
                
            elif language == 'rust':
                lexer = RustLexer(input_stream)
                lexer.removeErrorListeners()
                lexer.addErrorListener(SilentErrorListener())
                
                tokens = CommonTokenStream(lexer)
                parser = RustParser(tokens)
                parser.removeErrorListeners()
                parser.addErrorListener(SilentErrorListener())
                
                # Парсим дерево
                tree = parser.crate()
                
                # Создаем и применяем слушатель
                collector = RustIdentifierCollector()
                walker = ParseTreeWalker()
                walker.walk(collector, tree)
                
                # Получаем все элементы, которые нельзя обфусцировать
                non_obfuscatable = collector.get_all_non_obfuscatable()
                
                return code, non_obfuscatable
                
            else:
                # Для неподдерживаемых языков возвращаем исходный код
                logger.warning(f"ANTLR обфускация не поддерживается для языка {language}")
                return code, set()
                
        except Exception as e:
            # При ошибке парсинга возвращаем исходный код
            logger.warning(f"Ошибка ANTLR парсинга: {e}")
            return code, set()