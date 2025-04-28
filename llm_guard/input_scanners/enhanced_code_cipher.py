"""
Расширенный обфускатор кода, объединяющий подходы ANTLR и CodeCipher.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Импортируем переменную ANTLR_AVAILABLE из модуля antlr_grammars
from llm_guard.input_scanners.antlr_grammars import ANTLR_AVAILABLE

# Проверяем доступность ANTLR и импортируем необходимые компоненты
if ANTLR_AVAILABLE:
    try:
        from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker, ErrorListener
        
        # Определяем SilentErrorListener здесь для избежания проблем с импортом
        class SilentErrorListener(ErrorListener):
            def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
                pass
            
            def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigSet, configs):
                pass
            
            def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
                pass
            
            def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
                pass
    except ImportError:
        ANTLR_AVAILABLE = False
else:
    # Создаем пустой класс-заглушку если ANTLR не доступен
    class SilentErrorListener:
        pass

from llm_guard.input_scanners.code_cipher import CodeCipherObfuscator
from llm_guard.input_scanners.antlr_grammars import COLLECTOR_MAP, SUPPORTED_LANGUAGES, AVAILABLE_GRAMMARS
from llm_guard.input_scanners.discrete_gradient_search import DiscreteGradientSearch

# Импорты для конкретных языков будут добавляться динамически

logger = logging.getLogger(__name__)

class EnhancedCodeCipherObfuscator(CodeCipherObfuscator):
    """
    Расширенный обфускатор кода, объединяющий ANTLR для синтаксического анализа
    и методологию CodeCipher для обучения оптимального токен-отображения.
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
        use_antlr: bool = True,
        embedding_dim: int = 768,
        confusion_mapping_path: Optional[str] = None,
        similarity_threshold: float = 0.7
    ):
        """
        Инициализация расширенного обфускатора кода.
        
        Args:
            model_name: Название модели HuggingFace для эмбеддингов
            max_training_iterations: Максимальное число итераций обучения
            learning_rate: Скорость обучения для градиентного поиска
            perplexity_threshold: Порог перплексии для раннего останова
            early_stopping_patience: Терпение для раннего останова
            task_specific_dataset: Набор данных для обучения
            vault_dir: Директория для хранения обфусцированного кода
            skip_patterns: Паттерны для пропуска обфускации
            use_antlr: Использовать ли ANTLR когда доступен
            embedding_dim: Размерность эмбеддингов
            confusion_mapping_path: Путь к предварительно обученной матрице конфузии
            similarity_threshold: Порог косинусной схожести для замены токенов
        """
        super().__init__(
            model_name,
            max_training_iterations,
            learning_rate,
            perplexity_threshold,
            early_stopping_patience,
            task_specific_dataset,
            vault_dir,
            skip_patterns,
            use_antlr
        )
        
        # Параметры для матрицы конфузии
        self.embedding_dim = embedding_dim
        self.confusion_mapping_path = confusion_mapping_path
        self.similarity_threshold = similarity_threshold
        
        # Инициализация модели для эмбеддингов
        self.embedding_model = None
        self.tokenizer = None
        
        # Словари отображения токенов
        self.token_mapping = {}  # Отображение оригинальных токенов на обфусцированные
        self.reverse_mapping = {}  # Обратное отображение для деобфускации
        
        # Матрица конфузии
        self.confusion_matrix = None
        self.original_embeddings = None
        self.confused_embeddings = None
        
        # Загрузить предварительно обученное отображение, если доступно
        if confusion_mapping_path and os.path.exists(confusion_mapping_path):
            self._load_confusion_mapping(confusion_mapping_path)
    
    def _load_model(self):
        """Загрузить модель для эмбеддингов"""
        if self.embedding_model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.embedding_model = AutoModel.from_pretrained(self.model_name)
                logger.info(f"Модель {self.model_name} успешно загружена")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели {self.model_name}: {e}")
                # Fallback на простую матрицу эмбеддингов
                self.embedding_model = None
                self.tokenizer = None
    
    def _get_token_embedding(self, token: str) -> np.ndarray:
        """Получить эмбеддинг для токена"""
        if self.embedding_model is None:
            self._load_model()
        
        if self.embedding_model is None:
            # Fallback: создаем случайный вектор
            return np.random.randn(self.embedding_dim)
        
        # Токенизируем и получаем эмбеддинг
        inputs = self.tokenizer(token, return_tensors="pt")
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        
        # Берем среднее значение по всем токенам
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings
    
    def _load_confusion_mapping(self, path: str):
        """Загрузить предварительно обученное отображение конфузии"""
        try:
            data = torch.load(path)
            self.token_mapping = data.get('token_mapping', {})
            self.reverse_mapping = data.get('reverse_mapping', {})
            self.original_embeddings = data.get('original_embeddings', None)
            self.confused_embeddings = data.get('confused_embeddings', None)
            self.confusion_matrix = data.get('confusion_matrix', None)
            logger.info(f"Матрица конфузии успешно загружена из {path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки матрицы конфузии из {path}: {e}")
    
    def _save_confusion_mapping(self, path: str):
        """Сохранить обученное отображение конфузии"""
        data = {
            'token_mapping': self.token_mapping,
            'reverse_mapping': self.reverse_mapping,
            'original_embeddings': self.original_embeddings,
            'confused_embeddings': self.confused_embeddings,
            'confusion_matrix': self.confusion_matrix
        }
        try:
            torch.save(data, path)
            logger.info(f"Матрица конфузии успешно сохранена в {path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения матрицы конфузии в {path}: {e}")
    
    def _extract_tokens_with_antlr(self, code: str) -> Tuple[List[str], Dict[str, str], Set[str]]:
        """
        Извлечь токены и их категории с помощью ANTLR.
        
        Args:
            code: Исходный код
            
        Returns:
            Кортеж из списка токенов, словаря категорий токенов и множества необфусцируемых элементов
        """
        if not ANTLR_AVAILABLE:
            logger.warning("ANTLR не доступен. Используется наивный анализ токенов.")
            # Наивный анализ токенов
            tokens = code.replace('(', ' ( ').replace(')', ' ) ').replace('{', ' { ').replace('}', ' } ').split()
            categories = {token: 'obfuscatable' for token in tokens}
            non_obfuscatable = set()
            return tokens, categories, non_obfuscatable
        
        language = self._detect_language(code)
        if language not in AVAILABLE_GRAMMARS:
            logger.warning(f"Язык {language} не поддерживается ANTLR. Используется наивный анализ токенов.")
            # Наивный анализ токенов
            tokens = code.replace('(', ' ( ').replace(')', ' ) ').replace('{', ' { ').replace('}', ' } ').split()
            categories = {token: 'obfuscatable' for token in tokens}
            non_obfuscatable = set()
            return tokens, categories, non_obfuscatable
        
        # Результат обфускации с помощью ANTLR
        try:
            obfuscated_code, non_obfuscatable = self._obfuscate_with_antlr(code, language)
            
            # Создаем упрощенные категории на основе non_obfuscatable
            tokens = []
            categories = {}
            
            # Разбиваем код на токены (упрощенно)
            simple_tokens = obfuscated_code.replace('(', ' ( ').replace(')', ' ) ').replace('{', ' { ').replace('}', ' } ').split()
            
            for token in simple_tokens:
                tokens.append(token)
                if token in non_obfuscatable:
                    categories[token] = 'non_obfuscatable'
                else:
                    categories[token] = 'obfuscatable'
            
            return tokens, categories, non_obfuscatable
        except Exception as e:
            logger.error(f"Ошибка при анализе кода с помощью ANTLR: {e}")
            # Наивный анализ токенов в случае ошибки
            tokens = code.replace('(', ' ( ').replace(')', ' ) ').replace('{', ' { ').replace('}', ' } ').split()
            categories = {token: 'obfuscatable' for token in tokens}
            non_obfuscatable = set()
            return tokens, categories, non_obfuscatable
    
    def _discrete_gradient_search(self, original_mapping, code, target=None):
        """
        Выполняет дискретный градиентный поиск для оптимизации отображения.
        
        Args:
            original_mapping: Исходное отображение токенов
            code: Исходный код
            target: Целевой код (для задач перевода)
            
        Returns:
            Оптимизированное отображение токенов
        """
        if self.embedding_model is None:
            self._load_model()
            
        if self.embedding_model is None:
            logger.warning("Модель эмбеддингов не доступна. Использую исходное отображение.")
            return original_mapping
        
        # Извлекаем токены и их категории
        tokens, categories, non_obfuscatable = self._extract_tokens_with_antlr(code)
        
        # Создаем словарь допустимых токенов
        vocabulary = list(set(tokens))
        
        # Инициализируем дискретный градиентный поиск
        gradient_search = DiscreteGradientSearch(
            embedding_model=self.embedding_model,
            tokenizer=self.tokenizer,
            learning_rate=self.learning_rate,
            perplexity_threshold=self.perplexity_threshold
        )
        
        # Оптимизируем отображение
        optimized_mapping = gradient_search.optimize_mapping(
            code=code,
            original_mapping=original_mapping,
            vocabulary=vocabulary,
            categories=categories
        )
        
        return optimized_mapping
    
    def train_confusion_mapping(self, dataset: List[str], output_path: Optional[str] = None):
        """
        Обучить конфузионное отображение на основе набора данных кода.
        
        Args:
            dataset: Список строк кода для обучения
            output_path: Путь для сохранения обученной матрицы
        """
        # Загрузить модель
        self._load_model()
        
        # Словари для хранения токенов и их эмбеддингов
        original_tokens = set()
        token_categories = {}
        token_embeddings = {}
        
        # Сначала извлекаем все уникальные токены и их категории
        for code_sample in dataset:
            tokens, categories, non_obfuscatable = self._extract_tokens_with_antlr(code_sample)
            
            for token in tokens:
                original_tokens.add(token)
                token_categories[token] = categories.get(token, 'unknown')
        
        logger.info(f"Извлечено {len(original_tokens)} уникальных токенов")
        
        # Получаем эмбеддинги для всех токенов
        for token in original_tokens:
            token_embeddings[token] = self._get_token_embedding(token)
        
        # Создаем матрицы оригинальных токенов и их эмбеддингов
        original_tokens_list = list(original_tokens)
        original_embeddings_matrix = np.array([token_embeddings[token] for token in original_tokens_list])
        
        # Инициализируем матрицу конфузии случайной перестановкой оригинальных эмбеддингов
        indices = np.random.permutation(len(original_tokens_list))
        confused_tokens_list = [original_tokens_list[i] for i in indices]
        confused_embeddings_matrix = np.array([token_embeddings[token] for token in confused_tokens_list])
        
        # Создаем базовое отображение
        token_mapping = {
            original_tokens_list[i]: confused_tokens_list[i] 
            for i in range(len(original_tokens_list))
        }
        
        # Оптимизация с помощью дискретного градиентного поиска
        optimized_mapping = token_mapping.copy()
        for i, code_sample in enumerate(dataset[:min(10, len(dataset))]):
            logger.info(f"Оптимизация отображения на примере {i+1}/{min(10, len(dataset))}")
            optimized_mapping = self._discrete_gradient_search(optimized_mapping, code_sample)
        
        # Обновляем отображение
        self.token_mapping = optimized_mapping
        self.reverse_mapping = {v: k for k, v in optimized_mapping.items()}
        self.original_embeddings = original_embeddings_matrix
        self.confused_embeddings = confused_embeddings_matrix
        
        # Сохраняем обученное отображение, если указан путь
        if output_path:
            self._save_confusion_mapping(output_path)
        
        return optimized_mapping
    
    def obfuscate_with_confusion(self, code: str) -> str:
        """
        Обфусцировать код с использованием обученной матрицы конфузии.
        
        Args:
            code: Исходный код
            
        Returns:
            Обфусцированный код
        """
        if not self.token_mapping:
            logger.warning("Матрица конфузии не обучена. Использую стандартную обфускацию.")
            return self._obfuscate_code(code)
        
        language = self._detect_language(code)
        
        # Извлекаем токены с помощью ANTLR или наивного анализатора
        tokens, categories, non_obfuscatable = self._extract_tokens_with_antlr(code)
        
        if not tokens:
            logger.warning("Не удалось извлечь токены из кода. Использую стандартную обфускацию.")
            return self._obfuscate_code(code)
        
        # Применяем отображение к каждому токену
        obfuscated_tokens = []
        
        # Получаем список оригинальных токенов, если доступен
        original_tokens_list = list(self.token_mapping.keys()) if self.token_mapping else []
        
        for token in tokens:
            if token in non_obfuscatable:
                # Не обфусцируем токены, которые должны остаться нетронутыми
                obfuscated_tokens.append(token)
            elif token in self.token_mapping:
                # Применяем предварительно обученное отображение
                obfuscated_tokens.append(self.token_mapping[token])
            else:
                # Для новых токенов находим ближайший в пространстве эмбеддингов
                token_embedding = self._get_token_embedding(token)
                
                if self.original_embeddings is not None and len(self.original_embeddings) > 0 and len(original_tokens_list) > 0:
                    # Вычисляем косинусное сходство
                    try:
                        similarities = cosine_similarity([token_embedding], self.original_embeddings)[0]
                        most_similar_idx = np.argmax(similarities)
                        
                        if similarities[most_similar_idx] > self.similarity_threshold and most_similar_idx < len(original_tokens_list):
                            original_similar_token = original_tokens_list[most_similar_idx]
                            obfuscated_token = self.token_mapping.get(original_similar_token, token)
                            obfuscated_tokens.append(obfuscated_token)
                        else:
                            # Если нет достаточно похожего токена, оставляем оригинальный
                            obfuscated_tokens.append(token)
                    except Exception as e:
                        logger.error(f"Ошибка при вычислении косинусного сходства: {e}")
                        obfuscated_tokens.append(token)
                else:
                    # Если нет матрицы эмбеддингов, оставляем оригинальный токен
                    obfuscated_tokens.append(token)
        
        # Собираем обфусцированный код (упрощенно)
        obfuscated_code = ' '.join(obfuscated_tokens)
        
        # В реальной реализации здесь должна быть более сложная логика реконструкции кода
        # с учетом синтаксиса конкретного языка программирования
        
        return obfuscated_code
    
    def deobfuscate_with_confusion(self, obfuscated_code: str) -> str:
        """
        Деобфусцировать код с использованием обратной матрицы конфузии.
        
        Args:
            obfuscated_code: Обфусцированный код
            
        Returns:
            Деобфусцированный код
        """
        if not self.reverse_mapping:
            logger.warning("Обратная матрица конфузии не доступна. Использую стандартную деобфускацию.")
            return self.deobfuscate(obfuscated_code)
        
        try:
            # Разбиваем обфусцированный код на токены (упрощенно)
            tokens = obfuscated_code.replace('(', ' ( ').replace(')', ' ) ').replace('{', ' { ').replace('}', ' } ').split()
            
            # Применяем обратное отображение к каждому токену
            deobfuscated_tokens = []
            for token in tokens:
                if token in self.reverse_mapping:
                    deobfuscated_tokens.append(self.reverse_mapping[token])
                else:
                    deobfuscated_tokens.append(token)
            
            # Собираем деобфусцированный код (упрощенно)
            deobfuscated_code = ' '.join(deobfuscated_tokens)
            
            return deobfuscated_code
        except Exception as e:
            logger.error(f"Ошибка при деобфускации кода: {e}")
            # В случае ошибки, пытаемся использовать стандартную деобфускацию
            try:
                return self.deobfuscate(obfuscated_code)
            except:
                # Если и это не сработало, возвращаем исходный код
                return obfuscated_code
    
    def scan(self, prompt: str, conversation: Optional[List[Dict[str, str]]] = None, raw: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Переопределенный метод сканирования для использования улучшенной обфускации.
        
        Args:
            prompt: Текст промпта
            conversation: История предыдущей беседы
            raw: Вернуть необработанный результат

        Returns:
            Кортеж (обфусцированный_промпт, метаданные)
        """
        try:
            # Извлекаем блоки кода
            code_blocks = self._extract_code_blocks(prompt)
            
            if not code_blocks:
                # Если блоков кода нет, возвращаем оригинальный промпт
                return prompt, {"blocked": False, "obfuscated": False, "score": 0}
            
            # Обфусцируем каждый блок кода с использованием улучшенного метода
            for block in code_blocks:
                try:
                    if self.token_mapping:
                        # Если матрица конфузии обучена, используем ее
                        block.obfuscated = self.obfuscate_with_confusion(block.original)
                    else:
                        # Иначе используем стандартную обфускацию
                        block.obfuscated = self._obfuscate_code(block.original)
                except Exception as e:
                    logger.error(f"Ошибка при обфускации блока кода: {e}")
                    # В случае ошибки оставляем блок без изменений
                    block.obfuscated = block.original
            
            # Сохраняем обфусцированные блоки в хранилище, если указано
            try:
                if self.vault:
                    self._save_to_vault(code_blocks)
            except Exception as e:
                logger.error(f"Ошибка при сохранении в хранилище: {e}")
            
            # Заменяем блоки кода в оригинальном промпте
            obfuscated_prompt = self._obfuscate_code_blocks(prompt, code_blocks)
            
            return obfuscated_prompt, {
                "blocked": False,
                "obfuscated": True,
                "score": 1,
                "obfuscated_blocks": [
                    {"id": block.id, "language": block.language} 
                    for block in code_blocks
                ]
            }
        except Exception as e:
            # В случае критической ошибки возвращаем оригинальный промпт
            logger.error(f"Критическая ошибка в методе scan: {e}")
            return prompt, {"blocked": False, "obfuscated": False, "score": 0, "error": str(e)} 