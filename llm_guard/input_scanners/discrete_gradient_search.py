"""
Реализация алгоритма дискретного градиентного поиска из статьи CodeCipher.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional, Callable, Any
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class DiscreteGradientSearch:
    """
    Реализация дискретного градиентного поиска для оптимизации отображения токенов.
    
    Основан на методологии, описанной в статье 'CodeCipher: Learning to Obfuscate Source Code Against LLMs'.
    """
    
    def __init__(
        self,
        embedding_model,
        tokenizer,
        learning_rate: float = 0.002,
        max_iterations: int = 100,
        max_steps: int = 10,
        alpha: float = 1.5,
        beta: float = 10,
        perplexity_threshold: float = 50.0,
        similarity_threshold: float = 0.7
    ):
        """
        Инициализация алгоритма дискретного градиентного поиска.
        
        Args:
            embedding_model: Модель для получения эмбеддингов
            tokenizer: Токенизатор для модели
            learning_rate: Скорость обучения
            max_iterations: Максимальное число итераций
            max_steps: Максимальное число шагов внутри одной итерации
            alpha: Коэффициент для вычисления порога перплексии
            beta: Начальный порог перплексии
            perplexity_threshold: Порог перплексии для раннего останова
            similarity_threshold: Порог косинусной схожести
        """
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.max_steps = max_steps
        self.alpha = alpha
        self.beta = beta
        self.perplexity_threshold = perplexity_threshold
        self.similarity_threshold = similarity_threshold
    
    def _compute_perplexity(self, code: str) -> float:
        """
        Вычисляет перплексию для кода.
        
        Args:
            code: Исходный код
            
        Returns:
            Значение перплексии
        """
        try:
            # Токенизируем код
            inputs = self.tokenizer(code, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            # Вычисляем перплексию (упрощенно)
            with torch.no_grad():
                # Проверяем, поддерживает ли модель аргумент labels
                if hasattr(self.embedding_model, "config") and hasattr(self.embedding_model.config, "is_decoder") and self.embedding_model.config.is_decoder:
                    # Для моделей декодеров (например, GPT), можно использовать labels
                    outputs = self.embedding_model(input_ids, labels=input_ids)
                    loss = outputs.loss
                else:
                    # Для моделей, которые не поддерживают labels (например, базовый GPT2Model)
                    # Рассчитываем перплексию вручную
                    outputs = self.embedding_model(input_ids)
                    logits = outputs.last_hidden_state
                    # Используем простую метрику - среднее значение активаций
                    # Это не настоящая перплексия, но может служить прокси для сложности
                    pseudo_perplexity = torch.mean(torch.abs(logits)).item()
                    return pseudo_perplexity
                
                perplexity = torch.exp(loss).item()
            
            return perplexity
        except Exception as e:
            logger.error(f"Ошибка вычисления перплексии: {e}")
            # Возвращаем значение по умолчанию в случае ошибки
            return 1.0
    
    def _compute_task_loss(self, original_code: str, obfuscated_code: str, task_specific_fn: Optional[Callable] = None) -> float:
        """
        Вычисляет потерю для конкретной задачи.
        
        Args:
            original_code: Исходный код
            obfuscated_code: Обфусцированный код
            task_specific_fn: Функция для вычисления потери для конкретной задачи
            
        Returns:
            Значение потери
        """
        if task_specific_fn is not None:
            # Используем переданную функцию
            return task_specific_fn(original_code, obfuscated_code)
        
        # По умолчанию используем разницу в перплексии
        original_ppl = self._compute_perplexity(original_code)
        obfuscated_ppl = self._compute_perplexity(obfuscated_code)
        
        # Целевая функция: увеличение перплексии при сохранении синтаксической структуры
        loss = -np.log(obfuscated_ppl / original_ppl)
        
        return loss
    
    def _get_token_embedding(self, token: str) -> np.ndarray:
        """
        Получает эмбеддинг для токена.
        
        Args:
            token: Токен
            
        Returns:
            Векторное представление токена
        """
        # Токенизируем токен
        inputs = self.tokenizer(token, return_tensors="pt")
        
        # Получаем эмбеддинг
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Берем среднее значение по всем токенам
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        return embedding
    
    def _find_closest_token(self, embedding: np.ndarray, valid_tokens: List[str]) -> str:
        """
        Находит ближайший токен в пространстве эмбеддингов.
        
        Args:
            embedding: Эмбеддинг
            valid_tokens: Список допустимых токенов
            
        Returns:
            Ближайший токен
        """
        if not valid_tokens:
            return None
        
        # Получаем эмбеддинги для всех допустимых токенов
        valid_token_embeddings = [self._get_token_embedding(token) for token in valid_tokens]
        
        # Вычисляем косинусное сходство
        similarities = cosine_similarity([embedding], valid_token_embeddings)[0]
        
        # Находим наиболее похожий токен
        most_similar_idx = np.argmax(similarities)
        
        return valid_tokens[most_similar_idx]
    
    def _projection_operation(self, embedding: np.ndarray, vocabulary: List[str]) -> Tuple[str, np.ndarray]:
        """
        Проекция эмбеддинга на ближайший токен в словаре.
        
        Args:
            embedding: Эмбеддинг
            vocabulary: Список токенов словаря
            
        Returns:
            Кортеж (ближайший токен, эмбеддинг ближайшего токена)
        """
        closest_token = self._find_closest_token(embedding, vocabulary)
        closest_embedding = self._get_token_embedding(closest_token)
        
        return closest_token, closest_embedding
    
    def _apply_mapping(self, code: str, token_mapping: Dict[str, str]) -> str:
        """
        Применяет отображение токенов к коду.
        
        Args:
            code: Исходный код
            token_mapping: Отображение токенов
            
        Returns:
            Обфусцированный код
        """
        # Упрощенная реализация: разбиваем код на токены и заменяем их
        tokens = code.replace('(', ' ( ').replace(')', ' ) ').replace('{', ' { ').replace('}', ' } ').split()
        
        obfuscated_tokens = []
        for token in tokens:
            if token in token_mapping:
                obfuscated_tokens.append(token_mapping[token])
            else:
                obfuscated_tokens.append(token)
        
        return ' '.join(obfuscated_tokens)
    
    def optimize_mapping(
        self,
        code: str,
        original_mapping: Dict[str, str],
        vocabulary: List[str],
        categories: Dict[str, str],
        task_specific_fn: Optional[Callable] = None
    ) -> Dict[str, str]:
        """
        Оптимизирует отображение токенов с помощью дискретного градиентного поиска.
        
        Args:
            code: Исходный код
            original_mapping: Начальное отображение токенов
            vocabulary: Словарь допустимых токенов
            categories: Категории токенов
            task_specific_fn: Функция для вычисления потери для конкретной задачи
            
        Returns:
            Оптимизированное отображение токенов
        """
        current_mapping = original_mapping.copy()
        best_mapping = original_mapping.copy()
        best_loss = float('inf')
        
        # Получаем начальную перплексию
        original_ppl = self._compute_perplexity(code)
        
        for iteration in range(self.max_iterations):
            # Вычисляем порог перплексии для текущей итерации
            iteration_threshold = self.alpha * iteration + self.beta
            
            # Применяем текущее отображение к коду
            obfuscated_code = self._apply_mapping(code, current_mapping)
            
            # Вычисляем перплексию обфусцированного кода
            obfuscated_ppl = self._compute_perplexity(obfuscated_code)
            
            # Проверяем порог перплексии
            ppl_diff = obfuscated_ppl - original_ppl
            if ppl_diff > self.perplexity_threshold and ppl_diff > iteration_threshold:
                logger.info(f"Ранний останов на итерации {iteration}: перплексия {obfuscated_ppl} превысила порог")
                break
            
            # Вычисляем потерю для текущего отображения
            current_loss = self._compute_task_loss(code, obfuscated_code, task_specific_fn)
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_mapping = current_mapping.copy()
            
            # Для каждого токена в отображении
            for token in list(current_mapping.keys()):
                # Пропускаем токены, которые не должны быть обфусцированы
                if categories.get(token) == 'non_obfuscatable':
                    continue
                
                # Получаем допустимые замены для данного токена на основе его категории
                valid_replacements = [t for t in vocabulary if categories.get(t, 'unknown') == categories.get(token, 'unknown')]
                
                # Если нет допустимых замен, пропускаем токен
                if not valid_replacements:
                    continue
                
                # Получаем эмбеддинг текущего отображенного токена
                current_embedding = self._get_token_embedding(current_mapping[token])
                
                # Дискретный градиентный поиск
                best_token_loss = float('inf')
                best_token_replacement = current_mapping[token]
                
                # Выполняем несколько шагов градиентного поиска
                for step in range(self.max_steps):
                    # Создаем временное отображение с изменениями для текущего токена
                    temp_mapping = current_mapping.copy()
                    
                    # Проверяем случайные замены из допустимых
                    for _ in range(min(5, len(valid_replacements))):
                        replacement = np.random.choice(valid_replacements)
                        temp_mapping[token] = replacement
                        
                        # Применяем временное отображение
                        temp_obfuscated_code = self._apply_mapping(code, temp_mapping)
                        
                        # Вычисляем потерю
                        temp_loss = self._compute_task_loss(code, temp_obfuscated_code, task_specific_fn)
                        
                        if temp_loss < best_token_loss:
                            best_token_loss = temp_loss
                            best_token_replacement = replacement
                
                # Обновляем отображение для текущего токена
                current_mapping[token] = best_token_replacement
        
        return best_mapping 