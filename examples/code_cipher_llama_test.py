#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования качества ответов модели Llama на обфусцированные промпты с кодом.
Позволяет сравнить ответы на оригинальный и обфусцированный код и оценить эффективность обфускации.

Для работы скрипта нужна библиотека llama-cpp-python:
pip install llama-cpp-python

Для ускорения на GPU (опционально):
pip install llama-cpp-python --force-reinstall --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu121

Скрипт проверяет, насколько хорошо модель Llama может анализировать обфусцированный код
по сравнению с оригинальным, используя CodeCipherObfuscator.
"""

import os
import sys
import json
import time
import argparse
import logging
import difflib
import re
from typing import List, Dict, Tuple, Optional, Any, Union
import tempfile
from collections import Counter

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем CodeCipherObfuscator
from llm_guard.input_scanners.code_cipher import CodeCipherObfuscator

# Настраиваем логгирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Импортируем llama-cpp-python (должен быть установлен)
try:
    from llama_cpp import Llama
except ImportError:
    logger.error(
        "Библиотека llama-cpp-python не установлена. Установите с помощью:\n"
        "pip install llama-cpp-python\n"
        "Для ускорения на GPU используйте:\n"
        "pip install llama-cpp-python --force-reinstall --extra-index-url"
        " https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu121"
    )
    sys.exit(1)


class MockTokenizer:
    """
    Мок токенайзера для использования в CodeCipherObfuscator без загрузки модели.
    """
    def __init__(self):
        self.counter = 1000
        self.vocab = {}
        self.inverse_vocab = {}
        
    def encode(self, text):
        """Простая токенизация текста"""
        tokens = []
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = self.counter
                self.inverse_vocab[self.counter] = word
                self.counter += 1
            tokens.append(self.vocab[word])
        return tokens
    
    def decode(self, tokens):
        """Детокенизация списка токенов"""
        text = ""
        for token in tokens:
            if token in self.inverse_vocab:
                text += self.inverse_vocab[token]
        return text


class MockModel:
    """
    Мок модели для использования в CodeCipherObfuscator без загрузки модели.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids, **kwargs):
        # Возвращаем dummy логиты
        class Output:
            def __init__(self, logits):
                self.logits = logits
        
        # Создаем простые логиты для последнего токена
        logits = [[0.0] * 32000]
        return Output(logits)


def query_llama(model_path: str, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
    """
    Запрос к модели Llama и получение ответа.
    
    Args:
        model_path (str): Путь к модели Llama в формате GGUF
        prompt (str): Текст промпта для отправки модели
        max_tokens (int): Максимальное количество токенов в ответе
        temperature (float): Температура генерации (0.0-1.0)
        
    Returns:
        str: Ответ от модели
    """
    try:
        logger.info(f"Загрузка модели из {model_path}...")
        model = Llama(
            model_path=model_path,
            n_ctx=4096,  # Контекст
            n_gpu_layers=-1,  # Использовать все доступные GPU слои
        )
        
        logger.info("Отправка запроса к модели...")
        start_time = time.time()
        output = model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</answer>", "Human:", "USER:"],
        )
        end_time = time.time()
        
        logger.info(f"Получен ответ за {end_time - start_time:.2f} сек.")
        return output["choices"][0]["text"].strip()
    
    except Exception as e:
        logger.error(f"Ошибка при работе с моделью: {e}")
        if "Failed to load model" in str(e):
            logger.error("Убедитесь, что путь к модели корректен и модель в формате GGUF")
        elif "Prompt too long" in str(e):
            logger.error("Промпт слишком длинный для данного размера контекста модели")
        return f"ОШИБКА: {str(e)}"


def get_test_examples() -> List[Dict[str, str]]:
    """
    Возвращает список примеров кода для тестирования.
    
    Returns:
        List[Dict[str, str]]: Список словарей с именем, языком, кодом и вопросом.
    """
    return [
        {
            "name": "Python: Сортировка пузырьком",
            "language": "python",
            "code": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Пример использования
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = bubble_sort(numbers)
print(sorted_numbers)  # [11, 12, 22, 25, 34, 64, 90]
""",
            "question": "Объясните, как работает этот алгоритм и какова его сложность."
        },
        {
            "name": "JavaScript: Функция поиска",
            "language": "javascript",
            "code": """
function findElement(array, predicate) {
  if (!Array.isArray(array) || typeof predicate !== 'function') {
    throw new TypeError('Invalid arguments');
  }
  
  for (let i = 0; i < array.length; i++) {
    if (predicate(array[i], i, array)) {
      return array[i];
    }
  }
  
  return undefined;
}

// Пример использования
const users = [
  { id: 1, name: 'Alice', age: 25 },
  { id: 2, name: 'Bob', age: 30 },
  { id: 3, name: 'Charlie', age: 35 }
];

const user = findElement(users, user => user.age > 30);
console.log(user); // { id: 3, name: 'Charlie', age: 35 }
""",
            "question": "Для чего предназначена эта функция и как ее можно улучшить?"
        },
        {
            "name": "C++: Класс стека",
            "language": "cpp",
            "code": """
#include <iostream>
#include <vector>
#include <stdexcept>

template <typename T>
class Stack {
private:
    std::vector<T> elements;
    
public:
    void push(const T& element) {
        elements.push_back(element);
    }
    
    T pop() {
        if (elements.empty()) {
            throw std::runtime_error("Stack is empty");
        }
        
        T top_element = elements.back();
        elements.pop_back();
        return top_element;
    }
    
    T& top() {
        if (elements.empty()) {
            throw std::runtime_error("Stack is empty");
        }
        
        return elements.back();
    }
    
    bool empty() const {
        return elements.empty();
    }
    
    size_t size() const {
        return elements.size();
    }
};

int main() {
    Stack<int> stack;
    stack.push(1);
    stack.push(2);
    stack.push(3);
    
    std::cout << "Top element: " << stack.top() << std::endl;
    std::cout << "Size: " << stack.size() << std::endl;
    
    std::cout << "Popping: " << stack.pop() << std::endl;
    std::cout << "New top: " << stack.top() << std::endl;
    
    return 0;
}
""",
            "question": "Объясните реализацию этого класса стека и почему здесь используется шаблон."
        }
    ]


def create_prompt(language: str, code: str, question: str) -> str:
    """
    Создает промпт для модели на основе примера кода и вопроса.
    
    Args:
        language (str): Язык программирования
        code (str): Исходный код
        question (str): Вопрос о коде
        
    Returns:
        str: Текст промпта
    """
    return f"""Я покажу вам код на языке {language} и задам вопрос.
Пожалуйста, внимательно проанализируйте код и ответьте на вопрос.

```{language}
{code}
```

Вопрос: {question}

Дайте подробный и точный ответ.
<answer>
"""


def analyze_responses(original_response: str, obfuscated_response: str) -> Dict[str, Any]:
    """
    Анализирует различия между ответами на оригинальный и обфусцированный код.
    
    Args:
        original_response (str): Ответ на оригинальный код
        obfuscated_response (str): Ответ на обфусцированный код
        
    Returns:
        Dict[str, Any]: Статистика сравнения ответов
    """
    # Подсчет и анализ схожести ответов
    similarity_ratio = difflib.SequenceMatcher(None, original_response, obfuscated_response).ratio()
    similarity_percentage = similarity_ratio * 100
    
    # Анализ длины
    original_length = len(original_response)
    obfuscated_length = len(obfuscated_response)
    length_diff = abs(original_length - obfuscated_length)
    length_diff_percentage = (length_diff / max(original_length, obfuscated_length)) * 100
    
    # Выделение ключевых фраз (n-граммы слов)
    def extract_ngrams(text, n=2):
        words = re.findall(r'\b\w+\b', text.lower())
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    original_bigrams = set(extract_ngrams(original_response))
    obfuscated_bigrams = set(extract_ngrams(obfuscated_response))
    
    common_phrases = original_bigrams.intersection(obfuscated_bigrams)
    original_unique = original_bigrams - obfuscated_bigrams
    obfuscated_unique = obfuscated_bigrams - original_bigrams
    
    # Если ответы очень разные, найдем уникальные слова
    if len(common_phrases) < 10:
        original_words = set(re.findall(r'\b\w+\b', original_response.lower()))
        obfuscated_words = set(re.findall(r'\b\w+\b', obfuscated_response.lower()))
        common_words = original_words.intersection(obfuscated_words)
        original_unique = original_words - obfuscated_words
        obfuscated_unique = obfuscated_words - original_words
    
    return {
        "similarity_percentage": similarity_percentage,
        "original_length": original_length,
        "obfuscated_length": obfuscated_length,
        "length_diff": length_diff,
        "length_diff_percentage": length_diff_percentage,
        "common_phrases_count": len(common_phrases),
        "original_unique_count": len(original_unique),
        "obfuscated_unique_count": len(obfuscated_unique),
        "original_unique_samples": list(original_unique)[:5],
        "obfuscated_unique_samples": list(obfuscated_unique)[:5],
        "is_similar": similarity_percentage > 80,
    }


def display_results(example: Dict[str, str], original_response: str, obfuscated_response: str, 
                  analysis: Dict[str, Any], verbose: bool = False) -> None:
    """
    Выводит результаты анализа ответов на экран.
    
    Args:
        example (Dict[str, str]): Информация о тестовом примере
        original_response (str): Ответ на оригинальный код
        obfuscated_response (str): Ответ на обфусцированный код
        analysis (Dict[str, Any]): Статистика сравнения ответов
        verbose (bool): Флаг подробного вывода
    """
    print("=" * 80)
    print(f"ТЕСТ: {example['name']}")
    print("=" * 80)
    
    if verbose:
        print("\nОРИГИНАЛЬНЫЙ ОТВЕТ:")
        print("-" * 40)
        print(original_response)
        print("\nОТВЕТ НА ОБФУСЦИРОВАННЫЙ КОД:")
        print("-" * 40)
        print(obfuscated_response)
    
    print("\nСТАТИСТИКА ОТВЕТОВ:")
    print(f"- Сходство ответов: {analysis['similarity_percentage']:.2f}%")
    print(f"- Длина оригинального ответа: {analysis['original_length']} символов")
    print(f"- Длина ответа на обфусцированный код: {analysis['obfuscated_length']} символов")
    print(f"- Разница в длине: {analysis['length_diff']} символов ({analysis['length_diff_percentage']:.2f}%)")
    print(f"- Общих ключевых фраз: {analysis['common_phrases_count']}")
    print(f"- Уникальных фраз в оригинальном ответе: {analysis['original_unique_count']}")
    print(f"- Уникальных фраз в ответе на обфусцированный код: {analysis['obfuscated_unique_count']}")
    
    print("\nПРИМЕРЫ УНИКАЛЬНЫХ ФРАЗ В ОРИГИНАЛЬНОМ ОТВЕТЕ:")
    for phrase in analysis['original_unique_samples']:
        print(f"  - {phrase}")
    
    print("\nПРИМЕРЫ УНИКАЛЬНЫХ ФРАЗ В ОТВЕТЕ НА ОБФУСЦИРОВАННЫЙ КОД:")
    for phrase in analysis['obfuscated_unique_samples']:
        print(f"  - {phrase}")
    
    print("\nВЫВОД:")
    if analysis['is_similar']:
        print("✅ Ответы ПОХОЖИ. Обфускация успешно сохраняет способность модели анализировать код.")
    else:
        print("❌ Ответы РАЗЛИЧАЮТСЯ. Обфускация значительно влияет на анализ кода моделью.")
    
    print("\n")


def main():
    """
    Основная функция программы
    """
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Тестирование ответов Llama на обфусцированный код")
    parser.add_argument("--model", required=True, help="Путь к модели Llama в формате GGUF")
    parser.add_argument("--verbose", action="store_true", help="Показывать полные промпты и ответы")
    parser.add_argument("--examples", type=int, help="Номер примера для тестирования (начиная с 1)")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Максимум токенов в ответе")
    parser.add_argument("--temperature", type=float, default=0.7, help="Температура генерации (0.0-1.0)")
    parser.add_argument("--save", help="Путь для сохранения результатов в JSON-файл")
    args = parser.parse_args()
    
    # Проверка наличия модели
    if not os.path.exists(args.model):
        logger.error(f"Модель не найдена по пути: {args.model}")
        sys.exit(1)
    
    # Получаем список тестовых примеров
    examples = get_test_examples()
    
    # Создаем мок токенайзера и модели для обфускатора
    logger.info("Инициализация обфускатора с мок-моделью...")
    tokenizer = MockTokenizer()
    model = MockModel(tokenizer)
    
    # Создаем временную директорию для хранения vault
    with tempfile.TemporaryDirectory() as vault_dir:
        # Инициализируем CodeCipherObfuscator с мок-моделью
        scanner = CodeCipherObfuscator(
            model=model,
            tokenizer=tokenizer,
            max_training_iterations=10,  # минимальное значение для скорости
            vault_dir=vault_dir
        )
        
        # Хранение результатов для всех тестов
        all_results = []
        
        # Если указан конкретный пример
        if args.examples is not None:
            example_idx = args.examples - 1
            if 0 <= example_idx < len(examples):
                examples = [examples[example_idx]]
            else:
                logger.error(f"Неверный номер примера. Доступны примеры от 1 до {len(examples)}")
                sys.exit(1)
        
        # Проходим по всем примерам
        for i, example in enumerate(examples, 1):
            logger.info(f"Тест {i}/{len(examples)}: {example['name']}")
            
            # Создаем промпт с оригинальным кодом
            original_prompt = create_prompt(
                example["language"], 
                example["code"], 
                example["question"]
            )
            
            # Обфусцируем код и создаем промпт с обфусцированным кодом
            obfuscated_prompt, metadata = scanner.scan(original_prompt)
            
            if args.verbose:
                logger.info("Оригинальный промпт:")
                print(original_prompt)
                logger.info("Обфусцированный промпт:")
                print(obfuscated_prompt)
            
            # Запрашиваем ответы от модели
            logger.info("Получение ответа на оригинальный код...")
            original_response = query_llama(
                args.model, original_prompt, args.max_tokens, args.temperature
            )
            
            logger.info("Получение ответа на обфусцированный код...")
            obfuscated_response = query_llama(
                args.model, obfuscated_prompt, args.max_tokens, args.temperature
            )
            
            # Анализируем ответы
            analysis = analyze_responses(original_response, obfuscated_response)
            
            # Выводим результаты
            display_results(example, original_response, obfuscated_response, analysis, args.verbose)
            
            # Добавляем результаты в общий список
            result = {
                "example": example["name"],
                "original_response": original_response,
                "obfuscated_response": obfuscated_response,
                "analysis": analysis
            }
            all_results.append(result)
        
        # Сохраняем результаты в JSON-файл, если указан путь
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Результаты сохранены в файл: {args.save}")


if __name__ == "__main__":
    main() 