#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования качества ответов LLM на обфусцированные промпты с кодом.

Этот скрипт отправляет оригинальные и обфусцированные промпты с кодом в LLM через OpenAI API,
а затем сравнивает полученные ответы для оценки эффективности обфускации.
"""

import os
import sys
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse
from typing import Dict, List, Tuple, Any, Optional
import difflib
import re

# Добавляем родительскую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from llm_guard.input_scanners import CodeCipherObfuscator

# Проверяем наличие библиотеки OpenAI
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Библиотека OpenAI не установлена. Устанавливаю...")
    os.system("pip install openai")
    try:
        import openai
        from openai import OpenAI
        OPENAI_AVAILABLE = True
        print("Библиотека OpenAI успешно установлена.")
    except ImportError:
        print("Не удалось установить библиотеку OpenAI. Пожалуйста, установите её вручную: pip install openai")
        OPENAI_AVAILABLE = False

class MockTokenizer:
    """Мок для токенизатора."""
    
    def __init__(self):
        self._vocab = {f"token_{i}": i for i in range(100)}
        self.all_special_ids = [0, 1, 2]
        
    def encode(self, text, return_tensors=None):
        """Имитация токенизации."""
        result = [ord(c) + 1000 for c in text[:30]]
        if return_tensors == "pt":
            return MagicMock(tolist=lambda: [result])
        return result
    
    def decode(self, tokens):
        """Имитация детокенизации."""
        if isinstance(tokens, list):
            return ''.join([chr((t - 1000)) for t in tokens if t >= 1000])
        return "OBFUSCATED"
    
    def __len__(self):
        return 100
    
    def get_vocab(self):
        return self._vocab


class MockModel:
    """Мок для модели."""
    
    def __init__(self):
        self.get_input_embeddings = MagicMock(return_value=MagicMock(weight=MagicMock()))
        self.eval = MagicMock(return_value=self)
        self.to = MagicMock(return_value=self)
        self.clone = MagicMock(return_value=self)


def check_openai_api_key() -> bool:
    """Проверяет наличие API ключа OpenAI."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Не найден API ключ OpenAI. Пожалуйста, установите переменную окружения OPENAI_API_KEY.")
        return False
    return True


def query_openai(
    prompt: str, 
    model: str = "gpt-3.5-turbo", 
    temperature: float = 0.7, 
    max_tokens: int = 1000
) -> Optional[str]:
    """
    Отправляет запрос к OpenAI API и возвращает ответ.
    
    Args:
        prompt: Текст запроса
        model: Модель для использования
        temperature: Температура генерации
        max_tokens: Максимальное количество токенов в ответе
        
    Returns:
        str: Ответ от OpenAI API или None в случае ошибки
    """
    if not OPENAI_AVAILABLE:
        print("Библиотека OpenAI не установлена. Невозможно отправить запрос.")
        return None
    
    if not check_openai_api_key():
        return None
    
    try:
        client = OpenAI()
        
        # Создаем запрос
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Получаем ответ
        response = completion.choices[0].message.content
        return response
    
    except Exception as e:
        print(f"Ошибка при отправке запроса к OpenAI API: {e}")
        return None


def get_test_examples() -> List[Dict[str, str]]:
    """Возвращает список тестовых примеров с кодом на разных языках."""
    return [
        {
            "name": "Python Bubble Sort",
            "language": "python",
            "code": '''
def bubble_sort(arr):
    """
    Сортирует массив методом пузырька.
    
    Args:
        arr: Массив для сортировки
        
    Returns:
        Отсортированный массив
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Пример использования
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = bubble_sort(numbers)
print(f"Отсортированный массив: {sorted_numbers}")
''',
            "question": "Объясни, как работает этот алгоритм сортировки и какова его временная сложность?"
        },
        {
            "name": "JavaScript Fibonacci",
            "language": "javascript",
            "code": '''
/**
 * Вычисляет n-ое число Фибоначчи рекурсивно.
 * @param {number} n - Индекс числа Фибоначчи.
 * @return {number} - n-ое число Фибоначчи.
 */
function fibonacci(n) {
    // Базовые случаи
    if (n <= 0) return 0;
    if (n === 1) return 1;
    
    // Рекурсивный случай
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Оптимизированная версия с мемоизацией
function fibonacciMemo(n, memo = {}) {
    if (n in memo) return memo[n];
    if (n <= 0) return 0;
    if (n === 1) return 1;
    
    memo[n] = fibonacciMemo(n - 1, memo) + fibonacciMemo(n - 2, memo);
    return memo[n];
}

// Пример использования
console.log("Fibonacci(10):", fibonacci(10));
console.log("FibonacciMemo(10):", fibonacciMemo(10));
''',
            "question": "Сравни две реализации функции Фибоначчи. Какая из них более эффективна и почему?"
        },
        {
            "name": "Java OOP Example",
            "language": "java",
            "code": '''
public class BankAccount {
    private String accountNumber;
    private String ownerName;
    private double balance;
    
    public BankAccount(String accountNumber, String ownerName, double initialBalance) {
        this.accountNumber = accountNumber;
        this.ownerName = ownerName;
        this.balance = initialBalance;
    }
    
    public void deposit(double amount) {
        if (amount <= 0) {
            throw new IllegalArgumentException("Deposit amount must be positive");
        }
        balance += amount;
        System.out.println(String.format("Deposited %.2f. New balance: %.2f", amount, balance));
    }
    
    public void withdraw(double amount) {
        if (amount <= 0) {
            throw new IllegalArgumentException("Withdrawal amount must be positive");
        }
        if (amount > balance) {
            throw new IllegalArgumentException("Insufficient funds");
        }
        balance -= amount;
        System.out.println(String.format("Withdrew %.2f. New balance: %.2f", amount, balance));
    }
    
    public double getBalance() {
        return balance;
    }
    
    public String getAccountSummary() {
        return String.format("Account: %s, Owner: %s, Balance: %.2f", 
                            accountNumber, ownerName, balance);
    }
    
    public static void main(String[] args) {
        BankAccount account = new BankAccount("123456789", "John Doe", 1000.0);
        System.out.println(account.getAccountSummary());
        account.deposit(500.0);
        account.withdraw(200.0);
        System.out.println(account.getAccountSummary());
    }
}
''',
            "question": "Каковы основные принципы ООП, представленные в этом коде? Добавь проверку на отрицательный баланс при инициализации."
        }
    ]


def create_prompt_with_code(code_example: Dict[str, str]) -> str:
    """
    Создает промпт с кодом для отправки в LLM.
    
    Args:
        code_example: Словарь с информацией о примере кода
        
    Returns:
        str: Сформированный промпт
    """
    return f"""
Вот код на языке {code_example['language']}:

```{code_example['language']}
{code_example['code']}
```

{code_example['question']}
"""


def analyze_responses(original_response: str, obfuscated_response: str) -> Dict[str, Any]:
    """
    Анализирует и сравнивает ответы на оригинальный и обфусцированный промпты.
    
    Args:
        original_response: Ответ на оригинальный промпт
        obfuscated_response: Ответ на обфусцированный промпт
        
    Returns:
        Dict[str, Any]: Результаты анализа
    """
    # Очищаем текст от лишних пробелов и переносов строк для корректного сравнения
    def normalize_text(text):
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        return text
    
    original_normalized = normalize_text(original_response)
    obfuscated_normalized = normalize_text(obfuscated_response)
    
    # Считаем длину ответов
    original_length = len(original_response)
    obfuscated_length = len(obfuscated_response)
    length_diff = abs(original_length - obfuscated_length)
    length_diff_percent = length_diff / max(original_length, obfuscated_length) * 100
    
    # Вычисляем различие с помощью difflib
    similarity = difflib.SequenceMatcher(None, original_normalized, obfuscated_normalized).ratio() * 100
    
    # Выявляем ключевые фразы и термины в обоих ответах
    def extract_key_phrases(text, min_length=4):
        # Простой алгоритм для выявления возможных ключевых фраз
        words = re.findall(r'\b\w+\b', text.lower())
        phrases = [word for word in words if len(word) >= min_length and not word.isdigit()]
        return set(phrases)
    
    original_phrases = extract_key_phrases(original_response)
    obfuscated_phrases = extract_key_phrases(obfuscated_response)
    
    shared_phrases = original_phrases.intersection(obfuscated_phrases)
    unique_original = original_phrases - obfuscated_phrases
    unique_obfuscated = obfuscated_phrases - original_phrases
    
    # Формируем результат анализа
    analysis = {
        "similarity_percent": similarity,
        "length_original": original_length,
        "length_obfuscated": obfuscated_length,
        "length_diff": length_diff,
        "length_diff_percent": length_diff_percent,
        "shared_key_phrases_count": len(shared_phrases),
        "unique_phrases_original_count": len(unique_original),
        "unique_phrases_obfuscated_count": len(unique_obfuscated),
        "examples_unique_original": list(unique_original)[:5],
        "examples_unique_obfuscated": list(unique_obfuscated)[:5],
        "is_similar": similarity > 80  # Считаем ответы похожими, если сходство > 80%
    }
    
    return analysis


def display_results(
    example_name: str,
    original_prompt: str,
    obfuscated_prompt: str,
    original_response: str,
    obfuscated_response: str,
    analysis: Dict[str, Any]
) -> None:
    """Отображает результаты сравнения ответов."""
    print("=" * 80)
    print(f"ТЕСТ: {example_name}")
    print("=" * 80)
    
    print(f"\nСТАТИСТИКА ОТВЕТОВ:")
    print(f"- Сходство ответов: {analysis['similarity_percent']:.2f}%")
    print(f"- Длина оригинального ответа: {analysis['length_original']} символов")
    print(f"- Длина ответа на обфусцированный код: {analysis['length_obfuscated']} символов")
    print(f"- Разница в длине: {analysis['length_diff']} символов ({analysis['length_diff_percent']:.2f}%)")
    print(f"- Общих ключевых фраз: {analysis['shared_key_phrases_count']}")
    print(f"- Уникальных фраз в оригинальном ответе: {analysis['unique_phrases_original_count']}")
    print(f"- Уникальных фраз в ответе на обфусцированный код: {analysis['unique_phrases_obfuscated_count']}")
    
    print(f"\nПРИМЕРЫ УНИКАЛЬНЫХ ФРАЗ В ОРИГИНАЛЬНОМ ОТВЕТЕ:")
    for phrase in analysis["examples_unique_original"]:
        print(f"  - {phrase}")
    
    print(f"\nПРИМЕРЫ УНИКАЛЬНЫХ ФРАЗ В ОТВЕТЕ НА ОБФУСЦИРОВАННЫЙ КОД:")
    for phrase in analysis["examples_unique_obfuscated"]:
        print(f"  - {phrase}")
    
    print(f"\nВЫВОД:")
    if analysis["is_similar"]:
        print("✅ Ответы ПОХОЖИ. Обфускация успешно сохраняет способность LLM анализировать код.")
    else:
        print("❌ Ответы РАЗЛИЧАЮТСЯ. Обфускация может влиять на качество анализа кода LLM-моделью.")
    
    # Опциональное отображение промптов и ответов
    print("\nДля подробного анализа используйте опцию --verbose")


def main():
    parser = argparse.ArgumentParser(description="Тестирование ответов LLM на обфусцированные промпты с кодом")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Модель OpenAI для использования")
    parser.add_argument("--verbose", action="store_true", help="Показывать подробную информацию")
    parser.add_argument("--examples", type=int, default=0, help="Номер примера для тестирования (0 - все примеры)")
    args = parser.parse_args()
    
    # Патчим зависимости для CodeCipherObfuscator
    with patch('llm_guard.input_scanners.code_cipher.AutoTokenizer') as mock_tokenizer_cls, \
         patch('llm_guard.input_scanners.code_cipher.AutoModelForCausalLM') as mock_model_cls:
         
        # Настраиваем моки
        tokenizer = MockTokenizer()
        model = MockModel()
        mock_tokenizer_cls.from_pretrained = MagicMock(return_value=tokenizer)
        mock_model_cls.from_pretrained = MagicMock(return_value=model)
        
        # Инициализируем сканер
        scanner = CodeCipherObfuscator(
            model_name="mock_model",
            max_training_iterations=1
        )
        
        # Получаем тестовые примеры
        examples = get_test_examples()
        
        # Если указан конкретный пример, тестируем только его
        if args.examples > 0 and args.examples <= len(examples):
            examples = [examples[args.examples - 1]]
        
        # Тестируем каждый пример
        for example in examples:
            # Создаем промпт с кодом
            original_prompt = create_prompt_with_code(example)
            
            # Обфусцируем промпт
            obfuscated_prompt, metadata = scanner.scan(original_prompt)
            
            # Отправляем запросы к LLM
            print(f"Отправляю запрос для примера '{example['name']}' с оригинальным промптом...")
            original_response = query_openai(original_prompt, model=args.model)
            
            print(f"Отправляю запрос для примера '{example['name']}' с обфусцированным промптом...")
            obfuscated_response = query_openai(obfuscated_prompt, model=args.model)
            
            # Добавляем проверку на наличие ответов
            if not original_response or not obfuscated_response:
                print(f"Ошибка: Не удалось получить ответы от OpenAI API для примера '{example['name']}'.")
                continue
            
            # Анализируем ответы
            analysis = analyze_responses(original_response, obfuscated_response)
            
            # Отображаем результаты
            display_results(
                example["name"],
                original_prompt,
                obfuscated_prompt,
                original_response,
                obfuscated_response,
                analysis
            )
            
            # Подробная информация
            if args.verbose:
                print("\nОРИГИНАЛЬНЫЙ ПРОМПТ:")
                print("-" * 50)
                print(original_prompt)
                
                print("\nОБФУСЦИРОВАННЫЙ ПРОМПТ:")
                print("-" * 50)
                print(obfuscated_prompt)
                
                print("\nОТВЕТ НА ОРИГИНАЛЬНЫЙ ПРОМПТ:")
                print("-" * 50)
                print(original_response)
                
                print("\nОТВЕТ НА ОБФУСЦИРОВАННЫЙ ПРОМПТ:")
                print("-" * 50)
                print(obfuscated_response)
            
            # Пауза между запросами, чтобы не превысить лимиты API
            if example != examples[-1]:
                print("\nПауза перед следующим запросом...")
                time.sleep(2)


if __name__ == "__main__":
    main() 