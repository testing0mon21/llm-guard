#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Пример использования CodeCipherObfuscator для обфускации кода на разных языках программирования.

Этот пример демонстрирует, как сканер обрабатывает код на различных языках (Python, JavaScript, Java, C++)
и сохраняет структуру кода при обфускации.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Добавляем родительскую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from llm_guard.input_scanners import CodeCipherObfuscator


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


def main():
    """Демонстрация использования CodeCipherObfuscator с разными языками программирования."""
    # Создаем временную директорию для хранилища
    vault_dir = Path("./cipher_vault_multilang")
    os.makedirs(vault_dir, exist_ok=True)

    # Патчим зависимости
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
            max_training_iterations=1,
            vault_dir=str(vault_dir)
        )
        
        # Примеры кода на разных языках
        code_examples = {
            "python": '''
def calculate_fibonacci(n):
    """Вычисляет n-ое число Фибоначчи."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
''',
            "javascript": '''
function sortArray(arr) {
    // Сортировка массива методом пузырька
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr.length - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap elements
                let temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    return arr;
}
''',
            "java": '''
public class Person {
    private String name;
    private int age;
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    public String getName() {
        return name;
    }
    
    public int getAge() {
        return age;
    }
}
''',
            "cpp": '''
#include <iostream>
#include <vector>

std::vector<int> mergeArrays(const std::vector<int>& arr1, const std::vector<int>& arr2) {
    std::vector<int> result;
    size_t i = 0, j = 0;
    
    while (i < arr1.size() && j < arr2.size()) {
        if (arr1[i] < arr2[j]) {
            result.push_back(arr1[i++]);
        } else {
            result.push_back(arr2[j++]);
        }
    }
    
    while (i < arr1.size()) {
        result.push_back(arr1[i++]);
    }
    
    while (j < arr2.size()) {
        result.push_back(arr2[j++]);
    }
    
    return result;
}
'''
        }
        
        # Создаем промпт с кодом на разных языках
        prompt = """
Привет! У меня есть несколько примеров кода на разных языках программирования, которые я хотел бы проанализировать.

Вот пример на Python:

```python
{python_code}
```

Вот пример на JavaScript:

```javascript
{javascript_code}
```

Вот пример на Java:

```java
{java_code}
```

И наконец, пример на C++:

```cpp
{cpp_code}
```

Можешь объяснить, как каждый из этих примеров кода работает и какие есть особенности в реализации?
""".format(
            python_code=code_examples["python"],
            javascript_code=code_examples["javascript"],
            java_code=code_examples["java"],
            cpp_code=code_examples["cpp"]
        )
        
        # Обфусцируем промпт
        obfuscated_prompt, metadata = scanner.scan(prompt)
        
        # Выводим результаты
        print("=" * 50)
        print("РЕЗУЛЬТАТЫ ОБФУСКАЦИИ КОДА НА РАЗНЫХ ЯЗЫКАХ")
        print("=" * 50)
        
        print(f"\nНайдено блоков кода: {metadata['stats']['code_blocks_found']}")
        print(f"Обфусцировано блоков кода: {metadata['stats']['code_blocks_obfuscated']}")
        
        print("\nОРИГИНАЛЬНЫЙ ПРОМПТ:")
        print("-" * 50)
        print(prompt)
        
        print("\nОБФУСЦИРОВАННЫЙ ПРОМПТ:")
        print("-" * 50)
        print(obfuscated_prompt)
        
        # Деобфусцируем и проверяем
        deobfuscated_prompt = scanner.deobfuscate(obfuscated_prompt)
        
        print("\nДЕОБФУСЦИРОВАННЫЙ ПРОМПТ:")
        print("-" * 50)
        print(deobfuscated_prompt)
        
        # Создаем таблицу с информацией об идентификаторах
        print("\nАНАЛИЗ ОБФУСКАЦИИ ИДЕНТИФИКАТОРОВ:")
        print("-" * 50)
        
        # Извлекаем все блоки кода из обфусцированного промпта
        import re
        
        original_blocks = re.findall(r"```(\w+)\n(.*?)\n```", prompt, re.DOTALL)
        obfuscated_blocks = re.findall(r"```(\w+)\n(.*?)\n```", obfuscated_prompt, re.DOTALL)
        
        for i, ((orig_lang, orig_code), (obf_lang, obf_code)) in enumerate(zip(original_blocks, obfuscated_blocks)):
            print(f"\nЯзык: {orig_lang}")
            print(f"{'Оригинальный идентификатор':<30} | {'Обфусцированный идентификатор':<30}")
            print("-" * 62)
            
            # Найдем все идентификаторы в обоих блоках
            # Это упрощенный подход, для разных языков могут потребоваться разные регулярные выражения
            orig_identifiers = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", orig_code))
            obf_identifiers = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", obf_code))
            
            # Исключаем ключевые слова для соответствующего языка
            keywords = {
                "python": ["def", "if", "else", "elif", "return", "while", "for", "in", "not", "and", "or"],
                "javascript": ["function", "let", "const", "var", "for", "if", "else", "return"],
                "java": ["public", "private", "class", "void", "int", "String", "return", "this"],
                "cpp": ["include", "const", "size_t", "int", "void", "return", "while", "if", "else", "std"]
            }
            
            orig_identifiers = [ident for ident in orig_identifiers if ident not in keywords.get(orig_lang, [])]
            
            # Создаем примерное соответствие идентификаторов (упрощенно)
            # В реальном сценарии нужно было бы анализировать позиции в коде
            sorted_orig = sorted(orig_identifiers)
            sorted_obf = sorted([i for i in obf_identifiers if "_" in i or any(c.isdigit() for c in i)])
            
            # Выводим соответствия
            for j, orig_ident in enumerate(sorted_orig):
                if j < len(sorted_obf):
                    print(f"{orig_ident:<30} | {sorted_obf[j]:<30}")
                else:
                    print(f"{orig_ident:<30} | {'<не найден>':<30}")


if __name__ == "__main__":
    main() 