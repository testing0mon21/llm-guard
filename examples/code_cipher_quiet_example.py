#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Упрощенный пример использования CodeCipherObfuscator в тихом режиме.

Этот пример демонстрирует, как можно использовать CodeCipherObfuscator
с локальными моками, без загрузки реальной модели.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Добавляем родительскую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from llm_guard.input_scanners import CodeCipherObfuscator


class MockTokenizer:
    """Мок для токенизатора."""
    
    def __init__(self):
        # Создаем словарь токенов для теста
        self._vocab = {f"token_{i}": i for i in range(100)}
        self.all_special_ids = [0, 1, 2]
        
    def encode(self, text, return_tensors=None):
        """Имитация токенизации."""
        # Простое преобразование: каждый символ становится числом + 1000
        result = [ord(c) + 1000 for c in text[:30]]  # Берем только первые 30 символов
        if return_tensors == "pt":
            # Имитируем тензор
            return MagicMock(tolist=lambda: [result])
        return result
    
    def decode(self, tokens):
        """Имитация детокенизации."""
        # Для наглядности возвращаем зашифрованный текст
        if isinstance(tokens, list):
            # Заменяем буквы на их 'обфусцированные' версии
            mapping = {
                'd': 'x', 'e': 'z', 'f': 'p', 'u': 'o', 'm': 'n', 'a': '@', 'l': '1', 
                's': '$', 'v': 'w', 'r': '4', 't': '7', 'n': 'm', 'i': '!', 'o': '0'
            }
            
            # Декодируем обратно в символы
            text = ''.join([chr((t - 1000)) for t in tokens if t >= 1000])
            
            # Применяем обфускацию
            obfuscated = ''
            for char in text:
                if char.lower() in mapping:
                    # Сохраняем регистр
                    if char.isupper():
                        obfuscated += mapping[char.lower()].upper()
                    else:
                        obfuscated += mapping[char.lower()]
                else:
                    obfuscated += char
            
            return obfuscated
        
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
    """Демонстрация использования CodeCipherObfuscator с моком."""
    # Создаем временную директорию для хранилища
    vault_dir = Path("./cipher_vault_quiet")
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
            max_training_iterations=1,  # Минимальное значение для скорости
            vault_dir=str(vault_dir)
        )
        
        # Ручное переопределение маппинга токенов для наглядности
        # Это не нужно в реальном использовании, только для демонстрации
        simple_mapping = {}
        for i in range(1000, 1500):
            simple_mapping[i] = i + 500  # Смещаем все токены на 500
        scanner.token_mapping = simple_mapping

        # Пример простого кода
        simple_code = """
def sum_values(a, b):
    # Сложение двух значений
    return a + b
        """
        
        # Прямая обфускация кода
        print("Оригинальный код:")
        print("-" * 50)
        print(simple_code)
        
        # Ручная демонстрация обфускации
        print("\nДемонстрация обфускации:")
        print("-" * 50)
        
        # Кодируем и показываем токены
        tokens = tokenizer.encode(simple_code)
        print(f"Оригинальные токены: {tokens[:10]}...")
        
        # Применяем маппинг
        obfuscated_tokens = [scanner.token_mapping.get(token, token) for token in tokens]
        print(f"Обфусцированные токены: {obfuscated_tokens[:10]}...")
        
        # Декодируем обратно
        decoded = tokenizer.decode(obfuscated_tokens)
        print(f"\nРезультат обфускации: {decoded}")
        
        obfuscated_code = scanner._obfuscate_code(simple_code)
        print("\nОбфусцированный код (через метод _obfuscate_code):")
        print("-" * 50)
        print(obfuscated_code)
        
        # Пример промпта с кодом
        prompt = f"""
Привет! Можешь проанализировать функцию?

```python
{simple_code}
```

Спасибо!
        """
        
        # Обфусцируем промпт
        obfuscated_prompt, metadata = scanner.scan(prompt)
        
        print("\nОбфусцированный промпт (через scan):")
        print("-" * 50)
        print(obfuscated_prompt)
        
        # Выводим метаданные
        print("\nМетаданные сканирования:")
        print(f"Блоков кода найдено: {metadata['stats']['code_blocks_found']}")
        print(f"Блоков кода обфусцировано: {metadata['stats']['code_blocks_obfuscated']}")
        
        # Деобфусцируем обратно
        deobfuscated_prompt = scanner.deobfuscate(obfuscated_prompt)
        
        print("\nДеобфусцированный промпт:")
        print("-" * 50)
        print(deobfuscated_prompt)


if __name__ == "__main__":
    main() 