#!/usr/bin/env python3
"""
Пример использования CodeObfuscator для обфускации кода в промптах при сохранении
их читаемости для LLM, но затруднении для человека.

Также демонстрирует возможность сохранения оригинального кода для последующей деобфускации.
"""

from llm_guard.input_scanners import CodeObfuscator
from llm_guard.vault import Vault
import re

# Создаем пример промпта с кодом
prompt_with_code = """
Вот пример функции на Python для вычисления факториала:

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

# Пример использования
result = factorial(5)
print(f"Факториал числа 5 равен {result}")
```

А вот еще один пример кода на JavaScript:

```javascript
function calculateTotal(price, tax) {
    return price + (price * tax);
}

// Вычисляем итоговую сумму
const total = calculateTotal(100, 0.2);
console.log(`Итоговая сумма: ${total}`);
```

Напиши функцию для вычисления чисел Фибоначчи.
"""

# Инициализируем сканер с различными параметрами
obfuscator = CodeObfuscator(
    homoglyph_probability=0.4,         # Вероятность замены символов на гомоглифы
    invisible_char_probability=0.2,     # Вероятность добавления невидимых символов
    identifier_modification_probability=0.5,  # Вероятность модификации идентификаторов
    preserve_keywords=True             # Сохранять ключевые слова языка без изменений
)

# Обфусцируем код в промпте
obfuscated_prompt, _, _ = obfuscator.scan(prompt_with_code)

# Выводим результат
print("=== ОРИГИНАЛЬНЫЙ ПРОМПТ ===")
print(prompt_with_code)
print("\n=== ОБФУСЦИРОВАННЫЙ ПРОМПТ ===")
print(obfuscated_prompt)

# Пример с более агрессивной обфускацией
aggressive_obfuscator = CodeObfuscator(
    homoglyph_probability=0.8,
    invisible_char_probability=0.5,
    identifier_modification_probability=0.7,
    preserve_keywords=False  # Не сохраняем даже ключевые слова
)

aggressive_obfuscated_prompt, _, _ = aggressive_obfuscator.scan(prompt_with_code)

print("\n=== АГРЕССИВНО ОБФУСЦИРОВАННЫЙ ПРОМПТ ===")
print(aggressive_obfuscated_prompt)

print("\nЗаметка: В консоли некоторые символы могут выглядеть одинаково, но имеют разные Unicode-значения. Невидимые символы также не отображаются в консоли.")

# Пример с использованием Vault для сохранения и восстановления оригинального кода
print("\n\n=== ДЕМОНСТРАЦИЯ VAULT ДЛЯ ДЕОБФУСКАЦИИ ===")
vault = Vault()  # Создаем хранилище

# Создаем экземпляр обфускатора с включенным Vault
vault_obfuscator = CodeObfuscator(
    homoglyph_probability=0.6,
    invisible_char_probability=0.3,
    identifier_modification_probability=0.6,
    preserve_keywords=True,
    enable_vault=True,  # Включаем сохранение оригинального кода
    vault=vault         # Передаем объект хранилища
)

# Обфусцируем код и сохраняем его в Vault
vault_obfuscated_prompt, _, _ = vault_obfuscator.scan(prompt_with_code)

print("1. Обфусцированный промпт (с сохранением оригинала в Vault):")
print(vault_obfuscated_prompt)

# Проверяем, что сохранено в Vault
print("\n2. Содержимое Vault:")
for idx, (obf_code, orig_code, code_id) in enumerate(vault_obfuscator.get_vault().get()):
    print(f"Запись #{idx+1}, ID: {code_id}")
    print(f"  Оригинальный код: {orig_code[:50]}..." if len(orig_code) > 50 else f"  Оригинальный код: {orig_code}")

# Деобфусцируем промпт обратно
deobfuscated_prompt = vault_obfuscator.deobfuscate(vault_obfuscated_prompt)

print("\n3. Деобфусцированный промпт (восстановленный из Vault):")
print(deobfuscated_prompt)

# Проверяем, равен ли восстановленный промпт оригинальному
def normalize_text(text):
    """Нормализует текст для сравнения, удаляя лишние пробелы и переносы строк"""
    # Удаляем все пробелы и переносы строк
    return re.sub(r'\s+', '', text)

# Нормализуем оба текста перед сравнением, игнорируя разницу в пробелах
is_restored = normalize_text(prompt_with_code) == normalize_text(deobfuscated_prompt)
print(f"\n4. Успешно восстановлен: {'Да ✓' if is_restored else 'Нет ✗'}")

# Если есть различия, проверим их более детально
if not is_restored:
    # Сравним код при игнорировании пробелов и языковых меток
    pattern = r'```(?:\w*\n)?(.*?)```'
    orig_codes = re.findall(pattern, prompt_with_code, re.DOTALL)
    deob_codes = re.findall(pattern, deobfuscated_prompt, re.DOTALL)
    
    all_codes_restored = True
    for i, (orig, deob) in enumerate(zip(orig_codes, deob_codes)):
        orig_normalized = normalize_text(orig)
        deob_normalized = normalize_text(deob)
        is_code_same = orig_normalized == deob_normalized
        all_codes_restored = all_codes_restored and is_code_same
        print(f"   - Блок кода #{i+1}: {'Восстановлен ✓' if is_code_same else 'Различается ✗'}")
    
    print(f"   - Итог по блокам кода: {'Все восстановлены ✓' if all_codes_restored else 'Есть различия ✗'}")
    
    # Проверяем текст вне блоков кода
    def extract_non_code_parts(text):
        parts = []
        last_end = 0
        for match in re.finditer(pattern, text, re.DOTALL):
            start, end = match.span()
            if start > last_end:
                parts.append(text[last_end:start])
            last_end = end
        if last_end < len(text):
            parts.append(text[last_end:])
        return ''.join(parts)
    
    orig_text = normalize_text(extract_non_code_parts(prompt_with_code))
    deob_text = normalize_text(extract_non_code_parts(deobfuscated_prompt))
    is_text_same = orig_text == deob_text
    print(f"   - Текст вне блоков кода: {'Восстановлен ✓' if is_text_same else 'Различается ✗'}") 