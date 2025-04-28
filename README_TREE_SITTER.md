# Tree-sitter для обфускации кода в LLM Guard

## Обзор

Данный модуль расширяет возможности обфускации кода в LLM Guard с использованием Tree-sitter - эффективного парсера для анализа структуры кода. Tree-sitter предоставляет более точный синтаксический анализ кода по сравнению с регулярными выражениями или методами на основе ANTLR.

## Преимущества использования Tree-sitter

1. **Точный анализ исходного кода**: Tree-sitter строит полное синтаксическое дерево (AST), что позволяет точно определять идентификаторы, функции, классы и переменные.

2. **Поддержка многих языков программирования**: Включая Python, JavaScript, Java и другие.

3. **Высокая производительность**: Tree-sitter - быстрый и эффективный парсер.

4. **Интеллектуальная обфускация**: Лучше определяет, какие элементы можно безопасно обфусцировать, а какие должны остаться неизменными.

## Установка

Для использования Tree-sitter требуется:

1. Установить пакет tree-sitter:
   ```bash
   pip install tree-sitter
   ```

2. Скомпилировать грамматики для нужных языков программирования с помощью скрипта:
   ```bash
   python setup_tree_sitter.py
   ```

Скрипт скачает и скомпилирует грамматики для следующих языков:
- Python
- JavaScript
- Java
- C/C++
- Go
- Ruby
- Rust
- PHP
- TypeScript

## Использование

### Базовый вариант использования

```python
from llm_guard.input_scanners.tree_sitter_obfuscator import TreeSitterCodeCipherObfuscator

# Создаем обфускатор с использованием Tree-sitter
obfuscator = TreeSitterCodeCipherObfuscator(
    model_name="gpt2",
    training_iterations=1,
    learning_rate=0.001,
    use_tree_sitter=True  # Включаем использование Tree-sitter
)

# Обфусцируем код
code = """
def calculate_fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""

obfuscated_code, metadata = obfuscator.scan(code)
print(f"Обфусцированный код:\n{obfuscated_code}")
```

### Получение информации о коде с помощью коллекторов

Можно использовать коллекторы напрямую для анализа кода:

```python
from llm_guard.input_scanners.tree_sitter_collectors import get_collector_for_language

# Создаем коллектор для Python
collector = get_collector_for_language("python")

# Анализируем код
code = """
def hello(name):
    return f"Hello, {name}!"

result = hello("World")
print(result)
"""

collector.collect(code)

# Получаем информацию о коде
print(f"Функции: {collector.get_functions()}")
print(f"Переменные: {collector.get_variables()}")
print(f"Строковые литералы: {collector.get_string_literals()}")
print(f"Статистика: {collector.get_statistics()}")
```

## Поддерживаемые языки

На данный момент коллекторы реализованы для следующих языков:

- Python
- JavaScript
- Java

Вы можете легко добавить поддержку других языков, реализовав соответствующие коллекторы.

## Создание собственных коллекторов

Для создания коллектора для нового языка программирования:

1. Наследуйте класс `BaseTreeSitterCollector`
2. Реализуйте методы `_get_language_keywords` и `_walk_tree`
3. Добавьте коллектор в словарь `TREE_SITTER_COLLECTORS`

Пример:

```python
from llm_guard.input_scanners.tree_sitter_base import BaseTreeSitterCollector

class RubyTreeSitterCollector(BaseTreeSitterCollector):
    def __init__(self):
        super().__init__("ruby")
    
    def _get_language_keywords(self):
        return {"def", "class", "if", "else", "end", "module", "require", ...}
    
    def _walk_tree(self, node):
        # Реализация для Ruby
        pass

# Добавляем коллектор в словарь
TREE_SITTER_COLLECTORS["ruby"] = RubyTreeSitterCollector
```

## Тестирование

Для тестирования Tree-sitter коллекторов и обфускатора:

```bash
# Тестирование коллекторов
python test_tree_sitter_collectors.py

# Тестирование обфускации
python test_tree_sitter_obfuscator.py

# Расширенное тестирование с анализом различий
python test_tree_sitter_obfuscator_advanced.py
```

## Диагностика проблем

Если Tree-sitter недоступен или возникают проблемы с грамматиками:

1. Убедитесь, что пакет tree-sitter установлен:
   ```bash
   pip install tree-sitter
   ```

2. Перезапустите скрипт установки грамматик:
   ```bash
   python setup_tree_sitter.py
   ```

3. Проверьте, что грамматики успешно скомпилированы в директории `llm_guard/input_scanners/tree_sitter_grammars`.

4. В случае проблем с конкретным языком, попробуйте отладить его отдельно:
   ```python
   from tree_sitter import Parser, Language
   
   # Путь к библиотеке с грамматиками
   lib_path = 'llm_guard/input_scanners/tree_sitter_grammars/languages.so'
   
   # Загружаем грамматику для Python
   python_lang = Language(lib_path, 'python')
   
   # Создаем парсер
   parser = Parser()
   parser.set_language(python_lang)
   
   # Тестируем парсинг
   tree = parser.parse(b"print('Hello, world!')")
   print(tree.root_node.sexp())
   ``` 