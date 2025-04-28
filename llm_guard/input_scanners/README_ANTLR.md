# Инструкции по установке и использованию ANTLR для CodeCipherObfuscator

Модуль LLM Guard CodeCipherObfuscator может использовать ANTLR для более точного анализа и обфускации кода. В этом документе описаны шаги для настройки ANTLR и необходимых грамматик.

## Необходимые компоненты

1. Python библиотека ANTLR4:
   ```bash
   pip install antlr4-python3-runtime
   ```

2. JAR файл ANTLR для генерации парсеров:
   ```bash
   curl -O https://www.antlr.org/download/antlr-4.9.3-complete.jar
   ```

## Грамматики для поддерживаемых языков

CodeCipherObfuscator поддерживает обфускацию следующих языков программирования с использованием ANTLR:
- Python
- JavaScript
- Java
- C#
- Kotlin
- PHP
- Ruby
- Swift
- Go
- Rust

## Варианты установки грамматик

### Вариант 1: Использование готовых грамматик

Вы можете найти готовые грамматики и сгенерированные парсеры для различных языков в следующих репозиториях:

1. **Python**:
   - https://github.com/antlr/grammars-v4/tree/master/python/python3
   - Используйте готовые файлы Python3Lexer.py, Python3Parser.py и Python3Listener.py

2. **JavaScript**:
   - https://github.com/antlr/grammars-v4/tree/master/javascript/javascript
   - Используйте готовые файлы JavaScriptLexer.py, JavaScriptParser.py и JavaScriptListener.py

3. **Java**:
   - https://github.com/antlr/grammars-v4/tree/master/java/java9
   - Используйте готовые файлы JavaLexer.py, JavaParser.py и JavaListener.py

4. **C#**:
   - https://github.com/antlr/grammars-v4/tree/master/csharp
   - Используйте готовые файлы CSharpLexer.py, CSharpParser.py и CSharpListener.py

5. **Kotlin**:
   - https://github.com/antlr/grammars-v4/tree/master/kotlin/kotlin-formal
   - Используйте готовые файлы KotlinLexer.py, KotlinParser.py и KotlinListener.py

6. **PHP**:
   - https://github.com/antlr/grammars-v4/tree/master/php
   - Используйте готовые файлы PhpLexer.py, PhpParser.py и PhpListener.py

7. **Ruby**:
   - https://github.com/antlr/grammars-v4/tree/master/ruby
   - Используйте готовые файлы RubyLexer.py, RubyParser.py и RubyListener.py

8. **Swift**:
   - https://github.com/antlr/grammars-v4/tree/master/swift
   - Используйте готовые файлы SwiftLexer.py, SwiftParser.py и SwiftListener.py

9. **Go**:
   - https://github.com/antlr/grammars-v4/tree/master/golang
   - Используйте готовые файлы GoLexer.py, GoParser.py и GoListener.py

10. **Rust**:
    - https://github.com/antlr/grammars-v4/tree/master/rust
    - Используйте готовые файлы RustLexer.py, RustParser.py и RustListener.py

Скопируйте эти файлы в соответствующие директории:
```
llm_guard/input_scanners/antlr_grammars/python/
llm_guard/input_scanners/antlr_grammars/javascript/
llm_guard/input_scanners/antlr_grammars/java/
llm_guard/input_scanners/antlr_grammars/csharp/
llm_guard/input_scanners/antlr_grammars/kotlin/
llm_guard/input_scanners/antlr_grammars/php/
llm_guard/input_scanners/antlr_grammars/ruby/
llm_guard/input_scanners/antlr_grammars/swift/
llm_guard/input_scanners/antlr_grammars/golang/
llm_guard/input_scanners/antlr_grammars/rust/
```

### Вариант 2: Генерация из объединенных грамматик

Для каждого языка создайте объединенную грамматику и сгенерируйте парсер:

1. Создайте файл грамматики (например, Python3Combined.g4) объединяющий лексер и парсер
2. Сгенерируйте парсер с помощью команды:
   ```bash
   java -jar antlr-4.9.3-complete.jar -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/python/ Python3Combined.g4
   ```

## Проверка работоспособности

После установки грамматик вы можете проверить, что ANTLR правильно настроен, с помощью следующего кода:

```python
from antlr4 import InputStream, CommonTokenStream
from llm_guard.input_scanners.antlr_grammars.python.Python3Lexer import Python3Lexer
from llm_guard.input_scanners.antlr_grammars.python.Python3Parser import Python3Parser

# Проверка парсера Python
code = "def hello(): print('Hello, world!')"
input_stream = InputStream(code)
lexer = Python3Lexer(input_stream)
tokens = CommonTokenStream(lexer)
parser = Python3Parser(tokens)
tree = parser.file_input()
print("Парсер Python успешно работает!")
```

Аналогично для Kotlin:

```python
from antlr4 import InputStream, CommonTokenStream
from llm_guard.input_scanners.antlr_grammars.kotlin.KotlinLexer import KotlinLexer
from llm_guard.input_scanners.antlr_grammars.kotlin.KotlinParser import KotlinParser

# Проверка парсера Kotlin
code = "fun main() { println(\"Hello, world!\") }"
input_stream = InputStream(code)
lexer = KotlinLexer(input_stream)
tokens = CommonTokenStream(lexer)
parser = KotlinParser(tokens)
tree = parser.kotlinFile()
print("Парсер Kotlin успешно работает!")
```

## Устранение неполадок

Если при генерации парсеров возникают ошибки:
1. Убедитесь, что используете совместимую версию ANTLR Runtime и ANTLR JAR
2. При ошибках с токенами в разделенных грамматиках, используйте объединенные грамматики
3. Рассмотрите возможность использования готовых сгенерированных парсеров
4. Проверьте корректность путей и структуры директорий

## Дополнительные ресурсы

- [Официальный сайт ANTLR](https://www.antlr.org/)
- [Репозиторий грамматик ANTLR v4](https://github.com/antlr/grammars-v4)
- [Документация ANTLR для Python](https://github.com/antlr/antlr4/tree/master/runtime/Python3) 