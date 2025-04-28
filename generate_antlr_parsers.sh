#!/bin/bash

# Скрипт для генерации парсеров ANTLR для всех поддерживаемых языков

# Создаем директории для каждого языка
mkdir -p llm_guard/input_scanners/antlr_grammars/python
mkdir -p llm_guard/input_scanners/antlr_grammars/javascript
mkdir -p llm_guard/input_scanners/antlr_grammars/java
mkdir -p llm_guard/input_scanners/antlr_grammars/csharp
mkdir -p llm_guard/input_scanners/antlr_grammars/kotlin
mkdir -p llm_guard/input_scanners/antlr_grammars/php
mkdir -p llm_guard/input_scanners/antlr_grammars/ruby
mkdir -p llm_guard/input_scanners/antlr_grammars/swift
mkdir -p llm_guard/input_scanners/antlr_grammars/golang
mkdir -p llm_guard/input_scanners/antlr_grammars/rust

# Путь к JAR файлу ANTLR
ANTLR_JAR="antlr-4.9.3-complete.jar"

echo "Генерация парсеров Python..."
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/python/ -lib llm_guard/input_scanners/antlr_grammars/python/ llm_guard/input_scanners/antlr_grammars/python/Python3Lexer.g4
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/python/ -lib llm_guard/input_scanners/antlr_grammars/python/ llm_guard/input_scanners/antlr_grammars/python/Python3Parser.g4

echo "Генерация парсеров JavaScript..."
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/javascript/ -lib llm_guard/input_scanners/antlr_grammars/javascript/ llm_guard/input_scanners/antlr_grammars/javascript/JavaScriptLexer.g4
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/javascript/ -lib llm_guard/input_scanners/antlr_grammars/javascript/ llm_guard/input_scanners/antlr_grammars/javascript/JavaScriptParser.g4

echo "Генерация парсеров Java..."
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/java/ -lib llm_guard/input_scanners/antlr_grammars/java/ llm_guard/input_scanners/antlr_grammars/java/JavaLexer.g4
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/java/ -lib llm_guard/input_scanners/antlr_grammars/java/ llm_guard/input_scanners/antlr_grammars/java/JavaParser.g4

echo "Генерация парсеров C#..."
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/csharp/ -lib llm_guard/input_scanners/antlr_grammars/csharp/ llm_guard/input_scanners/antlr_grammars/csharp/CSharpLexer.g4
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/csharp/ -lib llm_guard/input_scanners/antlr_grammars/csharp/ llm_guard/input_scanners/antlr_grammars/csharp/CSharpParser.g4

echo "Генерация парсеров Kotlin..."
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/kotlin/ -lib llm_guard/input_scanners/antlr_grammars/kotlin/ llm_guard/input_scanners/antlr_grammars/kotlin/KotlinLexer.g4
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/kotlin/ -lib llm_guard/input_scanners/antlr_grammars/kotlin/ llm_guard/input_scanners/antlr_grammars/kotlin/KotlinParser.g4

echo "Генерация парсеров PHP..."
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/php/ -lib llm_guard/input_scanners/antlr_grammars/php/ llm_guard/input_scanners/antlr_grammars/php/PhpLexer.g4
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/php/ -lib llm_guard/input_scanners/antlr_grammars/php/ llm_guard/input_scanners/antlr_grammars/php/PhpParser.g4

echo "Генерация парсеров Ruby..."
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/ruby/ -lib llm_guard/input_scanners/antlr_grammars/ruby/ llm_guard/input_scanners/antlr_grammars/ruby/Corundum.g4

echo "Генерация парсеров Swift..."
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/swift/ -lib llm_guard/input_scanners/antlr_grammars/swift/ llm_guard/input_scanners/antlr_grammars/swift/Swift5Lexer.g4
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/swift/ -lib llm_guard/input_scanners/antlr_grammars/swift/ llm_guard/input_scanners/antlr_grammars/swift/Swift5Parser.g4

echo "Генерация парсеров Go..."
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/golang/ -lib llm_guard/input_scanners/antlr_grammars/golang/ llm_guard/input_scanners/antlr_grammars/golang/GoLexer.g4
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/golang/ -lib llm_guard/input_scanners/antlr_grammars/golang/ llm_guard/input_scanners/antlr_grammars/golang/GoParser.g4

echo "Генерация парсеров Rust..."
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/rust/ -lib llm_guard/input_scanners/antlr_grammars/rust/ llm_guard/input_scanners/antlr_grammars/rust/RustLexer.g4
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -o llm_guard/input_scanners/antlr_grammars/rust/ -lib llm_guard/input_scanners/antlr_grammars/rust/ llm_guard/input_scanners/antlr_grammars/rust/RustParser.g4

echo "Готово! Все парсеры сгенерированы." 