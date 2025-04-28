#!/bin/bash
set -e

# Версия ANTLR
ANTLR_VERSION="4.9.3"
ANTLR_JAR_URL="https://www.antlr.org/download/antlr-${ANTLR_VERSION}-complete.jar"
ANTLR_JAR="antlr-${ANTLR_VERSION}-complete.jar"

# Директории
WORK_DIR="/tmp/antlr"
GRAMMARS_REPO="https://github.com/antlr/grammars-v4.git"
TARGET_DIR="llm_guard/input_scanners/antlr_grammars"
GRAMMARS_DIR="${WORK_DIR}/grammars-v4"

# Список языков и их грамматик
declare -A GRAMMAR_FILES
GRAMMAR_FILES["python"]="python/python3/Python3Lexer.g4 python/python3/Python3Parser.g4"
GRAMMAR_FILES["javascript"]="javascript/javascript/JavaScriptLexer.g4 javascript/javascript/JavaScriptParser.g4"
GRAMMAR_FILES["java"]="java/java/JavaLexer.g4 java/java/JavaParser.g4"
GRAMMAR_FILES["csharp"]="csharp/CSharp/CSharpLexer.g4 csharp/CSharp/CSharpParser.g4"

echo "Starting ANTLR grammar generation..."

# Создание рабочих директорий
mkdir -p "${WORK_DIR}"
mkdir -p "${TARGET_DIR}"

# Скачивание ANTLR JAR
if [ ! -f "${WORK_DIR}/${ANTLR_JAR}" ]; then
    echo "Downloading ANTLR JAR..."
    curl -o "${WORK_DIR}/${ANTLR_JAR}" "${ANTLR_JAR_URL}"
fi

# Клонирование репозитория с грамматиками
if [ ! -d "${GRAMMARS_DIR}" ]; then
    echo "Cloning grammars repository..."
    git clone --depth 1 "${GRAMMARS_REPO}" "${GRAMMARS_DIR}"
fi

# Перемещение в рабочую директорию
cd "${WORK_DIR}"

# Генерация парсеров из грамматик
echo "Generating parsers..."
for language in "${!GRAMMAR_FILES[@]}"; do
    echo "Processing ${language} grammar..."
    
    # Создаем временную директорию для сборки грамматики
    LANG_BUILD_DIR="${WORK_DIR}/${language}_build"
    mkdir -p "${LANG_BUILD_DIR}"
    
    # Копируем все необходимые файлы грамматик во временную директорию
    for grammar_file in ${GRAMMAR_FILES[$language]}; do
        grammar_dir=$(dirname "${GRAMMARS_DIR}/${grammar_file}")
        # Копируем все файлы из директории грамматики
        cp -r "${grammar_dir}"/* "${LANG_BUILD_DIR}/"
    done
    
    # Переходим во временную директорию
    cd "${LANG_BUILD_DIR}"
    
    # Выполняем генерацию парсеров для всех .g4 файлов в директории
    for g4_file in *.g4; do
        echo "Generating parser for ${g4_file}..."
        java -jar "${WORK_DIR}/${ANTLR_JAR}" -Dlanguage=Python3 -visitor "${g4_file}"
    done
    
    # Копируем сгенерированные файлы в целевую директорию
    echo "Copying generated files to ${TARGET_DIR}..."
    mkdir -p "/${TARGET_DIR}/${language}"
    cp *.py "/${TARGET_DIR}/${language}/"
    
    # Возвращаемся в рабочую директорию
    cd "${WORK_DIR}"
done

# Создаем файл __init__.py в директории с грамматиками
echo "Creating __init__.py file..."
touch "/${TARGET_DIR}/__init__.py"

for language in "${!GRAMMAR_FILES[@]}"; do
    touch "/${TARGET_DIR}/${language}/__init__.py"
done

# Проверяем результат
echo "Checking generated files..."
find "/${TARGET_DIR}" -type f -name "*.py" | wc -l

echo "ANTLR grammar generation completed." 