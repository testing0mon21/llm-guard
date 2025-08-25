#!/bin/bash

# Скрипт для запуска системы временного шифрования и обфускации промптов

echo "🚀 Запуск системы временного шифрования LLM Guard"
echo "=================================================="

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.9+ для работы системы."
    exit 1
fi

# Проверяем виртуальное окружение
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Рекомендуется использовать виртуальное окружение"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
fi

# Устанавливаем переменные окружения по умолчанию если не заданы
export TEMPORAL_MASTER_KEY=${TEMPORAL_MASTER_KEY:-"$(openssl rand -base64 32)"}
export TEMPORAL_TIME_WINDOW_MINUTES=${TEMPORAL_TIME_WINDOW_MINUTES:-30}
export CONFIG_FILE=${CONFIG_FILE:-"./config/scanners.yml"}

echo "🔧 Конфигурация:"
echo "   TEMPORAL_MASTER_KEY: ${TEMPORAL_MASTER_KEY:0:20}..."
echo "   TIME_WINDOW: $TEMPORAL_TIME_WINDOW_MINUTES минут"
echo "   OPENAI_API_KEY: ${OPENAI_API_KEY:+установлен}"
echo "   ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:+установлен}"
echo ""

# Проверяем зависимости
echo "📦 Проверка зависимостей..."
python3 -c "import cryptography, aiohttp, fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Некоторые зависимости отсутствуют. Устанавливаем..."
    pip install cryptography aiohttp fastapi uvicorn
fi

# Запускаем тесты
echo "🧪 Запуск базовых тестов..."
python3 test_temporal_encryption.py
if [ $? -ne 0 ]; then
    echo "❌ Тесты не прошли. Проверьте конфигурацию."
    exit 1
fi

echo ""
echo "✅ Тесты пройдены. Запуск API сервера..."

# Запускаем сервер
echo "🌐 Сервер будет доступен на http://localhost:8000"
echo "📚 Документация API: http://localhost:8000/docs"
echo "🔧 Метрики: http://localhost:8000/metrics"
echo ""
echo "Для остановки используйте Ctrl+C"
echo ""

# Запуск с автоперезагрузкой в режиме разработки
uvicorn app.app:create_app \
    --factory \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info