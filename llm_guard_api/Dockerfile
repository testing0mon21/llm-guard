# Use the Python 3.12 slim image
FROM python:3.12-slim

LABEL org.opencontainers.image.source=https://github.com/protectai/llm-guard
LABEL org.opencontainers.image.description="LLM Guard API"
LABEL org.opencontainers.image.licenses=MIT

# Install system packages needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    openjdk-17-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Создание пользователя с UID 1000 (стандартный для Docker)
RUN useradd -m -u 1000 user

# Setting up environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    HOME=/home/user \
    VIRTUAL_ENV=/opt/venv

# Рабочая директория в контейнере
WORKDIR /app
ENV PYTHONPATH=/app:${PYTHONPATH}

# Копирование необходимых файлов
COPY --chown=user:user pyproject.toml ./
COPY --chown=user:user app ./app
COPY --chown=user:user generate_antlr_grammars.sh ./

# Установка зависимостей и базовых моделей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[cpu]" && \
    python -m spacy download en_core_web_sm && \
    chmod +x ./generate_antlr_grammars.sh && \
    ./generate_antlr_grammars.sh && \
    pip install --no-cache-dir antlr4-python3-runtime==4.9.3

# Switching to a non-root user
USER user

# Preloading language models
RUN python -c "from llm_guard.input_scanners.sensitive_information import preload_language; preload_language('en')"

# Запуск приложения
ENV PORT=8000
EXPOSE ${PORT}

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
