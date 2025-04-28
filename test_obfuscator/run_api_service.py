#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для запуска API-сервиса LLM Guard с поддержкой CodeCipherObfuscator.

Этот скрипт инициализирует и запускает API-сервис LLM Guard с включенным
сканером CodeCipherObfuscator для обфускации кода в запросах к LLM.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import yaml
import uvicorn

# Добавляем корневую директорию проекта в sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Директории
TEST_DIR = Path(__file__).parent
CONFIG_DIR = TEST_DIR / "config"
VAULT_DIR = TEST_DIR / "vault"
VAULT_DIR.mkdir(exist_ok=True)


def create_config(model_name, port, host):
    """
    Создает конфигурационный файл для API-сервиса.
    
    Args:
        model_name: Имя модели для CodeCipherObfuscator
        port: Порт для API-сервиса
        host: Хост для API-сервиса
    
    Returns:
        str: Путь к созданному конфигурационному файлу
    """
    # Создаем директорию для конфигурации, если она не существует
    config_path = TEST_DIR / "config"
    config_path.mkdir(exist_ok=True)
    
    # Создаем конфигурацию на базе существующей
    if (CONFIG_DIR / "scanners.yml").exists():
        with open(CONFIG_DIR / "scanners.yml", "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "app": {
                "name": "LLM Guard API with CodeCipherObfuscator",
                "log_level": "DEBUG",
                "log_json": False,
                "scan_fail_fast": False,
                "scan_prompt_timeout": 30,
                "scan_output_timeout": 30,
                "lazy_load": False
            },
            "rate_limit": {
                "enabled": False,
                "limit": "100/minute"
            },
            "input_scanners": [],
            "output_scanners": []
        }
    
    # Обновляем конфигурацию для CodeCipherObfuscator
    code_cipher_config = {
        "type": "CodeCipherObfuscator",
        "params": {
            "model_name": model_name,
            "max_training_iterations": 50,
            "learning_rate": 0.01,
            "perplexity_threshold": 50.0,
            "early_stopping_patience": 5,
            "vault_dir": str(VAULT_DIR),
            "skip_patterns": ["# COPYRIGHT", "# DO NOT OBFUSCATE"]
        }
    }
    
    # Добавляем или обновляем конфигурацию CodeCipherObfuscator
    code_cipher_exists = False
    for i, scanner in enumerate(config["input_scanners"]):
        if scanner["type"] == "CodeCipherObfuscator":
            config["input_scanners"][i] = code_cipher_config
            code_cipher_exists = True
            break
    
    if not code_cipher_exists:
        config["input_scanners"].append(code_cipher_config)
    
    # Сохраняем конфигурацию
    config_file = config_path / "scanners.yml"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(config_file)


def run_api(config_file, port, host):
    """
    Запускает API-сервис с указанной конфигурацией.
    
    Args:
        config_file: Путь к конфигурационному файлу
        port: Порт для API-сервиса
        host: Хост для API-сервиса
    """
    # Устанавливаем переменные окружения
    os.environ["CONFIG_PATH"] = config_file
    
    # Создаем простой FastAPI сервис напрямую здесь
    from fastapi import FastAPI, Request
    from pydantic import BaseModel
    
    # Импортируем CodeCipherObfuscator
    from llm_guard.input_scanners import CodeCipherObfuscator
    
    # Инициализируем обфускатор
    obfuscator = CodeCipherObfuscator(
        model_name="gpt2",
        max_training_iterations=30,
        learning_rate=0.03,
        perplexity_threshold=100.0,
        early_stopping_patience=3,
        vault_dir=str(VAULT_DIR),
        skip_patterns=["# COPYRIGHT", "# DO NOT OBFUSCATE"]
    )
    
    # Создаем простое API
    app = FastAPI(title="LLM Guard API with CodeCipherObfuscator")
    
    class PromptRequest(BaseModel):
        prompt: str
    
    @app.post("/v1/scan_prompt")
    async def scan_prompt(request: PromptRequest):
        # Обфускация кода с помощью CodeCipherObfuscator
        try:
            obfuscated_prompt, metadata = obfuscator.scan(request.prompt)
            return {"prompt": obfuscated_prompt, "valid": True, "metadata": metadata}
        except Exception as e:
            return {"prompt": request.prompt, "valid": False, "error": str(e)}
    
    @app.post("/v1/deobfuscate")
    async def deobfuscate_prompt(request: PromptRequest):
        # Деобфускация кода с помощью CodeCipherObfuscator
        try:
            deobfuscated_prompt = obfuscator.deobfuscate(request.prompt)
            return {"prompt": deobfuscated_prompt, "valid": True}
        except Exception as e:
            return {"prompt": request.prompt, "valid": False, "error": str(e)}
    
    # Запускаем API-сервис
    uvicorn.run(app, host=host, port=port)


def main():
    """Основная функция для запуска API-сервиса."""
    parser = argparse.ArgumentParser(description="Запуск API-сервиса LLM Guard с CodeCipherObfuscator")
    parser.add_argument("--model", type=str, default="gpt2", help="Имя модели для CodeCipherObfuscator")
    parser.add_argument("--port", type=int, default=8000, help="Порт для API-сервиса")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Хост для API-сервиса")
    
    args = parser.parse_args()
    
    print(f"Запускаем API-сервис LLM Guard с CodeCipherObfuscator (модель: {args.model})")
    print(f"Хост: {args.host}, Порт: {args.port}")
    
    # Создаем конфигурацию
    config_file = create_config(args.model, args.port, args.host)
    print(f"Конфигурационный файл создан: {config_file}")
    
    # Запускаем API-сервис
    run_api(config_file, args.port, args.host)


if __name__ == "__main__":
    main() 