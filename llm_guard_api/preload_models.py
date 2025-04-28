#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для предварительной загрузки моделей для сканеров LLM Guard.
Запускается во время сборки Docker-образа, чтобы ускорить работу API.
"""

import os
import logging
import yaml
import importlib
from typing import List, Dict, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Путь к конфигурационному файлу
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "scanners.yml")

def load_config() -> Dict[str, Any]:
    """Загрузка конфигурации сканеров из файла"""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
        return {}

def preload_input_scanners(config: Dict[str, Any]) -> None:
    """Предварительная загрузка моделей для input_scanners"""
    try:
        from llm_guard.input_scanners import util as input_util
        scanners = config.get("input_scanners", [])
        logger.info(f"Найдено {len(scanners)} input_scanners в конфигурации")
        
        for scanner_config in scanners:
            scanner_type = scanner_config.get("type")
            scanner_params = scanner_config.get("params", {})
            
            if scanner_type:
                logger.info(f"Загрузка моделей для сканера: {scanner_type}")
                try:
                    # Инициализируем сканер только для загрузки моделей
                    scanner = input_util.get_scanner_by_name(scanner_type, scanner_params)
                    logger.info(f"Модели для сканера {scanner_type} успешно загружены")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить модели для сканера {scanner_type}: {e}")
    except ImportError as e:
        logger.error(f"Ошибка импорта модуля llm_guard.input_scanners: {e}")

def preload_output_scanners(config: Dict[str, Any]) -> None:
    """Предварительная загрузка моделей для output_scanners"""
    try:
        from llm_guard.output_scanners import util as output_util
        scanners = config.get("output_scanners", [])
        logger.info(f"Найдено {len(scanners)} output_scanners в конфигурации")
        
        for scanner_config in scanners:
            scanner_type = scanner_config.get("type")
            scanner_params = scanner_config.get("params", {})
            
            if scanner_type:
                logger.info(f"Загрузка моделей для сканера: {scanner_type}")
                try:
                    # Инициализируем сканер только для загрузки моделей
                    scanner = output_util.get_scanner_by_name(scanner_type, scanner_params)
                    logger.info(f"Модели для сканера {scanner_type} успешно загружены")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить модели для сканера {scanner_type}: {e}")
    except ImportError as e:
        logger.error(f"Ошибка импорта модуля llm_guard.output_scanners: {e}")

def main():
    logger.info("Начинаем предварительную загрузку моделей для LLM Guard API")
    
    # Загружаем конфигурацию
    config = load_config()
    if not config:
        logger.error("Не удалось загрузить конфигурацию. Завершаем работу.")
        return
    
    # Загружаем модели для input_scanners
    preload_input_scanners(config)
    
    # Загружаем модели для output_scanners
    preload_output_scanners(config)
    
    logger.info("Предварительная загрузка моделей завершена")

if __name__ == "__main__":
    main() 