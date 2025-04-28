#!/usr/bin/env python3
"""
Скрипт для установки Tree-sitter и грамматик для разных языков программирования.
Этот скрипт:
1. Проверяет, установлен ли tree-sitter
2. Устанавливает tree-sitter, если не установлен
3. Скачивает и компилирует грамматики для языков программирования
"""

import os
import subprocess
import sys
import logging
from pathlib import Path
import shutil

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Языки программирования, для которых нужно установить грамматики
LANGUAGES = [
    {
        "name": "python", 
        "repo": "https://github.com/tree-sitter/tree-sitter-python"
    },
    {
        "name": "javascript", 
        "repo": "https://github.com/tree-sitter/tree-sitter-javascript"
    },
    {
        "name": "typescript", 
        "repo": "https://github.com/tree-sitter/tree-sitter-typescript"
    },
    {
        "name": "java", 
        "repo": "https://github.com/tree-sitter/tree-sitter-java"
    },
    {
        "name": "c", 
        "repo": "https://github.com/tree-sitter/tree-sitter-c"
    },
    {
        "name": "cpp", 
        "repo": "https://github.com/tree-sitter/tree-sitter-cpp"
    },
    {
        "name": "go", 
        "repo": "https://github.com/tree-sitter/tree-sitter-go"
    },
    {
        "name": "ruby", 
        "repo": "https://github.com/tree-sitter/tree-sitter-ruby"
    },
    {
        "name": "rust", 
        "repo": "https://github.com/tree-sitter/tree-sitter-rust"
    },
    {
        "name": "php", 
        "repo": "https://github.com/tree-sitter/tree-sitter-php"
    }
]

# Путь для сохранения грамматик
DEFAULT_PARSERS_PATH = os.path.join(os.path.expanduser("~"), ".tree-sitter", "parsers")


def check_pip_dependency(package_name):
    """Проверяет, установлен ли пакет через pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "show", package_name], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def install_pip_dependency(package_name):
    """Устанавливает пакет через pip."""
    try:
        logger.info(f"Устанавливаем {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"{package_name} успешно установлен!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при установке {package_name}: {e}")
        return False


def setup_tree_sitter():
    """Устанавливает tree-sitter и проверяет его работу."""
    # Проверяем, установлен ли tree-sitter
    if not check_pip_dependency("tree-sitter"):
        if not install_pip_dependency("tree-sitter"):
            logger.error("Не удалось установить tree-sitter. Выходим.")
            return False
    
    # Проверяем, можем ли мы импортировать tree-sitter
    try:
        import tree_sitter
        logger.info("Tree-sitter успешно импортирован!")
        return True
    except ImportError as e:
        logger.error(f"Не удалось импортировать tree-sitter: {e}")
        return False


def download_and_build_grammar(language, repo_url, parsers_path):
    """Скачивает и собирает грамматику для конкретного языка."""
    language_path = os.path.join(parsers_path, language)
    
    # Создаем директорию, если ее нет
    if not os.path.exists(language_path):
        os.makedirs(language_path, exist_ok=True)
    
    # Определяем имя репозитория из URL
    repo_name = repo_url.split("/")[-1]
    repo_path = os.path.join(language_path, repo_name)
    
    # Проверяем, существует ли директория репозитория
    if os.path.exists(repo_path):
        logger.info(f"Репозиторий {repo_name} уже существует. Обновляем...")
        # Обновляем репозиторий
        try:
            subprocess.check_call(["git", "pull"], cwd=repo_path)
        except subprocess.CalledProcessError:
            logger.warning(f"Не удалось обновить репозиторий {repo_name}. Удаляем и клонируем заново.")
            shutil.rmtree(repo_path)
            subprocess.check_call(["git", "clone", repo_url, repo_path])
    else:
        # Клонируем репозиторий
        logger.info(f"Клонируем репозиторий {repo_name}...")
        subprocess.check_call(["git", "clone", repo_url, repo_path])
    
    # Компилируем грамматику
    logger.info(f"Компилируем грамматику для {language}...")
    
    # Обрабатываем особые случаи
    if language == "typescript":
        # TypeScript имеет несколько парсеров
        ts_parser_path = os.path.join(repo_path, "typescript", "src")
        tsx_parser_path = os.path.join(repo_path, "tsx", "src")
        
        for parser_path, parser_name in [(ts_parser_path, "typescript"), (tsx_parser_path, "tsx")]:
            if os.path.exists(parser_path):
                logger.info(f"Компилируем {parser_name}...")
                try:
                    subprocess.check_call(["npm", "install"], cwd=os.path.dirname(parser_path))
                    library_path = os.path.join(language_path, f"lib{parser_name}.so")
                    if not os.path.exists(library_path):
                        import tree_sitter
                        tree_sitter.Language.build_library(
                            library_path,
                            [os.path.dirname(parser_path)]
                        )
                        logger.info(f"Грамматика для {parser_name} успешно скомпилирована!")
                except Exception as e:
                    logger.error(f"Ошибка при компиляции грамматики для {parser_name}: {e}")
    else:
        # Стандартный случай
        parser_path = os.path.join(repo_path, "src")
        if os.path.exists(parser_path):
            try:
                library_path = os.path.join(language_path, f"lib{language}.so")
                if not os.path.exists(library_path):
                    import tree_sitter
                    tree_sitter.Language.build_library(
                        library_path,
                        [repo_path]
                    )
                    logger.info(f"Грамматика для {language} успешно скомпилирована!")
                else:
                    logger.info(f"Грамматика для {language} уже существует.")
            except Exception as e:
                logger.error(f"Ошибка при компиляции грамматики для {language}: {e}")


def main():
    """Основная функция для установки Tree-sitter и грамматик."""
    logger.info("Начинаем установку Tree-sitter и грамматик...")
    
    # Устанавливаем tree-sitter
    if not setup_tree_sitter():
        logger.error("Не удалось настроить Tree-sitter. Выходим.")
        return
    
    # Создаем директорию для парсеров, если она не существует
    parsers_path = DEFAULT_PARSERS_PATH
    os.makedirs(parsers_path, exist_ok=True)
    
    # Скачиваем и компилируем грамматики для каждого языка
    for language in LANGUAGES:
        logger.info(f"Устанавливаем грамматику для {language['name']}...")
        download_and_build_grammar(language['name'], language['repo'], parsers_path)
    
    logger.info("Установка Tree-sitter и грамматик завершена!")
    logger.info(f"Грамматики установлены в: {parsers_path}")

if __name__ == "__main__":
    main() 