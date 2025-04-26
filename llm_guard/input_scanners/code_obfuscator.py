import re
import random
import unicodedata
import hashlib
from typing import Dict, List, Tuple, Optional

from llm_guard.util import get_logger
from llm_guard.vault import Vault

from .base import Scanner
from .code import Code
from .semantic_rename_helpers import SemanticIdentifierRenamer

LOGGER = get_logger()

# Словарь гомоглифов для замены ASCII символов
HOMOGLYPHS = {
    'a': ['а', 'ɑ', 'α', 'а'],  # Латинская 'a' -> кириллическая 'а', др.
    'b': ['ƅ', 'Ь', 'Ꮟ', 'Ᏼ'],
    'c': ['с', 'ϲ', 'ϲ', 'ꮯ'],
    'd': ['ԁ', 'ɗ', 'ⅆ', 'ꓒ'],
    'e': ['е', 'ҽ', 'ⅇ', 'ℯ'],
    'g': ['ց', 'ɡ', 'ģ', 'ǵ'],
    'h': ['һ', 'հ', 'Ꮒ', 'ℎ'],
    'i': ['і', 'ⅰ', 'ӏ', 'ι'],
    'j': ['ϳ', 'ј', 'ʝ', 'ⅉ'],
    'k': ['κ', 'ⱪ', 'ᴋ', 'ｋ'],
    'l': ['ӏ', 'ⅼ', 'ｌ', 'ℓ'],
    'm': ['ṃ', 'ⅿ', 'ｍ', 'ɱ'],
    'n': ['ո', 'ƞ', 'ⁿ', 'ո'],
    'o': ['о', 'ο', 'օ', 'ℴ'],
    'p': ['р', 'ρ', 'ṗ', 'ⲣ'],
    'q': ['ԛ', 'ɋ', 'ʠ', 'զ'],
    'r': ['г', 'ꭇ', 'ꮁ', 'ⲅ'],
    's': ['ѕ', 'ꜱ', 'ꭍ', 'ꮪ'],
    't': ['t', 'ⲧ', 'τ', 'ｔ'],
    'u': ['ս', 'υ', 'ꭎ', 'ц'],
    'v': ['ѵ', 'ν', 'ꮴ', 'ⅴ'],
    'w': ['ԝ', 'ѡ', 'ꮃ', 'ꮤ'],
    'x': ['х', 'ⅹ', 'ꭓ', 'ｘ'],
    'y': ['у', 'ỿ', 'ꭚ', 'ｙ'],
    'z': ['ᴢ', 'ꮓ', 'ꭗ', 'ｚ'],
    'A': ['Α', 'А', 'Ꭺ', 'Ａ'],
    'B': ['Β', 'В', 'Ᏼ', 'Ｂ'],
    'C': ['С', 'Ϲ', 'Ⅽ', 'Ｃ'],
    'D': ['Ꭰ', 'Ⅾ', 'Ꭰ', 'Ｄ'],
    'E': ['Ε', 'Е', 'Ꭼ', 'Ｅ'],
    'F': ['Ϝ', 'Ꮀ', 'Ｆ', 'Ғ'],
    'G': ['Ԍ', 'Ꮐ', 'Ｇ', 'Ꮆ'],
    'H': ['Η', 'Н', 'Ｈ', 'Ꮋ'],
    'I': ['Ι', 'І', 'Ⅰ', 'Ｉ'],
    'J': ['Ј', 'Ꭻ', 'Ｊ', 'Ꭻ'],
    'K': ['Κ', 'К', 'Ꮶ', 'Ｋ'],
    'L': ['Ⅼ', 'Ꮮ', 'Ｌ', 'Ꮮ'],
    'M': ['Μ', 'М', 'Ⅿ', 'Ｍ'],
    'N': ['Ν', 'Ｎ', 'Ⲛ', 'Ｎ'],
    'O': ['Ο', 'О', 'Ｏ', 'Ꮎ'],
    'P': ['Ρ', 'Р', 'Ｐ', 'Ꮲ'],
    'Q': ['Ԛ', 'Ｑ', 'Ω', 'Ｑ'],
    'R': ['Ꭱ', 'Ｒ', 'Ꮢ', 'Ｒ'],
    'S': ['Ѕ', 'Ꮪ', 'Ｓ', 'Ꮪ'],
    'T': ['Τ', 'Т', 'Ｔ', 'Ꭲ'],
    'U': ['Ս', 'Ｕ', 'Ꮜ', 'Ｕ'],
    'V': ['Ѵ', 'Ｖ', 'Ꮩ', 'Ｖ'],
    'W': ['Ԝ', 'Ｗ', 'Ꮃ', 'Ｗ'],
    'X': ['Χ', 'Х', 'Ｘ', 'Ⅹ'],
    'Y': ['Υ', 'У', 'Ｙ', 'Ꮍ'],
    'Z': ['Ꮓ', 'Ζ', 'Ｚ', 'Ꮓ'],
    '0': ['０', 'Ο', 'О', '⓪'],
    '1': ['１', 'Ⅰ', 'ⅰ', '①'],
    '2': ['２', 'Ⅱ', 'ⅱ', '②'],
    '3': ['３', 'Ⅲ', 'ⅲ', '③'],
    '4': ['４', 'Ⅳ', 'ⅳ', '④'],
    '5': ['５', 'Ⅴ', 'ⅴ', '⑤'],
    '6': ['６', 'Ⅵ', 'ⅵ', '⑥'],
    '7': ['７', 'Ⅶ', 'ⅶ', '⑦'],
    '8': ['８', 'Ⅷ', 'ⅷ', '⑧'],
    '9': ['９', 'Ⅸ', 'ⅸ', '⑨'],
}

# Невидимые Unicode-символы
INVISIBLE_CHARS = [
    '\u200B',  # ZERO WIDTH SPACE
    '\u200C',  # ZERO WIDTH NON-JOINER
    '\u200D',  # ZERO WIDTH JOINER
    '\u2060',  # WORD JOINER
    '\u2061',  # FUNCTION APPLICATION
    '\u2062',  # INVISIBLE TIMES
    '\u2063',  # INVISIBLE SEPARATOR
    '\u2064',  # INVISIBLE PLUS
    '\uFEFF',  # ZERO WIDTH NO-BREAK SPACE
]


class CodeObfuscator(Scanner):
    """
    Сканер для обфускации кода в промптах.
    
    Использует различные техники обфускации:
    1. Замена ASCII-символов на гомоглифы из других алфавитов
    2. Добавление невидимых Unicode-символов
    3. Комбинирование этих подходов
    4. Семантическое переименование идентификаторов с сохранением подсказок
    
    При этом сохраняется структура кода и его понимание для LLM-моделей.
    
    Также поддерживает сохранение оригинального кода для последующей деобфускации.
    """
    
    def __init__(
        self,
        homoglyph_probability: float = 0.5,
        invisible_char_probability: float = 0.3,
        identifier_modification_probability: float = 0.4,
        preserve_keywords: bool = True,
        semantic_rename: bool = False,
        semantic_preservation_level: float = 0.7,
        domain: str = "general",
        code_detector = None,
        enable_vault: bool = False,
        vault: Vault = None
    ) -> None:
        """
        Инициализирует CodeObfuscator.
        
        Args:
            homoglyph_probability: Вероятность замены символа на гомоглиф (0-1)
            invisible_char_probability: Вероятность добавления невидимого символа (0-1)
            identifier_modification_probability: Вероятность модификации идентификаторов (0-1)
            preserve_keywords: Сохранять ли ключевые слова языков программирования без изменений
            semantic_rename: Использовать ли семантическое переименование идентификаторов
            semantic_preservation_level: Уровень сохранения семантики (0.0-1.0)
                где 0.0 - полная обфускация, 1.0 - максимальное сохранение семантики
            domain: Предметная область кода (finance, medical, web, general)
            code_detector: Опциональный детектор кода, если None - будет использован встроенный Code
            enable_vault: Включить сохранение оригинального кода для деобфускации
            vault: Объект Vault для хранения оригинального кода, если None - создается новый
        """
        self._homoglyph_probability = homoglyph_probability
        self._invisible_char_probability = invisible_char_probability
        self._identifier_modification_probability = identifier_modification_probability
        self._preserve_keywords = preserve_keywords
        self._semantic_rename = semantic_rename
        self._semantic_preservation_level = semantic_preservation_level
        self._domain = domain
        self._enable_vault = enable_vault
        
        # Инициализация хранилища для сохранения оригинального кода
        if enable_vault:
            self._vault = vault if vault is not None else Vault()
        else:
            self._vault = None
        
        # Инициализация семантического переименователя, если включено
        if semantic_rename:
            self._semantic_renamer = SemanticIdentifierRenamer(
                domain=domain,
                semantic_preservation_level=semantic_preservation_level,
                preserve_keywords=preserve_keywords
            )
        else:
            self._semantic_renamer = None
        
        # Популярные ключевые слова в разных языках программирования
        self._keywords = {
            'python': ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 'import', 'from', 'as', 'with', 'try', 'except', 'finally', 'raise', 'assert', 'lambda', 'None', 'True', 'False'],
            'javascript': ['function', 'const', 'let', 'var', 'if', 'else', 'for', 'while', 'return', 'import', 'export', 'class', 'this', 'new', 'async', 'await', 'try', 'catch', 'finally', 'throw', 'null', 'undefined', 'true', 'false'],
            'java': ['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements', 'static', 'final', 'void', 'if', 'else', 'for', 'while', 'return', 'try', 'catch', 'finally', 'throw', 'new', 'null', 'true', 'false'],
            'cpp': ['int', 'float', 'double', 'char', 'bool', 'void', 'class', 'struct', 'enum', 'template', 'namespace', 'using', 'if', 'else', 'for', 'while', 'return', 'try', 'catch', 'throw', 'new', 'delete', 'nullptr', 'true', 'false'],
            # Добавьте другие языки при необходимости
        }
        
        # Получаем объединенный список ключевых слов из всех языков
        self._all_keywords = set()
        for keywords in self._keywords.values():
            self._all_keywords.update(keywords)
        
        # Инициализация детектора кода
        if code_detector is None:
            # Используем все доступные языки из списка SUPPORTED_LANGUAGES
            from .code import SUPPORTED_LANGUAGES
            self._code_detector = Code(languages=SUPPORTED_LANGUAGES, is_blocked=False)
        else:
            self._code_detector = code_detector
    
    def _apply_homoglyphs(self, text: str) -> str:
        """Заменяет символы в тексте на их гомоглифы с заданной вероятностью"""
        result = ""
        for char in text:
            if char in HOMOGLYPHS and random.random() < self._homoglyph_probability:
                # Если это ключевое слово и мы хотим их сохранять, пропускаем замену
                if self._preserve_keywords and any(keyword in text for keyword in self._all_keywords):
                    result += char
                else:
                    result += random.choice(HOMOGLYPHS[char])
            else:
                result += char
        return result
    
    def _add_invisible_chars(self, text: str) -> str:
        """Добавляет невидимые Unicode-символы в текст с заданной вероятностью"""
        result = ""
        for char in text:
            result += char
            if random.random() < self._invisible_char_probability:
                result += random.choice(INVISIBLE_CHARS)
        return result
    
    def _modify_identifiers(self, code: str) -> str:
        """
        Модифицирует идентификаторы в коде, заменяя буквы на похожие цифры или символы
        Например: calculate -> c4lcul4t3
        """
        # Простая замена для демонстрации
        replacements = {
            'a': '4', 'e': '3', 'i': '1', 'o': '0', 'l': '1', 's': '5', 't': '7'
        }
        
        # Находим идентификаторы в коде
        # Это упрощенный подход, для реальных случаев может потребоваться использование AST
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        
        def replace_identifier(match):
            identifier = match.group(0)
            
            # Пропускаем ключевые слова, если установлен соответствующий флаг
            if self._preserve_keywords and identifier in self._all_keywords:
                return identifier
            
            # С заданной вероятностью модифицируем идентификатор
            if random.random() < self._identifier_modification_probability:
                modified = ''
                for char in identifier:
                    if char.lower() in replacements and random.random() < 0.5:
                        modified += replacements[char.lower()]
                    else:
                        modified += char
                return modified
            return identifier
        
        return re.sub(pattern, replace_identifier, code)
    
    def _obfuscate_code(self, code: str) -> str:
        """Применяет все методы обфускации к фрагменту кода"""
        # Если включено семантическое переименование, применяем его
        if self._semantic_rename and self._semantic_renamer:
            code = self._semantic_renamer.rename_identifiers_in_code(code)
        else:
            # Иначе используем базовое переименование идентификаторов
            code = self._modify_identifiers(code)
        
        # Применяем другие методы обфускации
        code = self._apply_homoglyphs(code)
        code = self._add_invisible_chars(code)
        
        return code
    
    def _generate_code_id(self, code: str) -> str:
        """Генерирует уникальный идентификатор для фрагмента кода"""
        # Используем хеш-функцию для создания идентификатора
        return f"code_{hashlib.md5(code.encode('utf-8')).hexdigest()[:8]}"
    
    def _store_in_vault(self, original_code: str, obfuscated_code: str) -> str:
        """Сохраняет оригинальный код в Vault для последующей деобфускации"""
        if not self._enable_vault or self._vault is None:
            return obfuscated_code
        
        # Генерируем уникальный идентификатор для кода
        code_id = self._generate_code_id(original_code)
        
        # Сохраняем пару (обфусцированный_код, оригинальный_код)
        self._vault.append((obfuscated_code, original_code, code_id))
        
        LOGGER.debug(f"Stored original code in vault with ID: {code_id}")
        
        return obfuscated_code
    
    def deobfuscate(self, obfuscated_prompt: str) -> str:
        """
        Деобфусцирует промпт, восстанавливая оригинальный код из Vault.
        
        Args:
            obfuscated_prompt: Обфусцированный промпт
            
        Returns:
            str: Деобфусцированный промпт с восстановленным кодом
        """
        if not self._enable_vault or self._vault is None:
            LOGGER.warning("Vault is not enabled or initialized, cannot deobfuscate")
            return obfuscated_prompt
        
        # Получаем все сохраненные пары из Vault
        stored_tuples = self._vault.get()
        
        # Для более точного восстановления ищем блоки кода
        code_pattern = r'```(?:\w*\n)?(.*?)```'
        result = obfuscated_prompt
        
        # Находим все блоки кода в обфусцированном промпте
        for match in re.finditer(code_pattern, obfuscated_prompt, re.DOTALL):
            obfuscated_block = match.group(0)  # Весь блок, включая ```
            obfuscated_code = match.group(1)   # Только код внутри блока
            
            # Ищем соответствующий оригинальный код в Vault
            original_code = None
            for stored_obfuscated, stored_original, _ in stored_tuples:
                # Проверяем, содержится ли обфусцированный код в блоке
                if obfuscated_code in stored_obfuscated or stored_obfuscated in obfuscated_code:
                    original_code = stored_original
                    LOGGER.debug(f"Found matching code block in vault")
                    break
            
            # Если нашли оригинальный код, заменяем весь блок
            if original_code:
                new_block = f"```\n{original_code}\n```"
                result = result.replace(obfuscated_block, new_block)
                LOGGER.debug(f"Deobfuscated code block")
        
        return result
    
    def get_vault(self) -> Vault:
        """Возвращает объект Vault с сохраненными парами"""
        return self._vault
    
    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """
        Обфусцирует код в промпте, оставляя остальной текст без изменений.
        
        Args:
            prompt: Входной промпт, который может содержать фрагменты кода
            
        Returns:
            str: Промпт с обфусцированным кодом
            bool: Всегда True, так как это не фильтр, а трансформатор
            float: 0.0, так как мы не оцениваем риск, а просто обфусцируем код
        """
        # Используем детектор кода для нахождения фрагментов кода в промпте
        code_blocks = []
        non_code_blocks = []
        
        # Детектируем код в промпте
        # Для простоты, ищем блоки кода, заключенные в ```
        code_pattern = r'```(?:\w*\n)?(.*?)```'
        last_end = 0
        
        for match in re.finditer(code_pattern, prompt, re.DOTALL):
            start, end = match.span()
            code = match.group(1)
            
            # Добавляем текст перед кодом
            if start > last_end:
                non_code_blocks.append(prompt[last_end:start])
            
            # Обфусцируем код
            obfuscated_code = self._obfuscate_code(code)
            
            # Сохраняем оригинальный код в Vault, если включено
            if self._enable_vault and self._vault is not None:
                self._store_in_vault(code, obfuscated_code)
            
            code_blocks.append((f"```\n{obfuscated_code}\n```", start, end))
            
            last_end = end
        
        # Добавляем оставшийся текст после последнего блока кода
        if last_end < len(prompt):
            non_code_blocks.append(prompt[last_end:])
        
        # Сортируем блоки кода по их позиции в тексте
        code_blocks.sort(key=lambda x: x[1])
        
        # Собираем результат
        result = ""
        if code_blocks:
            current_pos = 0
            block_idx = 0
            for block_idx, (obfuscated, start, end) in enumerate(code_blocks):
                # Добавляем текст перед блоком кода
                if block_idx < len(non_code_blocks):
                    result += non_code_blocks[block_idx]
                
                # Добавляем обфусцированный код
                result += obfuscated
                current_pos = end
            
            # Добавляем оставшийся текст
            if current_pos < len(prompt) and block_idx + 1 < len(non_code_blocks):
                result += non_code_blocks[block_idx + 1]
        else:
            # Если блоков кода нет, возвращаем исходный промпт
            result = prompt
        
        LOGGER.info("Code in prompt has been obfuscated")
        
        return result, True, 0.0 