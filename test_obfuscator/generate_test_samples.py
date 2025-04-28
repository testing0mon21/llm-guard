#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для генерации тестовых примеров с кодом на разных языках программирования.

Этот скрипт создает набор тестов с примерами кода на разных языках для
тестирования функциональности обфускации кода в LLM Guard.
"""

import os
import json
import random
from pathlib import Path
import argparse

# Импортируем дополнительные примеры кода
from language_examples import ADDITIONAL_CODE_EXAMPLES

# Директории
TEST_DIR = Path(__file__).parent
OUTPUT_DIR = TEST_DIR / "generated_tests"
OUTPUT_DIR.mkdir(exist_ok=True)

# Языки программирования для тестирования
LANGUAGES = [
    "python", 
    "javascript", 
    "java", 
    "cpp", 
    "go", 
    "typescript", 
    "php", 
    "rust", 
    "ruby", 
    "csharp"
]

# Типы задач
TASK_TYPES = [
    "utility_function",
    "algorithm",
    "data_processing",
    "authentication",
    "security",
    "file_operations",
    "database",
    "api_client",
    "config",
    "web"
]

# Шаблоны запросов
PROMPT_TEMPLATES = [
    "Можешь ли ты улучшить следующую функцию на {language}?",
    "Проанализируй этот код на {language} и предложи улучшения:",
    "Какие недостатки есть в этой реализации на {language}?",
    "Как оптимизировать этот код на {language}?",
    "Проверь этот код на {language} на наличие проблем безопасности:",
    "Как сделать этот код на {language} более читаемым и поддерживаемым?",
    "Помоги улучшить этот код на {language}, особенно с точки зрения обработки ошибок:",
    "Как бы ты рефакторинговал этот код на {language}?",
    "Какие тесты ты бы написал для этого кода на {language}?",
    "Есть ли более элегантный способ написать этот код на {language}?"
]

# Примеры кода для разных языков программирования
CODE_EXAMPLES = {
    "python": {
        "utility_function": [
"""def validate_email(email):
    \"\"\"Validate an email address using regex.\"\"\"
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True
    else:
        return False""",

"""def generate_password(length=12):
    \"\"\"Generate a random password with the specified length.\"\"\"
    import random
    import string
    characters = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.choice(characters) for _ in range(length))
    return password""",

"""def flatten_list(nested_list):
    \"\"\"Flatten a nested list.\"\"\"
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened"""
        ],
        "algorithm": [
"""def binary_search(arr, target):
    \"\"\"Perform binary search on a sorted array.\"\"\"
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",

"""def quicksort(arr):
    \"\"\"Quicksort implementation.\"\"\"
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)""",

"""def fibonacci(n, memo={}):
    \"\"\"Calculate the nth Fibonacci number using memoization.\"\"\"
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]"""
        ],
        "security": [
"""def hash_password(password, salt=None):
    \"\"\"Hash a password using PBKDF2.\"\"\"
    import hashlib
    import os
    
    if salt is None:
        salt = os.urandom(32)
    
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return {'salt': salt, 'key': key}""",

"""def sanitize_input(user_input):
    \"\"\"Sanitize user input to prevent injection attacks.\"\"\"
    import re
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>\'";]', '', user_input)
    return sanitized""",

"""def verify_jwt_token(token, secret_key):
    \"\"\"Verify a JWT token.\"\"\"
    import jwt
    
    try:
        decoded = jwt.decode(token, secret_key, algorithms=['HS256'])
        return True, decoded
    except jwt.InvalidTokenError:
        return False, None"""
        ],
        "database": [
"""def connect_to_database(db_params):
    \"\"\"Connect to a PostgreSQL database.\"\"\"
    import psycopg2
    
    conn = None
    try:
        conn = psycopg2.connect(
            host=db_params['host'],
            database=db_params['database'],
            user=db_params['user'],
            password=db_params['password']
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        if conn is not None:
            conn.close()
        return None""",

"""def execute_query(connection, query, params=None):
    \"\"\"Execute a SQL query with parameters.\"\"\"
    try:
        cursor = connection.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        return results
    except Exception as e:
        print(f"Error executing query: {e}")
        return None""",

"""class DatabaseManager:
    \"\"\"A simple database connection manager.\"\"\"
    
    def __init__(self, connection_string):
        import sqlite3
        self.connection_string = connection_string
        self.connection = None
        
    def __enter__(self):
        self.connection = sqlite3.connect(self.connection_string)
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()"""
        ]
    },
    "javascript": {
        "utility_function": [
"""function validateEmail(email) {
  // Validate an email address using regex
  const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  return pattern.test(email);
}""",

"""function generatePassword(length = 12) {
  // Generate a random password with the specified length
  const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+~`|}{[]:;?><,./-=';
  let password = '';
  for (let i = 0; i < length; i++) {
    const randomIndex = Math.floor(Math.random() * charset.length);
    password += charset[randomIndex];
  }
  return password;
}""",

"""function flattenArray(arr) {
  // Flatten a nested array
  return arr.reduce((flat, toFlatten) => {
    return flat.concat(Array.isArray(toFlatten) ? flattenArray(toFlatten) : toFlatten);
  }, []);
}"""
        ],
        "algorithm": [
"""function binarySearch(arr, target) {
  // Perform binary search on a sorted array
  let left = 0;
  let right = arr.length - 1;
  
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    if (arr[mid] === target) {
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  
  return -1;
}""",

"""function quickSort(arr) {
  // Quicksort implementation
  if (arr.length <= 1) {
    return arr;
  }
  
  const pivot = arr[Math.floor(arr.length / 2)];
  const left = arr.filter(x => x < pivot);
  const middle = arr.filter(x => x === pivot);
  const right = arr.filter(x => x > pivot);
  
  return [...quickSort(left), ...middle, ...quickSort(right)];
}""",

"""function fibonacci(n, memo = {}) {
  // Calculate the nth Fibonacci number using memoization
  if (n in memo) {
    return memo[n];
  }
  if (n <= 2) {
    return 1;
  }
  
  memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo);
  return memo[n];
}"""
        ],
        "security": [
"""async function hashPassword(password, salt = null) {
  // Hash a password using Web Crypto API
  const encoder = new TextEncoder();
  
  if (!salt) {
    salt = window.crypto.getRandomValues(new Uint8Array(16));
  }
  
  const passwordData = encoder.encode(password);
  const importedKey = await window.crypto.subtle.importKey(
    'raw',
    passwordData,
    { name: 'PBKDF2' },
    false,
    ['deriveBits', 'deriveKey']
  );
  
  const derivedKey = await window.crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt: salt,
      iterations: 100000,
      hash: 'SHA-256'
    },
    importedKey,
    { name: 'AES-GCM', length: 256 },
    true,
    ['encrypt', 'decrypt']
  );
  
  return { salt, key: derivedKey };
}""",

"""function sanitizeInput(userInput) {
  // Sanitize user input to prevent XSS attacks
  const sanitized = userInput
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
  return sanitized;
}""",

"""function verifyJwtToken(token, secretKey) {
  // Simplified JWT verification (for demonstration)
  try {
    const [header, payload, signature] = token.split('.');
    
    // In a real implementation, you would:
    // 1. Decode header and payload
    // 2. Verify signature using the secret key
    // 3. Check expiration and other claims
    
    const decodedPayload = JSON.parse(atob(payload));
    const currentTime = Math.floor(Date.now() / 1000);
    
    if (decodedPayload.exp && decodedPayload.exp < currentTime) {
      return { valid: false, reason: 'Token expired' };
    }
    
    return { valid: true, payload: decodedPayload };
  } catch (error) {
    return { valid: false, reason: 'Invalid token format' };
  }
}"""
        ]
    },
    "java": {
        "utility_function": [
"""public class EmailValidator {
    /**
     * Validate an email address using regex
     * @param email the email to validate
     * @return true if valid, false otherwise
     */
    public static boolean validateEmail(String email) {
        String pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}$";
        return email.matches(pattern);
    }
}""",

"""import java.security.SecureRandom;

public class PasswordGenerator {
    private static final String CHAR_LOWER = "abcdefghijklmnopqrstuvwxyz";
    private static final String CHAR_UPPER = CHAR_LOWER.toUpperCase();
    private static final String NUMBER = "0123456789";
    private static final String SPECIAL = "!@#$%^&*()_-+=<>?/{}[]|";
    private static final String ALL_CHARS = CHAR_LOWER + CHAR_UPPER + NUMBER + SPECIAL;
    
    private static final SecureRandom random = new SecureRandom();
    
    /**
     * Generate a random password with a specified length
     * @param length the password length
     * @return a random password
     */
    public static String generatePassword(int length) {
        if (length < 8) {
            throw new IllegalArgumentException("Password length must be at least 8");
        }
        
        StringBuilder password = new StringBuilder(length);
        for (int i = 0; i < length; i++) {
            int randomIndex = random.nextInt(ALL_CHARS.length());
            password.append(ALL_CHARS.charAt(randomIndex));
        }
        
        return password.toString();
    }
}""",

"""import java.util.ArrayList;
import java.util.List;

public class ListFlattener {
    /**
     * Flatten a nested list of objects
     * @param nestedList the nested list
     * @return flattened list
     */
    public static <T> List<T> flattenList(List<?> nestedList) {
        List<T> result = new ArrayList<>();
        flattenListHelper(nestedList, result);
        return result;
    }
    
    @SuppressWarnings("unchecked")
    private static <T> void flattenListHelper(List<?> nestedList, List<T> result) {
        for (Object item : nestedList) {
            if (item instanceof List<?>) {
                flattenListHelper((List<?>) item, result);
            } else {
                result.add((T) item);
            }
        }
    }
}"""
        ],
        "algorithm": [
"""public class BinarySearch {
    /**
     * Perform binary search on a sorted array
     * @param arr the sorted array
     * @param target the target value
     * @return index of target or -1 if not found
     */
    public static int binarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }
}""",

"""import java.util.ArrayList;
import java.util.List;

public class QuickSort {
    /**
     * Sort an array using QuickSort algorithm
     * @param arr the array to sort
     * @return sorted array
     */
    public static List<Integer> quickSort(List<Integer> arr) {
        if (arr.size() <= 1) {
            return arr;
        }
        
        Integer pivot = arr.get(arr.size() / 2);
        List<Integer> left = new ArrayList<>();
        List<Integer> middle = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        
        for (Integer num : arr) {
            if (num < pivot) {
                left.add(num);
            } else if (num.equals(pivot)) {
                middle.add(num);
            } else {
                right.add(num);
            }
        }
        
        List<Integer> result = new ArrayList<>();
        result.addAll(quickSort(left));
        result.addAll(middle);
        result.addAll(quickSort(right));
        
        return result;
    }
}""",

"""import java.util.HashMap;
import java.util.Map;

public class Fibonacci {
    private static Map<Integer, Long> memo = new HashMap<>();
    
    /**
     * Calculate the nth Fibonacci number using memoization
     * @param n the position in the sequence
     * @return the nth Fibonacci number
     */
    public static long fibonacci(int n) {
        if (memo.containsKey(n)) {
            return memo.get(n);
        }
        
        if (n <= 2) {
            return 1;
        }
        
        long result = fibonacci(n - 1) + fibonacci(n - 2);
        memo.put(n, result);
        
        return result;
    }
}"""
        ]
    }
}

# Объединяем примеры кода из основного и дополнительного набора
for language, examples in ADDITIONAL_CODE_EXAMPLES.items():
    if language not in CODE_EXAMPLES:
        CODE_EXAMPLES[language] = {}
    
    for task_type, code_samples in examples.items():
        if task_type not in CODE_EXAMPLES[language]:
            CODE_EXAMPLES[language][task_type] = []
        
        CODE_EXAMPLES[language][task_type].extend(code_samples)

# Словарь соответствия языков программирования и их названий для промптов
LANGUAGE_NAMES = {
    "python": "Python",
    "javascript": "JavaScript",
    "java": "Java",
    "cpp": "C++",
    "go": "Go",
    "typescript": "TypeScript",
    "php": "PHP",
    "rust": "Rust",
    "ruby": "Ruby",
    "csharp": "C#"
}

# Словарь соответствия типов задач и их описаний
TASK_DESCRIPTIONS = {
    "utility_function": "утилитарная функция",
    "algorithm": "алгоритм",
    "data_processing": "обработка данных",
    "authentication": "аутентификация",
    "security": "безопасность",
    "file_operations": "работа с файлами",
    "database": "база данных",
    "api_client": "API клиент",
    "config": "конфигурация",
    "web": "веб-разработка"
}


def generate_test_case(language, task_type, code_example, case_id):
    """
    Генерирует тестовый случай с кодом на определенном языке.
    
    Args:
        language: Язык программирования
        task_type: Тип задачи
        code_example: Пример кода
        case_id: Идентификатор теста
        
    Returns:
        dict: Тестовый случай
    """
    # Выбираем случайный шаблон запроса
    prompt_template = random.choice(PROMPT_TEMPLATES)
    
    # Подставляем название языка в шаблон
    language_name = LANGUAGE_NAMES.get(language, language)
    prompt = prompt_template.format(language=language_name)
    
    # Добавляем описание типа задачи, если есть
    task_description = TASK_DESCRIPTIONS.get(task_type, task_type)
    if random.random() < 0.7:  # В 70% случаев добавляем описание типа задачи
        prompt += f" Этот код является {task_description}."
    
    # Формируем тестовый случай
    test_case = {
        "id": case_id,
        "language": language,
        "task_type": task_type,
        "prompt": prompt,
        "code": code_example
    }
    
    return test_case


def generate_test_cases(num_cases=100):
    """
    Генерирует указанное количество тестовых случаев.
    
    Args:
        num_cases: Количество тестовых случаев для генерации
        
    Returns:
        list: Список сгенерированных тестовых случаев
    """
    test_cases = []
    case_id = 0
    
    languages_with_examples = [lang for lang in LANGUAGES if lang in CODE_EXAMPLES]
    
    # Для равномерного распределения по языкам
    cases_per_language = num_cases // len(languages_with_examples)
    remaining_cases = num_cases % len(languages_with_examples)
    
    for language in languages_with_examples:
        language_code_examples = CODE_EXAMPLES[language]
        
        # Количество случаев для текущего языка
        num_lang_cases = cases_per_language + (1 if remaining_cases > 0 else 0)
        if remaining_cases > 0:
            remaining_cases -= 1
        
        # Генерируем тесты для текущего языка
        for _ in range(num_lang_cases):
            # Выбираем случайный тип задачи из доступных для этого языка
            available_task_types = list(language_code_examples.keys())
            task_type = random.choice(available_task_types)
            
            # Выбираем случайный пример кода для этого типа задачи
            code_examples = language_code_examples[task_type]
            code_example = random.choice(code_examples)
            
            # Создаем тестовый случай
            test_case = generate_test_case(language, task_type, code_example, case_id)
            test_cases.append(test_case)
            
            case_id += 1
    
    return test_cases


def save_test_cases(test_cases):
    """
    Сохраняет тестовые случаи в файл и создает примеры кода в отдельных файлах.
    
    Args:
        test_cases: Список тестовых случаев
    """
    # Сохраняем все тестовые случаи в JSON-файл
    json_file = OUTPUT_DIR / "generated_test_cases.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)
    
    print(f"Сохранено {len(test_cases)} тестовых случаев в {json_file}")
    
    # Создаем директорию для примеров кода
    code_samples_dir = OUTPUT_DIR / "code_samples"
    code_samples_dir.mkdir(exist_ok=True)
    
    # Сохраняем примеры кода в отдельные файлы по языкам
    for case in test_cases:
        language = case["language"]
        lang_dir = code_samples_dir / language
        lang_dir.mkdir(exist_ok=True)
        
        # Определяем расширение файла в зависимости от языка
        ext = get_file_extension(language)
        
        # Сохраняем код в файл
        with open(lang_dir / f"sample_{case['id']}{ext}", "w", encoding="utf-8") as f:
            f.write(case["code"])


def get_file_extension(language):
    """
    Возвращает расширение файла для указанного языка программирования.
    
    Args:
        language: Язык программирования
        
    Returns:
        str: Расширение файла
    """
    extensions = {
        "python": ".py",
        "javascript": ".js",
        "java": ".java",
        "cpp": ".cpp",
        "go": ".go",
        "typescript": ".ts",
        "php": ".php",
        "rust": ".rs",
        "ruby": ".rb",
        "csharp": ".cs"
    }
    return extensions.get(language, ".txt")


def main():
    """Основная функция для запуска генерации тестов."""
    parser = argparse.ArgumentParser(description="Генерация тестовых примеров с кодом на разных языках")
    parser.add_argument("--num", type=int, default=100, help="Количество тестовых случаев для генерации")
    parser.add_argument("--output", type=str, default=None, help="Директория для сохранения результатов")
    
    args = parser.parse_args()
    
    # Если указана пользовательская директория для вывода
    global OUTPUT_DIR
    if args.output:
        OUTPUT_DIR = Path(args.output)
        OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"Генерация {args.num} тестовых примеров...")
    
    # Генерируем тестовые случаи
    test_cases = generate_test_cases(args.num)
    
    # Сохраняем результаты
    save_test_cases(test_cases)
    
    print("Генерация тестовых примеров завершена.")
    print(f"Результаты сохранены в директории: {OUTPUT_DIR}")
    
    # Выводим статистику по языкам
    language_stats = {}
    for case in test_cases:
        language = case["language"]
        if language not in language_stats:
            language_stats[language] = 0
        language_stats[language] += 1
    
    print("\nСтатистика по языкам:")
    for language, count in language_stats.items():
        print(f"- {LANGUAGE_NAMES.get(language, language)}: {count}")


if __name__ == "__main__":
    main() 