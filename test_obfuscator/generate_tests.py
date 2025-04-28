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

# Директории
TEST_DIR = Path(__file__).parent
OUTPUT_DIR = TEST_DIR / "test_data"
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

"""def verify_jwt_token(token, secret_key):
    \"\"\"Verify a JWT token.\"\"\"
    import jwt
    
    try:
        decoded = jwt.decode(token, secret_key, algorithms=['HS256'])
        return True, decoded
    except jwt.InvalidTokenError:
        return False, None"""
        ],
        "data_processing": [
"""def parse_csv(file_path):
    \"\"\"Parse a CSV file and return a list of dictionaries.\"\"\"
    import csv
    
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data""",

"""def json_to_csv(json_data, output_file):
    \"\"\"Convert JSON data to CSV format.\"\"\"
    import csv
    import json
    
    with open(json_data, 'r') as json_file:
        data = json.load(json_file)
    
    keys = data[0].keys() if data else []
    
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)"""
        ],
        "database": [
"""def connect_to_postgres(db_params):
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
        return None"""
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
}"""
        ],
        "security": [
"""function hashPassword(password, salt = null) {
  // Hash a password using PBKDF2
  const crypto = require('crypto');
  
  if (!salt) {
    salt = crypto.randomBytes(16).toString('hex');
  }
  
  const hash = crypto.pbkdf2Sync(password, salt, 1000, 64, 'sha512').toString('hex');
  return { salt, hash };
}""",

"""function verifyJwtToken(token, secretKey) {
  // Verify a JWT token
  const jwt = require('jsonwebtoken');
  
  try {
    const decoded = jwt.verify(token, secretKey);
    return { valid: true, payload: decoded };
  } catch (err) {
    return { valid: false, payload: null };
  }
}"""
        ]
    },
    "java": {
        "utility_function": [
"""public class EmailValidator {
    /**
     * Validate an email address using regex.
     */
    public static boolean validateEmail(String email) {
        String pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$";
        return email.matches(pattern);
    }
}""",

"""import java.security.SecureRandom;

public class PasswordGenerator {
    /**
     * Generate a random password with the specified length.
     */
    public static String generatePassword(int length) {
        final String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?";
        SecureRandom random = new SecureRandom();
        StringBuilder sb = new StringBuilder();
        
        for (int i = 0; i < length; i++) {
            int randomIndex = random.nextInt(chars.length());
            sb.append(chars.charAt(randomIndex));
        }
        
        return sb.toString();
    }
}"""
        ],
        "algorithm": [
"""public class BinarySearch {
    /**
     * Perform binary search on a sorted array.
     */
    public static int binarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left <= right) {
            int mid = (left + right) / 2;
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
     * Quicksort implementation for integers.
     */
    public static List<Integer> quickSort(List<Integer> arr) {
        if (arr.size() <= 1) {
            return arr;
        }
        
        Integer pivot = arr.get(arr.size() / 2);
        List<Integer> left = new ArrayList<>();
        List<Integer> middle = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        
        for (Integer x : arr) {
            if (x < pivot) {
                left.add(x);
            } else if (x.equals(pivot)) {
                middle.add(x);
            } else {
                right.add(x);
            }
        }
        
        List<Integer> result = new ArrayList<>();
        result.addAll(quickSort(left));
        result.addAll(middle);
        result.addAll(quickSort(right));
        
        return result;
    }
}"""
        ],
        "security": [
"""import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;
import java.security.Key;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public class JwtTokenUtil {
    private final Key key = Keys.secretKeyFor(SignatureAlgorithm.HS256);
    
    /**
     * Generate a JWT token for a user.
     */
    public String generateToken(String username) {
        Map<String, Object> claims = new HashMap<>();
        return createToken(claims, username);
    }
    
    private String createToken(Map<String, Object> claims, String subject) {
        return Jwts.builder()
            .setClaims(claims)
            .setSubject(subject)
            .setIssuedAt(new Date(System.currentTimeMillis()))
            .setExpiration(new Date(System.currentTimeMillis() + 1000 * 60 * 60 * 10)) // 10 hours
            .signWith(key)
            .compact();
    }
    
    /**
     * Validate a JWT token.
     */
    public Boolean validateToken(String token) {
        try {
            Jwts.parserBuilder().setSigningKey(key).build().parseClaimsJws(token);
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Extract username from JWT token.
     */
    public String extractUsername(String token) {
        return extractClaim(token, Claims::getSubject);
    }
    
    private <T> T extractClaim(String token, Function<Claims, T> claimsResolver) {
        final Claims claims = extractAllClaims(token);
        return claimsResolver.apply(claims);
    }
    
    private Claims extractAllClaims(String token) {
        return Jwts.parserBuilder().setSigningKey(key).build().parseClaimsJws(token).getBody();
    }
}"""
        ]
    }
}

def generate_test_case(language, task_type, code_example, case_id):
    """
    Генерирует тестовый пример с кодом на указанном языке программирования.
    """
    # Выбираем случайный шаблон запроса
    prompt_template = random.choice(PROMPT_TEMPLATES)
    prompt = prompt_template.format(language=language)
    
    test_case = {
        "id": case_id,
        "language": language,
        "prompt": prompt,
        "task_type": task_type,
        "code": code_example
    }
    
    return test_case

def generate_test_cases(num_cases=100):
    """
    Генерирует указанное количество тестовых примеров для разных языков.
    """
    test_cases = []
    case_id = 0
    
    # Распределяем примеры равномерно по языкам и типам задач
    languages_count = len(LANGUAGES)
    cases_per_language = num_cases // languages_count
    
    for language in LANGUAGES:
        if language not in CODE_EXAMPLES:
            # Пропускаем языки, для которых у нас нет примеров
            continue
            
        language_examples = CODE_EXAMPLES[language]
        
        # Для каждого типа задачи в данном языке
        for task_type, examples in language_examples.items():
            # Для каждого примера кода в данном типе задачи
            for example in examples:
                if case_id >= num_cases:
                    break
                    
                test_case = generate_test_case(language, task_type, example, case_id)
                test_cases.append(test_case)
                case_id += 1
                
            if case_id >= num_cases:
                break
                
        if case_id >= num_cases:
            break
            
    # Если у нас недостаточно примеров, используем повторно существующие
    while len(test_cases) < num_cases:
        # Выбираем случайный язык
        language = random.choice(list(CODE_EXAMPLES.keys()))
        language_examples = CODE_EXAMPLES[language]
        
        # Выбираем случайный тип задачи
        task_type = random.choice(list(language_examples.keys()))
        examples = language_examples[task_type]
        
        # Выбираем случайный пример кода
        example = random.choice(examples)
        
        test_case = generate_test_case(language, task_type, example, case_id)
        test_cases.append(test_case)
        case_id += 1
    
    return test_cases

def save_test_cases(test_cases):
    """
    Сохраняет тестовые примеры в JSON-файл.
    """
    output_file = OUTPUT_DIR / "test_cases.json"
    
    with open(output_file, 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"Сохранено {len(test_cases)} тестовых примеров в {output_file}")
    
    # Также сохраняем отдельные файлы с кодом для каждого языка
    for language in set(case["language"] for case in test_cases):
        language_dir = OUTPUT_DIR / "code_samples" / language
        language_dir.mkdir(exist_ok=True, parents=True)
        
        language_cases = [case for case in test_cases if case["language"] == language]
        for idx, case in enumerate(language_cases):
            ext = get_file_extension(language)
            file_path = language_dir / f"sample_{idx}{ext}"
            
            with open(file_path, 'w') as f:
                f.write(case["code"])

def get_file_extension(language):
    """
    Возвращает расширение файла для данного языка программирования.
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
    parser = argparse.ArgumentParser(description="Генератор тестовых примеров с кодом на различных языках программирования")
    parser.add_argument('--num', type=int, default=100, help="Количество примеров для генерации (по умолчанию: 100)")
    args = parser.parse_args()
    
    test_cases = generate_test_cases(args.num)
    save_test_cases(test_cases)

if __name__ == "__main__":
    main() 