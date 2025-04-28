"""
Модуль с примерами кода на различных языках программирования для генерации тестов.
"""

# Примеры кода на Go
GO_EXAMPLES = {
    "utility_function": [
"""package utils

import (
	"regexp"
)

// ValidateEmail validates an email address using regex
func ValidateEmail(email string) bool {
	pattern := `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
	matched, _ := regexp.MatchString(pattern, email)
	return matched
}""",

"""package utils

import (
	"crypto/rand"
	"math/big"
)

// GeneratePassword generates a random password with the specified length
func GeneratePassword(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+"
	password := make([]byte, length)
	
	for i := range password {
		n, _ := rand.Int(rand.Reader, big.NewInt(int64(len(charset))))
		password[i] = charset[n.Int64()]
	}
	
	return string(password)
}""",

"""package utils

// FlattenSlice flattens a nested slice of interfaces
func FlattenSlice(nested []interface{}) []interface{} {
	result := make([]interface{}, 0)
	
	for _, item := range nested {
		switch v := item.(type) {
		case []interface{}:
			result = append(result, FlattenSlice(v)...)
		default:
			result = append(result, item)
		}
	}
	
	return result
}"""
    ],
    "algorithm": [
"""package algorithm

// BinarySearch performs binary search on a sorted slice
func BinarySearch(arr []int, target int) int {
	left, right := 0, len(arr)-1
	
	for left <= right {
		mid := (left + right) / 2
		
		if arr[mid] == target {
			return mid
		} else if arr[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	
	return -1
}""",

"""package algorithm

// QuickSort sorts a slice using the QuickSort algorithm
func QuickSort(arr []int) []int {
	if len(arr) <= 1 {
		return arr
	}
	
	pivot := arr[len(arr)/2]
	var left, middle, right []int
	
	for _, num := range arr {
		if num < pivot {
			left = append(left, num)
		} else if num == pivot {
			middle = append(middle, num)
		} else {
			right = append(right, num)
		}
	}
	
	result := append(QuickSort(left), middle...)
	result = append(result, QuickSort(right)...)
	
	return result
}""",

"""package algorithm

// Fibonacci calculates the nth Fibonacci number using memoization
func Fibonacci(n int, memo map[int]int) int {
	if memo == nil {
		memo = make(map[int]int)
	}
	
	if val, ok := memo[n]; ok {
		return val
	}
	
	if n <= 2 {
		return 1
	}
	
	memo[n] = Fibonacci(n-1, memo) + Fibonacci(n-2, memo)
	return memo[n]
}"""
    ],
    "security": [
"""package security

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"golang.org/x/crypto/argon2"
)

// HashPassword hashes a password using Argon2
func HashPassword(password string, salt []byte) (string, []byte) {
	if salt == nil {
		salt = make([]byte, 16)
		rand.Read(salt)
	}
	
	hash := argon2.IDKey([]byte(password), salt, 3, 64*1024, 4, 32)
	
	return base64.StdEncoding.EncodeToString(hash), salt
}

// VerifyPassword verifies a password against a hash
func VerifyPassword(password, encodedHash string, salt []byte) bool {
	hash, err := base64.StdEncoding.DecodeString(encodedHash)
	if err != nil {
		return false
	}
	
	newHash := argon2.IDKey([]byte(password), salt, 3, 64*1024, 4, 32)
	
	return subtle.ConstantTimeCompare(hash, newHash) == 1
}""",

"""package security

import (
	"html"
	"strings"
)

// SanitizeInput sanitizes user input to prevent XSS attacks
func SanitizeInput(input string) string {
	// Replace potentially dangerous characters
	sanitized := html.EscapeString(input)
	
	// Remove suspicious patterns
	sanitized = strings.ReplaceAll(sanitized, "javascript:", "")
	sanitized = strings.ReplaceAll(sanitized, "data:", "")
	
	return sanitized
}"""
    ]
}

# Примеры кода на Ruby
RUBY_EXAMPLES = {
    "utility_function": [
"""# Validate an email address using regex
def validate_email(email)
  pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
  !!(email =~ pattern)
end""",

"""# Generate a random password with the specified length
def generate_password(length = 12)
  charset = ('a'..'z').to_a + ('A'..'Z').to_a + ('0'..'9').to_a + ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '-', '=']
  password = (0...length).map { charset.sample }.join
  password
end""",

"""# Flatten a nested array
def flatten_array(nested_array)
  result = []
  
  nested_array.each do |element|
    if element.is_a?(Array)
      result.concat(flatten_array(element))
    else
      result << element
    end
  end
  
  result
end"""
    ],
    "algorithm": [
"""# Perform binary search on a sorted array
def binary_search(arr, target)
  left = 0
  right = arr.length - 1
  
  while left <= right
    mid = (left + right) / 2
    
    if arr[mid] == target
      return mid
    elsif arr[mid] < target
      left = mid + 1
    else
      right = mid - 1
    end
  end
  
  -1  # Not found
end""",

"""# Sort an array using the QuickSort algorithm
def quick_sort(arr)
  return arr if arr.length <= 1
  
  pivot = arr[arr.length / 2]
  left = arr.select { |x| x < pivot }
  middle = arr.select { |x| x == pivot }
  right = arr.select { |x| x > pivot }
  
  quick_sort(left) + middle + quick_sort(right)
end""",

"""# Calculate the nth Fibonacci number using memoization
def fibonacci(n, memo = {})
  return memo[n] if memo.key?(n)
  return 1 if n <= 2
  
  memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
  memo[n]
end"""
    ],
    "security": [
"""require 'openssl'
require 'securerandom'
require 'base64'

# Hash a password using PBKDF2-HMAC-SHA256
def hash_password(password, salt = nil)
  salt ||= SecureRandom.random_bytes(16)
  
  # 10000 iterations, 32 bytes output
  digest = OpenSSL::Digest::SHA256.new
  key_len = 32
  iterations = 10000
  
  key = OpenSSL::PKCS5.pbkdf2_hmac(
    password,
    salt,
    iterations,
    key_len,
    digest
  )
  
  {
    salt: Base64.strict_encode64(salt),
    key: Base64.strict_encode64(key)
  }
end

# Verify a password against a stored hash
def verify_password(password, stored_hash)
  salt = Base64.strict_decode64(stored_hash[:salt])
  
  new_hash = hash_password(password, salt)
  new_hash[:key] == stored_hash[:key]
end""",

"""# Sanitize user input to prevent XSS attacks
def sanitize_input(user_input)
  # Replace potentially dangerous characters
  sanitized = user_input
    .gsub('<', '&lt;')
    .gsub('>', '&gt;')
    .gsub('"', '&quot;')
    .gsub("'", '&#39;')
    .gsub('/', '&#x2F;')
    
  sanitized
end"""
    ]
}

# Примеры кода на PHP
PHP_EXAMPLES = {
    "utility_function": [
"""<?php
/**
 * Validate an email address using regex
 * 
 * @param string $email The email to validate
 * @return bool True if valid, false otherwise
 */
function validateEmail($email) {
    $pattern = '/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/';
    return (bool) preg_match($pattern, $email);
}
?>""",

"""<?php
/**
 * Generate a random password with the specified length
 * 
 * @param int $length The password length
 * @return string The generated password
 */
function generatePassword($length = 12) {
    $charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+';
    $password = '';
    
    $charsetLength = strlen($charset) - 1;
    
    for ($i = 0; $i < $length; $i++) {
        $password .= $charset[random_int(0, $charsetLength)];
    }
    
    return $password;
}
?>""",

"""<?php
/**
 * Flatten a nested array
 * 
 * @param array $nestedArray The array to flatten
 * @return array The flattened array
 */
function flattenArray($nestedArray) {
    $result = [];
    
    foreach ($nestedArray as $item) {
        if (is_array($item)) {
            $result = array_merge($result, flattenArray($item));
        } else {
            $result[] = $item;
        }
    }
    
    return $result;
}
?>"""
    ],
    "security": [
"""<?php
/**
 * Hash a password using password_hash
 * 
 * @param string $password The password to hash
 * @return string The hashed password
 */
function hashPassword($password) {
    // Using PASSWORD_DEFAULT will use the strongest algorithm available
    $hashedPassword = password_hash($password, PASSWORD_DEFAULT);
    
    return $hashedPassword;
}

/**
 * Verify a password against a hash
 * 
 * @param string $password The password to verify
 * @param string $hash The hash to verify against
 * @return bool True if password matches, false otherwise
 */
function verifyPassword($password, $hash) {
    return password_verify($password, $hash);
}
?>""",

"""<?php
/**
 * Sanitize user input to prevent XSS attacks
 * 
 * @param string $input The user input to sanitize
 * @return string The sanitized input
 */
function sanitizeInput($input) {
    // Convert special characters to HTML entities
    $sanitized = htmlspecialchars($input, ENT_QUOTES | ENT_HTML5, 'UTF-8');
    
    return $sanitized;
}
?>"""
    ]
}

# Примеры кода на Rust
RUST_EXAMPLES = {
    "utility_function": [
"""use regex::Regex;

/// Validate an email address using regex
pub fn validate_email(email: &str) -> bool {
    let re = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
    re.is_match(email)
}""",

"""use rand::{Rng, thread_rng};
use rand::distributions::Alphanumeric;

/// Generate a random password with the specified length
pub fn generate_password(length: usize) -> String {
    let mut rng = thread_rng();
    let charset: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                            abcdefghijklmnopqrstuvwxyz\
                            0123456789\
                            !@#$%^&*()";
    
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..charset.len());
            charset[idx] as char
        })
        .collect()
}""",

"""/// Flatten a nested vector
pub fn flatten_vec<T: Clone>(nested: &Vec<Vec<T>>) -> Vec<T> {
    let mut result = Vec::new();
    
    for inner in nested {
        for item in inner {
            result.push(item.clone());
        }
    }
    
    result
}"""
    ],
    "algorithm": [
"""/// Perform binary search on a sorted vector
pub fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let mut left = 0;
    let mut right = arr.len();
    
    while left < right {
        let mid = left + (right - left) / 2;
        
        match arr[mid].cmp(target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => left = mid + 1,
            std::cmp::Ordering::Greater => right = mid,
        }
    }
    
    None
}""",

"""/// Sort a vector using the QuickSort algorithm
pub fn quick_sort<T: Ord + Clone>(arr: &[T]) -> Vec<T> {
    if arr.len() <= 1 {
        return arr.to_vec();
    }
    
    let pivot = arr[arr.len() / 2].clone();
    
    let left: Vec<T> = arr.iter()
        .filter(|&x| x < &pivot)
        .cloned()
        .collect();
    
    let middle: Vec<T> = arr.iter()
        .filter(|&x| x == &pivot)
        .cloned()
        .collect();
    
    let right: Vec<T> = arr.iter()
        .filter(|&x| x > &pivot)
        .cloned()
        .collect();
    
    let mut result = quick_sort(&left);
    result.extend(middle);
    result.extend(quick_sort(&right));
    
    result
}""",

"""use std::collections::HashMap;

/// Calculate the nth Fibonacci number using memoization
pub fn fibonacci(n: usize, memo: &mut HashMap<usize, u64>) -> u64 {
    if let Some(&result) = memo.get(&n) {
        return result;
    }
    
    if n <= 2 {
        return 1;
    }
    
    let result = fibonacci(n-1, memo) + fibonacci(n-2, memo);
    memo.insert(n, result);
    
    result
}"""
    ]
}

# Примеры кода на C++
CPP_EXAMPLES = {
    "utility_function": [
"""#include <string>
#include <regex>

/**
 * Validate an email address using regex
 * 
 * @param email The email to validate
 * @return true if valid, false otherwise
 */
bool validateEmail(const std::string& email) {
    const std::regex pattern("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$");
    return std::regex_match(email, pattern);
}""",

"""#include <string>
#include <random>
#include <algorithm>

/**
 * Generate a random password with the specified length
 * 
 * @param length The password length
 * @return The generated password
 */
std::string generatePassword(size_t length = 12) {
    const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+";
    
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(0, charset.size() - 1);
    
    std::string password;
    password.reserve(length);
    
    for (size_t i = 0; i < length; ++i) {
        password += charset[distribution(generator)];
    }
    
    return password;
}""",

"""#include <vector>

/**
 * Flatten a nested vector
 * 
 * @param nestedVector The vector to flatten
 * @return The flattened vector
 */
template<typename T>
std::vector<T> flattenVector(const std::vector<std::vector<T>>& nestedVector) {
    std::vector<T> result;
    
    for (const auto& innerVector : nestedVector) {
        result.insert(result.end(), innerVector.begin(), innerVector.end());
    }
    
    return result;
}"""
    ],
    "algorithm": [
"""#include <vector>
#include <optional>

/**
 * Perform binary search on a sorted vector
 * 
 * @param arr The sorted vector
 * @param target The target value
 * @return Optional index of the target
 */
template<typename T>
std::optional<size_t> binarySearch(const std::vector<T>& arr, const T& target) {
    size_t left = 0;
    size_t right = arr.size();
    
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return std::nullopt;
}""",

"""#include <vector>
#include <algorithm>

/**
 * Sort a vector using the QuickSort algorithm
 * 
 * @param arr The vector to sort
 * @return Sorted vector
 */
template<typename T>
std::vector<T> quickSort(const std::vector<T>& arr) {
    if (arr.size() <= 1) {
        return arr;
    }
    
    const T pivot = arr[arr.size() / 2];
    
    std::vector<T> left, middle, right;
    
    for (const auto& element : arr) {
        if (element < pivot) {
            left.push_back(element);
        } else if (element == pivot) {
            middle.push_back(element);
        } else {
            right.push_back(element);
        }
    }
    
    auto result = quickSort(left);
    result.insert(result.end(), middle.begin(), middle.end());
    
    auto sortedRight = quickSort(right);
    result.insert(result.end(), sortedRight.begin(), sortedRight.end());
    
    return result;
}""",

"""#include <unordered_map>

/**
 * Calculate the nth Fibonacci number using memoization
 * 
 * @param n The position in the sequence
 * @param memo The memoization map
 * @return The nth Fibonacci number
 */
uint64_t fibonacci(unsigned int n, std::unordered_map<unsigned int, uint64_t>& memo) {
    auto it = memo.find(n);
    if (it != memo.end()) {
        return it->second;
    }
    
    if (n <= 2) {
        return 1;
    }
    
    uint64_t result = fibonacci(n-1, memo) + fibonacci(n-2, memo);
    memo[n] = result;
    
    return result;
}"""
    ]
}

# Примеры кода на TypeScript
TYPESCRIPT_EXAMPLES = {
    "utility_function": [
"""/**
 * Validate an email address using regex
 * @param email The email to validate
 * @returns true if valid, false otherwise
 */
function validateEmail(email: string): boolean {
  const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  return pattern.test(email);
}""",

"""/**
 * Generate a random password with the specified length
 * @param length The password length
 * @returns The generated password
 */
function generatePassword(length: number = 12): string {
  const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+~`|}{[]:;?><,./-=';
  let password = '';
  
  for (let i = 0; i < length; i++) {
    const randomIndex = Math.floor(Math.random() * charset.length);
    password += charset[randomIndex];
  }
  
  return password;
}""",

"""/**
 * Flatten a nested array
 * @param arr The array to flatten
 * @returns The flattened array
 */
function flattenArray<T>(arr: (T | T[])[]): T[] {
  return arr.reduce((flat: T[], toFlatten) => {
    return flat.concat(Array.isArray(toFlatten) ? flattenArray(toFlatten) : toFlatten);
  }, []);
}"""
    ],
    "algorithm": [
"""/**
 * Perform binary search on a sorted array
 * @param arr The sorted array
 * @param target The target value
 * @returns Index of the target or -1 if not found
 */
function binarySearch<T>(arr: T[], target: T): number {
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

"""/**
 * Sort an array using the QuickSort algorithm
 * @param arr The array to sort
 * @returns The sorted array
 */
function quickSort<T>(arr: T[]): T[] {
  if (arr.length <= 1) {
    return arr;
  }
  
  const pivot = arr[Math.floor(arr.length / 2)];
  const left = arr.filter(x => x < pivot);
  const middle = arr.filter(x => x === pivot);
  const right = arr.filter(x => x > pivot);
  
  return [...quickSort(left), ...middle, ...quickSort(right)];
}""",

"""/**
 * Calculate the nth Fibonacci number using memoization
 * @param n The position in the sequence
 * @param memo The memoization object
 * @returns The nth Fibonacci number
 */
function fibonacci(n: number, memo: Record<number, number> = {}): number {
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
"""/**
 * Hash a password using Web Crypto API
 * @param password The password to hash
 * @param salt Optional salt for the hash
 * @returns Promise resolving to hash data
 */
async function hashPassword(
  password: string,
  salt: Uint8Array | null = null
): Promise<{ salt: Uint8Array; hash: ArrayBuffer }> {
  const encoder = new TextEncoder();
  
  if (!salt) {
    salt = window.crypto.getRandomValues(new Uint8Array(16));
  }
  
  const passwordData = encoder.encode(password);
  
  const hashBuffer = await window.crypto.subtle.digest(
    'SHA-256',
    new Uint8Array([...salt, ...passwordData])
  );
  
  return {
    salt,
    hash: hashBuffer
  };
}

/**
 * Verify a password against a stored hash
 * @param password The password to verify
 * @param storedHash The stored hash data
 * @returns Promise resolving to a boolean
 */
async function verifyPassword(
  password: string,
  storedHash: { salt: Uint8Array; hash: ArrayBuffer }
): Promise<boolean> {
  const result = await hashPassword(password, storedHash.salt);
  
  // Compare the hashes
  const hashA = new Uint8Array(result.hash);
  const hashB = new Uint8Array(storedHash.hash);
  
  if (hashA.length !== hashB.length) {
    return false;
  }
  
  // Time-constant comparison
  let diff = 0;
  for (let i = 0; i < hashA.length; i++) {
    diff |= hashA[i] ^ hashB[i];
  }
  
  return diff === 0;
}""",

"""/**
 * Sanitize user input to prevent XSS attacks
 * @param input The user input to sanitize
 * @returns The sanitized input
 */
function sanitizeInput(input: string): string {
  const sanitized = input
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
  
  return sanitized;
}"""
    ]
}

# Экспортируем все примеры
ADDITIONAL_CODE_EXAMPLES = {
    "go": GO_EXAMPLES,
    "ruby": RUBY_EXAMPLES,
    "php": PHP_EXAMPLES,
    "rust": RUST_EXAMPLES,
    "cpp": CPP_EXAMPLES,
    "typescript": TYPESCRIPT_EXAMPLES
} 