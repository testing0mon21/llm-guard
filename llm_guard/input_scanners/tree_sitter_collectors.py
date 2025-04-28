#!/usr/bin/env python3
"""
Коллекторы идентификаторов для различных языков программирования с использованием Tree-sitter.
"""

import logging
from typing import Set, Dict, List, Tuple, Optional, Any
from llm_guard.input_scanners.tree_sitter_base import BaseTreeSitterCollector, TREE_SITTER_AVAILABLE

logger = logging.getLogger(__name__)

class PythonTreeSitterCollector(BaseTreeSitterCollector):
    """Коллектор идентификаторов для Python с использованием Tree-sitter"""
    
    def __init__(self):
        super().__init__("python")
    
    def _get_language_keywords(self) -> Set[str]:
        """
        Возвращает множество ключевых слов Python.
        
        Returns:
            Set[str]: Множество ключевых слов
        """
        return {
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif',
            'else', 'except', 'False', 'finally', 'for', 'from', 'global', 'if', 'import',
            'in', 'is', 'lambda', 'None', 'nonlocal', 'not', 'or', 'pass', 'raise',
            'return', 'True', 'try', 'while', 'with', 'yield', 'print', 'len', 'range',
            'list', 'dict', 'set', 'tuple', 'int', 'float', 'str', 'bool', 'sum',
            'min', 'max', 'any', 'all', 'enumerate', 'zip', 'map', 'filter', 'sorted',
            'abs', 'chr', 'ord', 'open', 'input', 'super'
        }
    
    def _walk_tree(self, node: Any) -> None:
        """
        Рекурсивный обход дерева разбора для Python.
        
        Args:
            node: Узел дерева
        """
        if not node:
            return
        
        # Обработка различных типов узлов Python
        if node.type == "identifier":
            # Добавляем идентификатор в общий список
            identifier_text = node.text.decode('utf8')
            self.identifiers.add(identifier_text)
            
            # Определяем, является ли он функцией, классом или переменной
            parent = node.parent
            if parent:
                if parent.type == "function_definition" and parent.child_by_field_name("name") == node:
                    self.functions.add(identifier_text)
                elif parent.type == "class_definition" and parent.child_by_field_name("name") == node:
                    self.classes.add(identifier_text)
                elif parent.type in ["assignment", "variable_declaration"]:
                    self.variables.add(identifier_text)
        
        # Обработка импортов
        elif node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    self.imports.add(child.text.decode('utf8'))
        
        # Обработка строковых литералов
        elif node.type in ["string", "string_literal"]:
            self.string_literals.add(node.text.decode('utf8'))
        
        # Обработка числовых литералов
        elif node.type in ["integer", "float", "float_literal", "integer_literal"]:
            self.numeric_literals.add(node.text.decode('utf8'))
        
        # Рекурсивный обход всех дочерних узлов
        for child in node.children:
            self._walk_tree(child)

class JavaScriptTreeSitterCollector(BaseTreeSitterCollector):
    """Коллектор идентификаторов для JavaScript с использованием Tree-sitter"""
    
    def __init__(self):
        super().__init__("javascript")
    
    def _get_language_keywords(self) -> Set[str]:
        """
        Возвращает множество ключевых слов JavaScript.
        
        Returns:
            Set[str]: Множество ключевых слов
        """
        return {
            'await', 'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger',
            'default', 'delete', 'do', 'else', 'enum', 'export', 'extends', 'false',
            'finally', 'for', 'function', 'if', 'implements', 'import', 'in', 'instanceof',
            'interface', 'let', 'new', 'null', 'package', 'private', 'protected', 'public',
            'return', 'super', 'switch', 'static', 'this', 'throw', 'try', 'true',
            'typeof', 'var', 'void', 'while', 'with', 'yield', 'console', 'log', 'alert',
            'document', 'window', 'Array', 'Object', 'String', 'Number', 'Boolean',
            'Math', 'JSON', 'Promise', 'Map', 'Set', 'Date', 'setTimeout', 'setInterval'
        }
    
    def _walk_tree(self, node: Any) -> None:
        """
        Рекурсивный обход дерева разбора для JavaScript.
        
        Args:
            node: Узел дерева
        """
        if not node:
            return
        
        # Обработка различных типов узлов JavaScript
        if node.type == "identifier":
            # Добавляем идентификатор в общий список
            identifier_text = node.text.decode('utf8')
            self.identifiers.add(identifier_text)
            
            # Определяем, является ли он функцией, классом или переменной
            parent = node.parent
            if parent:
                if parent.type == "function_declaration" and parent.child_by_field_name("name") == node:
                    self.functions.add(identifier_text)
                elif parent.type == "class_declaration" and parent.child_by_field_name("name") == node:
                    self.classes.add(identifier_text)
                elif parent.type in ["variable_declarator", "assignment_expression"]:
                    self.variables.add(identifier_text)
        
        # Обработка импортов
        elif node.type == "import_declaration":
            for child in node.children:
                if child.type == "import_specifier":
                    for grandchild in child.children:
                        if grandchild.type == "identifier":
                            self.imports.add(grandchild.text.decode('utf8'))
        
        # Обработка строковых литералов
        elif node.type in ["string", "template_string"]:
            self.string_literals.add(node.text.decode('utf8'))
        
        # Обработка числовых литералов
        elif node.type in ["number", "integer", "float"]:
            self.numeric_literals.add(node.text.decode('utf8'))
        
        # Рекурсивный обход всех дочерних узлов
        for child in node.children:
            self._walk_tree(child)

class JavaTreeSitterCollector(BaseTreeSitterCollector):
    """Коллектор идентификаторов для Java с использованием Tree-sitter"""
    
    def __init__(self):
        super().__init__("java")
    
    def _get_language_keywords(self) -> Set[str]:
        """
        Возвращает множество ключевых слов Java.
        
        Returns:
            Set[str]: Множество ключевых слов
        """
        return {
            'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
            'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
            'extends', 'false', 'final', 'finally', 'float', 'for', 'goto', 'if',
            'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native',
            'new', 'null', 'package', 'private', 'protected', 'public', 'return',
            'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this',
            'throw', 'throws', 'transient', 'true', 'try', 'void', 'volatile', 'while',
            'String', 'Integer', 'Long', 'Double', 'Float', 'Boolean', 'System', 'out',
            'println', 'print', 'List', 'Map', 'Set', 'ArrayList', 'HashMap', 'HashSet',
            'Object', 'Exception'
        }
    
    def _walk_tree(self, node: Any) -> None:
        """
        Рекурсивный обход дерева разбора для Java.
        
        Args:
            node: Узел дерева
        """
        if not node:
            return
        
        # Обработка различных типов узлов Java
        if node.type == "identifier":
            # Добавляем идентификатор в общий список
            identifier_text = node.text.decode('utf8')
            self.identifiers.add(identifier_text)
            
            # Определяем, является ли он методом, классом или переменной
            parent = node.parent
            if parent:
                if parent.type == "method_declaration" and parent.child_by_field_name("name") == node:
                    self.functions.add(identifier_text)
                elif parent.type == "class_declaration" and parent.child_by_field_name("name") == node:
                    self.classes.add(identifier_text)
                elif parent.type in ["variable_declarator", "field_declaration"]:
                    self.variables.add(identifier_text)
        
        # Обработка импортов
        elif node.type == "import_declaration":
            for child in node.children:
                if child.type == "scoped_identifier" or child.type == "identifier":
                    self.imports.add(child.text.decode('utf8'))
        
        # Обработка строковых литералов
        elif node.type in ["string_literal"]:
            self.string_literals.add(node.text.decode('utf8'))
        
        # Обработка числовых литералов
        elif node.type in ["decimal_integer_literal", "hex_integer_literal", "floating_point_literal"]:
            self.numeric_literals.add(node.text.decode('utf8'))
        
        # Рекурсивный обход всех дочерних узлов
        for child in node.children:
            self._walk_tree(child)

# Словарь доступных коллекторов
TREE_SITTER_COLLECTORS = {
    "python": PythonTreeSitterCollector,
    "javascript": JavaScriptTreeSitterCollector,
    "java": JavaTreeSitterCollector
}

def get_collector_for_language(language: str) -> Optional[BaseTreeSitterCollector]:
    """
    Возвращает коллектор для заданного языка.
    
    Args:
        language: Название языка
        
    Returns:
        Optional[BaseTreeSitterCollector]: Коллектор для языка или None
    """
    if not TREE_SITTER_AVAILABLE:
        return None
    
    collector_class = TREE_SITTER_COLLECTORS.get(language)
    if collector_class:
        return collector_class()
    
    return None 