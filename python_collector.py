#!/usr/bin/env python3
"""
Коллектор для языка Python на основе Tree-sitter.
Собирает информацию о функциях, классах, переменных и импортах.
"""

import logging
from tree_sitter_parser import TreeSitterParser, TreeSitterCollector

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PythonCollector(TreeSitterCollector):
    """
    Коллектор для языка Python.
    Использует Tree-sitter для сбора информации о коде на Python.
    """
    
    def __init__(self, parsers_path=None):
        """
        Инициализация коллектора для Python.
        
        Args:
            parsers_path (str, optional): Путь к директории с грамматиками.
                                         По умолчанию используется ~/.tree-sitter/parsers
        """
        parser = TreeSitterParser("python", parsers_path)
        super().__init__(parser)

    def collect(self, code):
        """
        Собирает информацию о коде на Python.
        
        Args:
            code (str): Исходный код для анализа
            
        Returns:
            dict: Словарь с информацией о коде, содержащий:
                - imports: список импортов
                - functions: список функций
                - classes: список классов
                - variables: список переменных
        """
        root_node = self.parser.get_root_node(code)
        
        result = {
            "imports": self._collect_imports(root_node, code),
            "functions": self._collect_functions(root_node, code),
            "classes": self._collect_classes(root_node, code),
            "variables": self._collect_variables(root_node, code)
        }
        
        return result
    
    def _collect_imports(self, root_node, code):
        """
        Собирает информацию об импортах.
        
        Args:
            root_node (Node): Корневой узел дерева синтаксического разбора
            code (str): Исходный код
            
        Returns:
            list: Список импортов
        """
        imports = []
        
        # Находим все узлы типа import_statement и import_from_statement
        import_nodes = self.parser.find_nodes_by_type(root_node, "import_statement")
        import_from_nodes = self.parser.find_nodes_by_type(root_node, "import_from_statement")
        
        # Обрабатываем import_statement
        for node in import_nodes:
            import_text = self.parser.get_node_text(node, code)
            imports.append({
                "type": "import",
                "text": import_text.strip(),
                "line": node.start_point[0] + 1
            })
        
        # Обрабатываем import_from_statement
        for node in import_from_nodes:
            import_text = self.parser.get_node_text(node, code)
            imports.append({
                "type": "from_import",
                "text": import_text.strip(),
                "line": node.start_point[0] + 1
            })
        
        return imports
    
    def _collect_functions(self, root_node, code):
        """
        Собирает информацию о функциях.
        
        Args:
            root_node (Node): Корневой узел дерева синтаксического разбора
            code (str): Исходный код
            
        Returns:
            list: Список функций
        """
        functions = []
        
        # Находим все узлы типа function_definition
        function_nodes = self.parser.find_nodes_by_type(root_node, "function_definition")
        
        for node in function_nodes:
            # Находим имя функции
            name_node = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break
            
            if name_node is None:
                continue
            
            # Получаем имя функции
            name = self.parser.get_node_text(name_node, code)
            
            # Находим параметры функции
            parameters = []
            params_node = None
            for child in node.children:
                if child.type == "parameters":
                    params_node = child
                    break
            
            if params_node:
                # Обрабатываем параметры
                for param_node in params_node.children:
                    if param_node.type == "identifier":
                        param_name = self.parser.get_node_text(param_node, code)
                        parameters.append(param_name)
            
            # Получаем тело функции
            body_node = None
            for child in node.children:
                if child.type == "block":
                    body_node = child
                    break
            
            body_text = ""
            if body_node:
                body_text = self.parser.get_node_text(body_node, code)
            
            # Формируем информацию о функции
            function_info = {
                "name": name,
                "parameters": parameters,
                "body": body_text,
                "line": node.start_point[0] + 1,
                "full_text": self.parser.get_node_text(node, code)
            }
            
            functions.append(function_info)
        
        return functions
    
    def _collect_classes(self, root_node, code):
        """
        Собирает информацию о классах.
        
        Args:
            root_node (Node): Корневой узел дерева синтаксического разбора
            code (str): Исходный код
            
        Returns:
            list: Список классов
        """
        classes = []
        
        # Находим все узлы типа class_definition
        class_nodes = self.parser.find_nodes_by_type(root_node, "class_definition")
        
        for node in class_nodes:
            # Находим имя класса
            name_node = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break
            
            if name_node is None:
                continue
            
            # Получаем имя класса
            name = self.parser.get_node_text(name_node, code)
            
            # Находим методы класса
            methods = []
            
            # Находим все функции внутри класса
            method_nodes = self.parser.find_nodes_by_type(node, "function_definition")
            
            for method_node in method_nodes:
                # Находим имя метода
                method_name_node = None
                for child in method_node.children:
                    if child.type == "identifier":
                        method_name_node = child
                        break
                
                if method_name_node is None:
                    continue
                
                # Получаем имя метода
                method_name = self.parser.get_node_text(method_name_node, code)
                
                methods.append({
                    "name": method_name,
                    "line": method_node.start_point[0] + 1,
                    "full_text": self.parser.get_node_text(method_node, code)
                })
            
            # Формируем информацию о классе
            class_info = {
                "name": name,
                "methods": methods,
                "line": node.start_point[0] + 1,
                "full_text": self.parser.get_node_text(node, code)
            }
            
            classes.append(class_info)
        
        return classes
    
    def _collect_variables(self, root_node, code):
        """
        Собирает информацию о переменных.
        
        Args:
            root_node (Node): Корневой узел дерева синтаксического разбора
            code (str): Исходный код
            
        Returns:
            list: Список переменных
        """
        variables = []
        
        # Находим все узлы типа assignment
        assignment_nodes = self.parser.find_nodes_by_type(root_node, "assignment")
        
        for node in assignment_nodes:
            # Получаем левую часть присваивания (идентификатор)
            left_node = None
            for child in node.children:
                if child.type == "identifier" or child.type == "pattern_list":
                    left_node = child
                    break
            
            if left_node is None:
                continue
            
            # Получаем имя переменной
            var_name = ""
            if left_node.type == "identifier":
                var_name = self.parser.get_node_text(left_node, code)
            elif left_node.type == "pattern_list":
                # Если слева несколько переменных (распаковка)
                for pattern_child in left_node.children:
                    if pattern_child.type == "identifier":
                        if var_name:
                            var_name += ", "
                        var_name += self.parser.get_node_text(pattern_child, code)
            
            # Получаем правую часть присваивания (выражение)
            right_node = None
            for child in node.children:
                if child.type != "identifier" and child.type != "=" and child.type != "pattern_list":
                    right_node = child
                    break
            
            value = ""
            if right_node:
                value = self.parser.get_node_text(right_node, code)
            
            # Формируем информацию о переменной
            variable_info = {
                "name": var_name,
                "value": value,
                "line": node.start_point[0] + 1,
                "full_text": self.parser.get_node_text(node, code)
            }
            
            variables.append(variable_info)
        
        return variables


if __name__ == "__main__":
    # Пример использования
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python python_collector.py <файл.py>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        with open(filename, 'r') as f:
            code = f.read()
        
        collector = PythonCollector()
        result = collector.collect(code)
        
        print("\nИмпорты:")
        for imp in result["imports"]:
            print(f"  {imp['text']} (строка {imp['line']})")
        
        print("\nФункции:")
        for func in result["functions"]:
            print(f"  {func['name']}({', '.join(func['parameters'])}) (строка {func['line']})")
        
        print("\nКлассы:")
        for cls in result["classes"]:
            print(f"  {cls['name']} (строка {cls['line']})")
            for method in cls["methods"]:
                print(f"    - {method['name']} (строка {method['line']})")
        
        print("\nПеременные:")
        for var in result["variables"]:
            print(f"  {var['name']} = {var['value']} (строка {var['line']})")
    
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1) 