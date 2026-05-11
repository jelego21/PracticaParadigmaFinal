# nodes creation
class Node:
    def __init__(self, label, children=None): #label variable interna nodo (lo guarda)
        self.label    = label
        self.children = children or []

    def is_terminal(self): 
        return len(self.children) == 0 #cuenta hijos

    def __repr__(self):
        if self.is_terminal():
            return self.label
        return f"{self.label}({', '.join(repr(c) for c in self.children)})" #objeto-string con label e hijos (usa bucle para mostrar hijos)

def tokenize(expr): #convierte minima expresion a lista de tokens (separa)
    tokens = []
    i = 0
    expr = expr.replace(' ', '')
    while i < len(expr):
        ch = expr[i] #caracter actual
        if ch in '()+-*/':
            tokens.append(ch)#sisi es un operador lo agrega
            i += 1
        elif ch.isalpha():
            j = i #cambio de indice para no afectar el original
            while j < len(expr) and (expr[j].isalpha() or expr[j].isdigit()):
                j += 1
            tokens.append(expr[i:j])#forma identificadores que empiezan desde letra y lo para hasta que termina con otro id
            i = j
        elif ch.isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(expr[i:j]) 
            i = j
        else:
            raise ValueError(f"Carácter inválido: '{ch}'")
    return tokens

class Parser: #interpeta tokens y construye el arbol
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos    = 0

    def peek(self): #lee la pos del token avanzando sin consumir
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected=None): #lee, consume y avanza (verifica error)
        tok = self.peek()
        if expected and tok != expected:
            raise SyntaxError(f"Se esperaba '{expected}', se encontró '{tok}'")
        self.pos += 1
        return tok
    
#FUNCIONES GRAMATICALES 

    def parse_E(self):
        # E → T ('+' T | '-' T)
        left = self.parse_T()
        while self.peek() in ('+', '-'):
            op    = self.consume()   
            right = self.parse_T()
            left  = Node('E', [left, Node(op), right]) #crea nodo E con hijos:izq, op y der
        return Node('E', [left])

    def parse_T(self):
        # T → F ('*' F | '/' F)
        left = self.parse_F()
        while self.peek() in ('*', '/'):
            op    = self.consume()
            right = self.parse_F()
            left  = Node('T', [left, Node(op), right])
        return Node('T', [left])

    def parse_F(self):
        # F → '(' E ')' | identifier | number
        # Los paréntesis se guardan como hijos para el árbol de análisis completo.
        tok = self.peek()
        if tok == '(':
            self.consume('(')
            e = self.parse_E()
            self.consume(')')
            return Node('F', [Node('('), e, Node(')')])
        elif tok is not None and tok not in ')+-*/':
            self.consume()
            return Node('F', [Node(tok)])
        else:
            raise SyntaxError(f"Token inesperado: '{tok}'")

    def parse(self):
        # Punto de entrada; lanza SyntaxError si sobran tokens al final.
        tree = self.parse_E()
        if self.pos != len(self.tokens):
            raise SyntaxError(f"Token inesperado al final: '{self.peek()}'")
        return tree

# Conjunto de no terminales de la gramática.
NT = {'E', 'T', 'F'}

def _surface_symbols(node):
    return [child.label for child in node.children] #retorna las equiquetas originales de los hijos para cada nodo 

def _collect_productions(node, reverse_children=False):
    # Recorre el árbol en pre-orden y registra cada producción.
    productions = [] #acumulación de reglas
    if node.is_terminal():
        return productions
    if node.label in NT:
        productions.append((node.label, _surface_symbols(node))) #escribe nodo actual y sus hijos en productions
    children = reversed(node.children) if reverse_children else node.children #  si reverse_children es True entonces derecha izq (invierte el orden)
    for child in children:
        productions.extend(_collect_productions(child, reverse_children))
    return productions

def _apply_derivation(productions, from_right=False):
    current = ['E']        # inicio del estado
    steps   = [current[:]] #"copia" del estado actual 

    for (nt, rhs) in productions:
        if from_right:
            idx = None
            for i, sym in enumerate(current):
                if sym == nt:
                    idx = i
        else:
            idx = None
            for i, sym in enumerate(current):
                if sym == nt:
                    idx = i
                    break

        if idx is None:
            continue  # NT ya expandido en un paso anterior, saltar.

        current = current[:idx] + rhs + current[idx + 1:]
        steps.append(current[:])

    return steps

def left_derivation(tree):
    productions = _collect_productions(tree, reverse_children=False)
    return _apply_derivation(productions, from_right=False)

def right_derivation(tree):
    productions = _collect_productions(tree, reverse_children=True)
    return _apply_derivation(productions, from_right=True)

def build_ast(node):
    if node.is_terminal():
        return Node(node.label)

    if node.label == 'F':
        if len(node.children) == 3 and node.children[0].label == '(':
            return build_ast(node.children[1])  # F → '(' E ')': eliminar paréntesis
        return build_ast(node.children[0])       # F → id | num

    if node.label in ('E', 'T'):
        if len(node.children) == 1:
            return build_ast(node.children[0])  # Producción unitaria: colapsar
        if len(node.children) == 3:
            # E → E op T / T → T op F: el operador es la raíz del nodo AST
            left  = build_ast(node.children[0])
            op    = node.children[1].label
            right = build_ast(node.children[2])
            return Node(op, [left, right])
        return build_ast(node.children[0])

    return Node(node.label, [build_ast(c) for c in node.children])

def layout_tree(node, depth=0, counter=None):
    if counter is None:
        counter = [0]  # Lista mutable para compartir el contador en recursión.
    node.depth = depth
    if not node.children:
        node.x = counter[0]
        counter[0] += 1
    else:
        for child in node.children:
            layout_tree(child, depth + 1, counter)
        node.x = sum(c.x for c in node.children) / len(node.children)
    node.y = -depth


def collect_nodes(node, result=None):
    # Recolecta todos los nodos en una lista plana (pre-orden).
    if result is None:
        result = []
    result.append(node)
    for c in node.children:
        collect_nodes(c, result)
    return result

# Función de conveniencia: tokeniza, parsea y retorna todo de una vez.
def process_expression(expr, side='left'):
    tokens   = tokenize(expr)
    tree     = Parser(tokens).parse()
    steps    = left_derivation(tree) if side == 'left' else right_derivation(tree)
    ast_tree = build_ast(tree)
    return tree, ast_tree, steps