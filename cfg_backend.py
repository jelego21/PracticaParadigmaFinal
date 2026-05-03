
# Nodo del árbol: etiqueta + hijos. Sin hijos = terminal.
class Node:
    def __init__(self, label, children=None):
        self.label    = label
        self.children = children or []

    def is_terminal(self):
        return len(self.children) == 0

    def __repr__(self):
        if self.is_terminal():
            return self.label
        return f"{self.label}({', '.join(repr(c) for c in self.children)})"


# Convierte la expresión cruda en lista de tokens.
# "(5*x)+y" → ['(', '5', '*', 'x', ')', '+', 'y']
def tokenize(expr):
    tokens = []
    i = 0
    expr = expr.replace(' ', '')
    while i < len(expr):
        ch = expr[i]
        if ch in '()+-*/':
            tokens.append(ch)
            i += 1
        elif ch.isalpha():
            j = i
            while j < len(expr) and (expr[j].isalpha() or expr[j].isdigit() or expr[j] == '_'):
                j += 1
            tokens.append(expr[i:j])
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


# Parser recursivo descendente.
# Usa iteración en vez de recursión izquierda pura para evitar recursión infinita,
# pero el árbol generado respeta asociatividad izquierda.
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos    = 0

    def peek(self):
        # Token actual sin consumirlo.
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected=None):
        # Consume el token actual; lanza SyntaxError si no coincide con 'expected'.
        tok = self.peek()
        if expected and tok != expected:
            raise SyntaxError(f"Se esperaba '{expected}', se encontró '{tok}'")
        self.pos += 1
        return tok

    def parse_E(self):
        # E → T ('+' T | '-' T)*  — suma y resta (menor precedencia)
        left = self.parse_T()
        while self.peek() in ('+', '-'):
            op    = self.consume()
            right = self.parse_T()
            left  = Node('E', [left, Node(op), right])
        return Node('E', [left])

    def parse_T(self):
        # T → F ('*' F | '/' F)*  — multiplicación y división
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


# ── DERIVACIONES ──────────────────────────────
# Estrategia de dos fases:
#   1. Recolectar producciones (NT, RHS) recorriendo el árbol en el orden correcto.
#   2. Reproducirlas sobre la forma sentencial, guardando cada reemplazo como un paso.

def _surface_symbols(node):
    # Retorna las etiquetas de los hijos directos → equivale al RHS de la producción.
    return [child.label for child in node.children]


def _collect_productions(node, reverse_children=False):
    # Recorre el árbol en pre-orden y registra cada producción NT → RHS.
    # reverse_children=False → izquierda→derecha (derivación izquierda)
    # reverse_children=True  → derecha→izquierda (derivación derecha)
    productions = []
    if node.is_terminal():
        return productions
    if node.label in NT:
        productions.append((node.label, _surface_symbols(node)))
    children = reversed(node.children) if reverse_children else node.children
    for child in children:
        productions.extend(_collect_productions(child, reverse_children))
    return productions


def _apply_derivation(productions, from_right=False):
    # Por cada producción (NT, RHS): busca el NT en la forma sentencial actual
    # (primera ocurrencia si izquierda, última si derecha), lo reemplaza por RHS
    # y guarda el resultado como un nuevo paso.
    current = ['E']        # Símbolo de inicio
    steps   = [current[:]]

    for (nt, rhs) in productions:
        if from_right:
            # Última ocurrencia: recorrer todo, idx queda con el último hallazgo.
            idx = None
            for i, sym in enumerate(current):
                if sym == nt:
                    idx = i
        else:
            # Primera ocurrencia: parar en el primer hallazgo.
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
    # Genera los pasos expandiendo siempre el NT más a la izquierda.
    productions = _collect_productions(tree, reverse_children=False)
    return _apply_derivation(productions, from_right=False)


def right_derivation(tree):
    # Genera los pasos expandiendo siempre el NT más a la derecha.
    productions = _collect_productions(tree, reverse_children=True)
    return _apply_derivation(productions, from_right=True)


# ── AST ───────────────────────────────────────
# Simplifica el árbol de análisis:
#   - Colapsa producciones unitarias (E→T, T→F).
#   - Elimina paréntesis (solo sintaxis).
#   - Los operadores pasan a ser nodos raíz.

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


# ── LAYOUT ────────────────────────────────────
# Asigna coordenadas x/y a cada nodo para dibujarlo.
# Hojas: x enteras consecutivas. Internos: promedio x de sus hijos.
# y = -depth para que la raíz quede arriba.

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