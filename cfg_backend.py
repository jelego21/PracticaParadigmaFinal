
# ─────────────────────────────────────────────
#  NODO DEL ÁRBOL
#  Estructura de datos central: cada nodo tiene una etiqueta
#  y una lista de hijos. Sin hijos = terminal (hoja).
# ─────────────────────────────────────────────
class Node:
    def __init__(self, label, children=None):
        self.label    = label
        self.children = children or []

    def is_terminal(self):
        """Retorna True si el nodo es una hoja (sin hijos)."""
        return len(self.children) == 0

    def __repr__(self):
        if self.is_terminal():
            return self.label
        return f"{self.label}({', '.join(repr(c) for c in self.children)})"


# ─────────────────────────────────────────────
#  TOKENIZADOR
#  Convierte la cadena cruda en lista de tokens.
#  "(5*x)+y"  →  ['(', '5', '*', 'x', ')', '+', 'y']
# ─────────────────────────────────────────────
def tokenize(expr):
    """
    Divide la expresión en tokens individuales:
      - Operadores y paréntesis: un carácter cada uno.
      - Identificadores: secuencia de letras/dígitos/guion bajo.
      - Números: secuencia de dígitos.
    Lanza ValueError si encuentra un carácter no reconocido.
    """
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


# ─────────────────────────────────────────────
#  PARSER RECURSIVO DESCENDENTE
#
#  Implementa la gramática con precedencia de operadores:
#    E (suma/resta) < T (mult/div) < F (factor)
#
#  Se usa iteración (en vez de recursión izquierda pura)
#  para evitar recursión infinita, pero el árbol generado
#  respeta la asociatividad izquierda correctamente.
# ─────────────────────────────────────────────
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos    = 0       # Índice del token que se está leyendo

    def peek(self):
        """Devuelve el token actual sin consumirlo (None si se acabaron)."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected=None):
        """
        Consume y retorna el token actual.
        Si se indica 'expected', verifica que coincida; lanza SyntaxError si no.
        """
        tok = self.peek()
        if expected and tok != expected:
            raise SyntaxError(f"Se esperaba '{expected}', se encontró '{tok}'")
        self.pos += 1
        return tok

    def parse_E(self):
        """
        Regla: E → T ('+' T | '-' T)*
        Acumula a la izquierda para reflejar asociatividad izquierda:
          a+b+c  →  E( E( E(T(a)), +, T(b) ), +, T(c) )
        El resultado se envuelve en un nodo E para que la raíz sea siempre E.
        """
        left = self.parse_T()
        while self.peek() in ('+', '-'):
            op    = self.consume()
            right = self.parse_T()
            left  = Node('E', [left, Node(op), right])
        return Node('E', [left])

    def parse_T(self):
        """
        Regla: T → F ('*' F | '/' F)*
        Misma lógica de acumulación izquierda que parse_E.
        """
        left = self.parse_F()
        while self.peek() in ('*', '/'):
            op    = self.consume()
            right = self.parse_F()
            left  = Node('T', [left, Node(op), right])
        return Node('T', [left])

    def parse_F(self):
        """
        Regla: F → '(' E ')' | identifier | number
        Maneja subexpresiones entre paréntesis o valores atómicos.
        Los paréntesis se conservan como hijos para el árbol de análisis completo.
        """
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
        """
        Punto de entrada: parsea la expresión completa.
        Lanza SyntaxError si quedan tokens sin consumir al final.
        """
        tree = self.parse_E()
        if self.pos != len(self.tokens):
            raise SyntaxError(f"Token inesperado al final: '{self.peek()}'")
        return tree


# ─────────────────────────────────────────────
#  CONJUNTO DE NO TERMINALES
# ─────────────────────────────────────────────
NT = {'E', 'T', 'F'}


# ─────────────────────────────────────────────
#  DERIVACIONES — ESTRATEGIA DE DOS FASES
#
#  Fase 1: recolectar producciones en el orden correcto
#          recorriendo el árbol (izquierda o derecha).
#  Fase 2: reproducir cada producción sobre la forma
#          sentencial actual, guardando cada paso.
#
#  Una "producción" es el par (NT, RHS) donde:
#    NT  = símbolo no terminal que se expande
#    RHS = lista de etiquetas de sus hijos directos
# ─────────────────────────────────────────────

def _surface_symbols(node):
    """
    Retorna las etiquetas de los hijos directos de un nodo.
    Equivale al lado derecho (RHS) de la producción que ese nodo representa.
    Ejemplo: nodo E con hijos [E, +, T]  →  ['E', '+', 'T']
    """
    return [child.label for child in node.children]


def _collect_productions(node, reverse_children=False):
    """
    Recorre el árbol en pre-orden y registra cada producción NT → RHS.
    - reverse_children=False → orden izquierda→derecha  (derivación izquierda)
    - reverse_children=True  → orden derecha→izquierda  (derivación derecha)
    """
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
    """
    Reproduce los pasos de derivación sobre la forma sentencial.

    Por cada producción (NT, RHS):
      1. Busca la primera (izquierda) o última (derecha) ocurrencia de NT.
      2. La reemplaza por RHS.
      3. Guarda el nuevo estado como un paso.

    Retorna lista de listas: cada sublista es la forma sentencial en ese paso.
    """
    current = ['E']          # Símbolo de inicio de la gramática
    steps   = [current[:]]   # Paso 0: sólo 'E'

    for (nt, rhs) in productions:
        # Buscar el índice donde reemplazar
        if from_right:
            # Derivación derecha → última ocurrencia del NT
            idx = None
            for i, sym in enumerate(current):
                if sym == nt:
                    idx = i          # Se sobrescribe en cada hallazgo → queda el último
        else:
            # Derivación izquierda → primera ocurrencia del NT
            idx = None
            for i, sym in enumerate(current):
                if sym == nt:
                    idx = i
                    break

        if idx is None:
            continue   # NT ya fue expandido en un paso anterior, saltar

        # Reemplazar NT por su RHS en la posición encontrada
        current = current[:idx] + rhs + current[idx + 1:]
        steps.append(current[:])

    return steps


def left_derivation(tree):
    """
    Genera los pasos de derivación por la IZQUIERDA.
    Siempre expande el no terminal más a la izquierda en cada paso.
    Retorna lista de formas sentenciales (listas de símbolos).
    """
    productions = _collect_productions(tree, reverse_children=False)
    return _apply_derivation(productions, from_right=False)


def right_derivation(tree):
    """
    Genera los pasos de derivación por la DERECHA.
    Siempre expande el no terminal más a la derecha en cada paso.
    Retorna lista de formas sentenciales (listas de símbolos).
    """
    productions = _collect_productions(tree, reverse_children=True)
    return _apply_derivation(productions, from_right=True)


# ─────────────────────────────────────────────
#  CONSTRUCCIÓN DEL AST
#
#  Simplifica el árbol de análisis eliminando:
#    - Producciones unitarias (E→T, T→F): se colapsan
#    - Paréntesis (sólo sintaxis, sin semántica)
#  Los operadores pasan a ser nodos raíz de sus subárboles.
# ─────────────────────────────────────────────
def build_ast(node):
    """
    Construye el Árbol de Sintaxis Abstracta (AST).
    Reglas de simplificación:
      F → '(' E ')'  : eliminar paréntesis, conservar subárbol E
      F → id/num     : bajar directamente al terminal
      E/T con 1 hijo : colapsar (producción unitaria)
      E/T con 3 hijos: el operador se convierte en raíz del nodo
    """
    if node.is_terminal():
        return Node(node.label)

    if node.label == 'F':
        if len(node.children) == 3 and node.children[0].label == '(':
            # F → '(' E ')': los paréntesis son sólo sintaxis
            return build_ast(node.children[1])
        return build_ast(node.children[0])   # F → identifier | number

    if node.label in ('E', 'T'):
        if len(node.children) == 1:
            # Producción unitaria E→T o T→F: colapsar
            return build_ast(node.children[0])
        if len(node.children) == 3:
            # E → E op T  /  T → T op F: el operador es la raíz
            left  = build_ast(node.children[0])
            op    = node.children[1].label
            right = build_ast(node.children[2])
            return Node(op, [left, right])
        return build_ast(node.children[0])   # fallback

    return Node(node.label, [build_ast(c) for c in node.children])


# ─────────────────────────────────────────────
#  LAYOUT DEL ÁRBOL (coordenadas X / Y)
#
#  Algoritmo simple:
#    - Hojas: posiciones x enteras consecutivas (contador global).
#    - Nodos internos: promedio x de sus hijos (quedan centrados).
#    - Profundidad → y negativa (raíz arriba, hojas abajo).
# ─────────────────────────────────────────────
def layout_tree(node, depth=0, counter=None):
    """
    Asigna atributos .x e .y a cada nodo para poder dibujarlo.
    counter es una lista de un elemento para poder mutarla en recursión.
    """
    if counter is None:
        counter = [0]
    node.depth = depth
    if not node.children:
        node.x = counter[0]
        counter[0] += 1
    else:
        for child in node.children:
            layout_tree(child, depth + 1, counter)
        node.x = sum(c.x for c in node.children) / len(node.children)
    node.y = -depth   # negativo: la raíz (depth=0) queda en la parte superior


def collect_nodes(node, result=None):
    """Recolecta todos los nodos del árbol en una lista plana (pre-orden)."""
    if result is None:
        result = []
    result.append(node)
    for c in node.children:
        collect_nodes(c, result)
    return result


# ─────────────────────────────────────────────
#  FUNCIÓN DE CONVENIENCIA
#  Recibe una expresión en texto y retorna todo de una vez.
# ─────────────────────────────────────────────
def process_expression(expr, side='left'):
    """
    Tokeniza, parsea y genera derivación + AST para 'expr'.
    side: 'left' o 'right'
    Retorna: (parse_tree, ast_tree, steps) o lanza ValueError/SyntaxError.
    """
    tokens    = tokenize(expr)
    tree      = Parser(tokens).parse()
    steps     = left_derivation(tree) if side == 'left' else right_derivation(tree)
    ast_tree  = build_ast(tree)
    return tree, ast_tree, steps
