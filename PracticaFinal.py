
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton, QButtonGroup,
    QTextEdit, QTabWidget, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#   NODO DEL ÁRBOL
#   Cada nodo tiene una etiqueta (ej. 'E', '+', 'x')
#   y una lista de hijos. Nodos sin hijos = terminales (hojas).
# ─────────────────────────────────────────────
class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []

    def is_terminal(self):
        """Un nodo es terminal si no tiene hijos (es una hoja del árbol)."""
        return len(self.children) == 0

    def __repr__(self):
        if self.is_terminal():
            return self.label
        return f"{self.label}({', '.join(repr(c) for c in self.children)})"


# ─────────────────────────────────────────────
#   TOKENIZADOR
#   Convierte la cadena de entrada en una lista de tokens.
#   Ejemplo: "(5*x)+y" → ['(', '5', '*', 'x', ')', '+', 'y']
# ─────────────────────────────────────────────
def tokenize(expr):
    tokens = []
    i = 0
    expr = expr.replace(' ', '')
    while i < len(expr):
        ch = expr[i]
        if ch in '()+-*/':
            # Operadores y paréntesis → token de un solo carácter
            tokens.append(ch)
            i += 1
        elif ch.isalpha():
            # Identificador: letras seguidas opcionalmente de dígitos o '_'
            j = i
            while j < len(expr) and (expr[j].isalpha() or expr[j].isdigit() or expr[j] == '_'):
                j += 1
            tokens.append(expr[i:j])
            i = j
        elif ch.isdigit():
            # Número: secuencia de dígitos
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(expr[i:j])
            i = j
        else:
            raise ValueError(f"Carácter inválido: '{ch}'")
    return tokens


# ─────────────────────────────────────────────
#   PARSER RECURSIVO DESCENDENTE
#
#   Implementa la gramática con precedencia correcta:
#     E (suma/resta) < T (mult/div) < F (factor/átomo)
#
#   Se usa iteración en lugar de recursión izquierda pura
#   para evitar recursión infinita, pero el árbol producido
#   refleja la asociatividad izquierda correctamente.
# ─────────────────────────────────────────────
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0  # Índice del token actual

    def peek(self):
        """Devuelve el token actual sin consumirlo; None si se acabaron."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected=None):
        """Consume y retorna el token actual. Lanza error si no coincide con 'expected'."""
        tok = self.peek()
        if expected and tok != expected:
            raise SyntaxError(f"Se esperaba '{expected}', se encontró '{tok}'")
        self.pos += 1
        return tok

    def parse_E(self):
        """
        Regla: E → T ('+' T | '-' T)*
        Maneja suma y resta (menor precedencia).
        Construye árbol con asociatividad izquierda:
          a+b+c  →  E( E( E(T(a)), +, T(b) ), +, T(c) )
        El resultado final se envuelve en un nodo E para que
        la raíz del árbol siempre sea E.
        """
        left = self.parse_T()
        while self.peek() in ('+', '-'):
            op = self.consume()
            right = self.parse_T()
            # Acumular a la izquierda: el nuevo E tiene como hijo izquierdo
            # la expresión ya construida
            left = Node('E', [left, Node(op), right])
        return Node('E', [left])

    def parse_T(self):
        """
        Regla: T → F ('*' F | '/' F)*
        Maneja multiplicación y división (mayor precedencia que E).
        Misma lógica de acumulación izquierda que parse_E.
        """
        left = self.parse_F()
        while self.peek() in ('*', '/'):
            op = self.consume()
            right = self.parse_F()
            left = Node('T', [left, Node(op), right])
        return Node('T', [left])

    def parse_F(self):
        """
        Regla: F → '(' E ')' | identifier | number
        Maneja factores atómicos: subexpresiones entre paréntesis o valores.
        """
        tok = self.peek()
        if tok == '(':
            self.consume('(')
            e = self.parse_E()
            self.consume(')')
            # Conserva los paréntesis como hijos para el árbol de análisis completo
            return Node('F', [Node('('), e, Node(')')])
        elif tok is not None and tok not in ')+-*/':
            # Identificador o número
            self.consume()
            return Node('F', [Node(tok)])
        else:
            raise SyntaxError(f"Token inesperado: '{tok}'")

    def parse(self):
        """Punto de entrada: parsea la expresión completa."""
        tree = self.parse_E()
        if self.pos != len(self.tokens):
            raise SyntaxError(f"Token inesperado al final: '{self.peek()}'")
        return tree


# ─────────────────────────────────────────────
#   CONJUNTO DE NO TERMINALES
# ─────────────────────────────────────────────
NT = {'E', 'T', 'F'}


# ─────────────────────────────────────────────
#   OBTENER SÍMBOLOS SUPERFICIALES DE UN NODO
#
#   Dado un nodo, retorna la lista de etiquetas de sus hijos
#   directos. Esto equivale al lado derecho (RHS) de la
#   producción que ese nodo representa.
#
#   Ejemplo: nodo E con hijos [E, +, T]  →  ['E', '+', 'T']
# ─────────────────────────────────────────────
def surface_symbols(node):
    """
    Retorna las etiquetas de los hijos directos de un nodo.
    Equivale al RHS de la producción representada por ese nodo.
    """
    result = []
    for child in node.children:
        result.append(child.label)
    return result


# ─────────────────────────────────────────────
#   RECOLECCIÓN ORDENADA DE PRODUCCIONES
#
#   Estrategia para derivaciones correctas:
#
#   En lugar de intentar modificar la forma sentencial mientras
#   recorremos el árbol (lo cual genera inconsistencias), primero
#   recolectamos TODAS las producciones en el orden correcto
#   (izquierda o derecha), y luego las "reproducimos" una a una
#   sobre la forma sentencial.
#
#   Una "producción" aquí es el par (NT, RHS) donde:
#     NT  = símbolo no terminal que se expande
#     RHS = lista de símbolos que lo reemplazan (sus hijos directos)
# ─────────────────────────────────────────────

def collect_productions_left(node):
    """
    Recorre el árbol en pre-orden izquierda → derecha.
    Registra cada producción NT → RHS en ese orden.
    El orden resultante coincide exactamente con la derivación
    por la IZQUIERDA: siempre se expande el NT más a la izquierda.

    Ejemplo para (5*x)+y:
      E → E + T
      E → T          (el E izquierdo)
      T → F          (el T del E izquierdo)
      F → ( E )
      E → T          (el E dentro de paréntesis)
      T → T * F
      ...
    """
    productions = []
    if node.is_terminal():
        return productions
    if node.label in NT:
        rhs = surface_symbols(node)
        productions.append((node.label, rhs))
    # Recorrer hijos de izquierda a derecha (orden natural)
    for child in node.children:
        productions.extend(collect_productions_left(child))
    return productions


def collect_productions_right(node):
    """
    Recorre el árbol en pre-orden derecha → izquierda.
    Registra cada producción NT → RHS en ese orden.
    El orden resultante coincide con la derivación por la DERECHA:
    siempre se expande el NT más a la derecha.
    """
    productions = []
    if node.is_terminal():
        return productions
    if node.label in NT:
        rhs = surface_symbols(node)
        productions.append((node.label, rhs))
    # Recorrer hijos de derecha a izquierda
    for child in reversed(node.children):
        productions.extend(collect_productions_right(child))
    return productions


def apply_derivation(productions, from_right=False):
    """
    Reproduce los pasos de derivación sobre la forma sentencial.

    Recibe la lista ordenada de producciones [(NT, RHS), ...]
    y en cada paso:
      1. Busca la primera (izquierda) o última (derecha) ocurrencia
         del NT en la forma sentencial actual.
      2. La reemplaza por su RHS.
      3. Guarda el estado como un nuevo paso.

    Retorna una lista de listas: cada lista interior es la forma
    sentencial en ese paso de la derivación.

    Ejemplo de salida para (5*x)+y, derivación izquierda:
      ['E']
      ['E', '+', 'T']
      ['T', '+', 'T']
      ['F', '+', 'T']
      ['(', 'E', ')', '+', 'T']
      ...
      ['(', '5', '*', 'x', ')', '+', 'y']
    """
    current = ['E']          # La forma sentencial comienza con el símbolo de inicio
    steps = [current[:]]     # Guardamos el estado inicial como primer paso

    for (nt, rhs) in productions:
        if from_right:
            # Derivación derecha: buscar la ÚLTIMA ocurrencia del NT
            idx = None
            for i, sym in enumerate(current):
                if sym == nt:
                    idx = i          # Se sobrescribe en cada hallazgo → queda el último
            if idx is None:
                continue             # El NT ya fue expandido en un paso anterior, saltar
        else:
            # Derivación izquierda: buscar la PRIMERA ocurrencia del NT
            idx = None
            for i, sym in enumerate(current):
                if sym == nt:
                    idx = i
                    break            # Parar en la primera ocurrencia
            if idx is None:
                continue

        # Reemplazar el NT en la posición encontrada por su RHS
        current = current[:idx] + rhs + current[idx + 1:]
        steps.append(current[:])    # Guardar copia del nuevo estado

    return steps


def left_derivation(tree):
    """
    Genera todos los pasos de derivación por la IZQUIERDA.
    Siempre expande el no terminal más a la izquierda en cada paso.
    """
    productions = collect_productions_left(tree)
    return apply_derivation(productions, from_right=False)


def right_derivation(tree):
    """
    Genera todos los pasos de derivación por la DERECHA.
    Siempre expande el no terminal más a la derecha en cada paso.
    """
    productions = collect_productions_right(tree)
    return apply_derivation(productions, from_right=True)


# ─────────────────────────────────────────────
#   CONSTRUCCIÓN DEL AST
#
#   El AST (Árbol de Sintaxis Abstracta) simplifica el árbol
#   de análisis eliminando:
#     - Nodos intermedios redundantes (producciones unitarias)
#     - Paréntesis (sólo sintaxis, sin semántica)
#   Los operadores pasan a ser nodos raíz de sus subexpresiones.
# ─────────────────────────────────────────────
def build_ast(node):
    """
    Construye el AST a partir del árbol de análisis completo.

    Reglas de simplificación aplicadas:
      - F → '(' E ')' : eliminar paréntesis, conservar subárbol de E
      - F → id/num    : bajar directamente al terminal
      - E/T con 1 hijo: colapsar (producción unitaria, ej. E → T)
      - E/T con 3 hijos (izq op der): el operador es la raíz del nodo AST
    """
    if node.is_terminal():
        # Hoja: retornar como está
        return Node(node.label)

    if node.label == 'F':
        if len(node.children) == 3 and node.children[0].label == '(':
            # F → '(' E ')': los paréntesis son azúcar sintáctica,
            # en el AST sólo importa el subárbol de E
            return build_ast(node.children[1])
        # F → identifier | number: bajar al hijo terminal
        return build_ast(node.children[0])

    if node.label in ('E', 'T'):
        if len(node.children) == 1:
            # Producción unitaria E → T o T → F: colapsar, no aporta información
            return build_ast(node.children[0])
        if len(node.children) == 3:
            # E → E op T  o  T → T op F
            # En el AST el operador es la raíz, y los operandos sus hijos
            left  = build_ast(node.children[0])
            op    = node.children[1].label   # '+', '-', '*' o '/'
            right = build_ast(node.children[2])
            return Node(op, [left, right])
        # Fallback: colapsar al primer hijo
        return build_ast(node.children[0])

    # Nodo no reconocido: reconstruir recursivamente
    return Node(node.label, [build_ast(c) for c in node.children])


# ─────────────────────────────────────────────
#   LAYOUT DEL ÁRBOL (POSICIONAMIENTO X / Y)
#
#   Algoritmo simple de posicionamiento:
#     - Las hojas reciben posiciones x enteras consecutivas
#       (con un contador global compartido).
#     - Los nodos internos toman el promedio x de sus hijos
#       (así quedan centrados sobre ellos).
#     - La profundidad determina y (negativa → raíz arriba).
# ─────────────────────────────────────────────
def layout_tree(node, depth=0, counter=None):
    """Asigna coordenadas (x, y) a cada nodo para su visualización."""
    if counter is None:
        counter = [0]   # Lista de un elemento para poder modificarla en recursión
    node.depth = depth
    if not node.children:
        # Hoja: asignar posición horizontal única y avanzar el contador
        node.x = counter[0]
        counter[0] += 1
    else:
        # Nodo interno: posicionar hijos primero, luego centrar
        for child in node.children:
            layout_tree(child, depth + 1, counter)
        node.x = sum(c.x for c in node.children) / len(node.children)
    node.y = -depth   # y negativo: la raíz (depth=0) queda arriba


def collect_nodes(node, result=None):
    """Recolecta todos los nodos del árbol en una lista plana (pre-orden)."""
    if result is None:
        result = []
    result.append(node)
    for c in node.children:
        collect_nodes(c, result)
    return result


# ─────────────────────────────────────────────
#   CANVAS DE MATPLOTLIB EMBEBIDO EN QT
# ─────────────────────────────────────────────
class TreeCanvas(FigureCanvas):
    """
    Widget de Qt que embebe una figura de Matplotlib para
    renderizar el árbol de análisis o el AST.
      is_ast=False → paleta ámbar (árbol de análisis completo)
      is_ast=True  → paleta azul  (AST simplificado)
    """
    def __init__(self, parent=None, is_ast=False):
        self.fig = Figure(figsize=(7, 5), facecolor='white')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.is_ast = is_ast
        self.setParent(parent)
        self.ax.set_axis_off()
        self.fig.tight_layout(pad=0.5)

    def draw_tree(self, root):
        """Limpia el canvas y dibuja el árbol desde la raíz dada."""
        self.ax.clear()
        self.ax.set_axis_off()
        if root is None:
            self.draw_idle()
            return

        # Calcular posiciones x/y de todos los nodos
        layout_tree(root)
        nodes = collect_nodes(root)

        # Ajustar límites del gráfico con margen (padding)
        xs = [n.x for n in nodes]
        ys = [n.y for n in nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = max(0.6, (max_x - min_x) * 0.08)
        pad_y = 0.6
        self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
        self.ax.set_ylim(min_y - pad_y, max_y + pad_y)

        # Dibujar aristas (líneas de padre a hijo)
        for node in nodes:
            for child in node.children:
                self.ax.plot(
                    [node.x, child.x], [node.y, child.y],
                    color='#B4B2A9', linewidth=1.2, zorder=1
                )

        # Dibujar nodos: círculo coloreado + etiqueta
        for node in nodes:
            is_nt = node.label in NT   # ¿Es un no terminal (E, T, F)?

            # Paleta de colores según el tipo de árbol y de nodo
            if self.is_ast:
                fill       = '#E6F1FB' if is_nt else '#EAF3DE'
                edge       = '#185FA5' if is_nt else '#3B6D11'
                text_color = '#0C447C' if is_nt else '#27500A'
            else:
                fill       = '#FAEEDA' if is_nt else '#EAF3DE'
                edge       = '#BA7517' if is_nt else '#3B6D11'
                text_color = '#633806' if is_nt else '#27500A'

            circle = plt.Circle(
                (node.x, node.y), 0.28,
                color=fill, ec=edge, linewidth=1.4, zorder=2
            )
            self.ax.add_patch(circle)
            self.ax.text(
                node.x, node.y, node.label,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color=text_color, zorder=3,
                fontfamily='monospace'
            )

        self.fig.tight_layout(pad=0.3)
        self.draw_idle()


# ─────────────────────────────────────────────
#   VENTANA PRINCIPAL
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generación de Árbol Sintáctico — Alexander Narváez")
        self.setMinimumSize(1000, 700)
        self._build_ui()

    def _build_ui(self):
        """Construye toda la interfaz gráfica de la ventana."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(8)
        root_layout.setContentsMargins(12, 12, 12, 12)

        # ── Título de la aplicación
        title = QLabel("Gramáticas Libres de Contexto — Árbol de Derivación y AST")
        title.setFont(QFont("Consolas", 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(title)

        # ── Mostrar la gramática como referencia visual
        gram_box = QGroupBox("Gramática (expresión infija)")
        gram_layout = QVBoxLayout(gram_box)
        gram_text = QLabel(
            "E  →  E '+' T  |  E '-' T  |  T\n"
            "T  →  T '*' F  |  T '/' F  |  F\n"
            "F  →  '(' E ')'  |  identifier  |  number"
        )
        gram_text.setFont(QFont("Consolas", 11))
        gram_text.setStyleSheet("background:#f8f8f2; padding:8px; border-radius:4px;")
        gram_layout.addWidget(gram_text)
        root_layout.addWidget(gram_box)

        # ── Campo de entrada + selección de tipo de derivación
        input_box = QGroupBox("Expresión de entrada")
        input_layout = QHBoxLayout(input_box)

        self.expr_input = QLineEdit()
        self.expr_input.setFont(QFont("Consolas", 12))
        self.expr_input.setPlaceholderText("Ej: (5*x)+y  o  4+(a-b)*x")
        self.expr_input.setText("(5*x)+y")
        self.expr_input.returnPressed.connect(self.generate)
        input_layout.addWidget(self.expr_input, 3)

        # Radio buttons para elegir derivación izquierda o derecha
        deriv_group = QButtonGroup(self)
        self.radio_left = QRadioButton("Izquierda")
        self.radio_right = QRadioButton("Derecha")
        self.radio_left.setChecked(True)
        deriv_group.addButton(self.radio_left)
        deriv_group.addButton(self.radio_right)
        input_layout.addWidget(QLabel("Derivación:"))
        input_layout.addWidget(self.radio_left)
        input_layout.addWidget(self.radio_right)

        btn = QPushButton("Generar →")
        btn.setFont(QFont("Consolas", 11, QFont.Bold))
        btn.setFixedHeight(36)
        btn.clicked.connect(self.generate)
        input_layout.addWidget(btn)
        root_layout.addWidget(input_box)

        # ── Botones de ejemplos rápidos (los mismos del PDF del docente)
        ex_layout = QHBoxLayout()
        ex_layout.addWidget(QLabel("Ejemplos:"))
        for expr in ["(5*x)+y", "x*y+z", "4+(a-b)*x", "2*4+8", "a+b*c-d"]:
            b = QPushButton(expr)
            b.setFont(QFont("Consolas", 10))
            # lambda con valor por defecto para capturar 'expr' correctamente
            b.clicked.connect(lambda _, e=expr: self._set_expr(e))
            ex_layout.addWidget(b)
        ex_layout.addStretch()
        root_layout.addLayout(ex_layout)

        # ── Etiqueta de error (visible sólo cuando hay un problema)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red; font-size: 12px;")
        root_layout.addWidget(self.error_label)

        # ── Pestañas de resultados
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Consolas", 10))

        # Pestaña 1: Derivación paso a paso
        deriv_widget = QWidget()
        deriv_layout = QVBoxLayout(deriv_widget)
        self.deriv_title = QLabel("Derivación:")
        self.deriv_title.setFont(QFont("Consolas", 11, QFont.Bold))
        deriv_layout.addWidget(self.deriv_title)
        self.deriv_text = QTextEdit()
        self.deriv_text.setFont(QFont("Consolas", 12))
        self.deriv_text.setReadOnly(True)
        self.deriv_text.setStyleSheet("QTextEdit { background:#fafafa; padding:10px; }")
        deriv_layout.addWidget(self.deriv_text)
        self.tabs.addTab(deriv_widget, "📋  Derivación paso a paso")

        # Pestaña 2: Árbol de análisis sintáctico completo
        self.parse_canvas = TreeCanvas(is_ast=False)
        scroll1 = QScrollArea()
        scroll1.setWidget(self.parse_canvas)
        scroll1.setWidgetResizable(True)
        self.tabs.addTab(scroll1, "🌳  Árbol de análisis sintáctico")

        # Pestaña 3: AST simplificado
        self.ast_canvas = TreeCanvas(is_ast=True)
        scroll2 = QScrollArea()
        scroll2.setWidget(self.ast_canvas)
        scroll2.setWidgetResizable(True)
        self.tabs.addTab(scroll2, "✳️   AST (Árbol de Sintaxis Abstracta)")

        root_layout.addWidget(self.tabs)

    def _set_expr(self, expr):
        """Carga un ejemplo en el campo de texto y genera automáticamente."""
        self.expr_input.setText(expr)
        self.generate()

    def generate(self):
        """
        Acción principal al presionar 'Generar' o Enter:
          1. Tokeniza la expresión ingresada.
          2. La parsea y construye el árbol de análisis.
          3. Calcula la derivación (izquierda o derecha) paso a paso.
          4. Muestra cada paso en la pestaña de derivación.
          5. Dibuja el árbol de análisis y el AST en sus pestañas.
        """
        self.error_label.setText("")
        raw = self.expr_input.text().strip()

        # Paso 1: tokenizar la expresión cruda
        try:
            tokens = tokenize(raw)
        except ValueError as e:
            self.error_label.setText(f"Error de tokenización: {e}")
            return

        if not tokens:
            self.error_label.setText("La expresión está vacía.")
            return

        # Paso 2: parsear y construir el árbol de análisis sintáctico
        try:
            parser = Parser(tokens)
            tree = parser.parse()
        except SyntaxError as e:
            self.error_label.setText(f"Error sintáctico: {e}")
            return

        # Paso 3: calcular la derivación según la opción seleccionada
        side = 'left' if self.radio_left.isChecked() else 'right'
        if side == 'left':
            steps = left_derivation(tree)
            self.deriv_title.setText(
                "Derivación por la IZQUIERDA  "
                "(en cada paso se expande el NT más a la izquierda):"
            )
        else:
            steps = right_derivation(tree)
            self.deriv_title.setText(
                "Derivación por la DERECHA  "
                "(en cada paso se expande el NT más a la derecha):"
            )

        # Paso 4: formatear los pasos para mostrarlos
        # Formato igual al de las diapositivas del docente:
        #   E  ⇒  E + T  ⇒  T + T  ⇒  F + T  ⇒  ...
        lines = []
        for i, step in enumerate(steps):
            sym = ' '.join(step)
            if i == 0:
                lines.append(f"     {sym}")       # Primer paso: sin flecha
            else:
                lines.append(f"  ⇒  {sym}")       # Pasos siguientes: con flecha ⇒

        self.deriv_text.setPlainText("\n".join(lines))

        # Paso 5a: dibujar el árbol de análisis sintáctico completo
        self.parse_canvas.draw_tree(tree)

        # Paso 5b: construir el AST y dibujarlo
        ast_root = build_ast(tree)
        self.ast_canvas.draw_tree(ast_root)

        # Mostrar la pestaña de derivación al generar
        self.tabs.setCurrentIndex(0)


# ─────────────────────────────────────────────
#   MAIN — Punto de entrada de la aplicación
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')   # Estilo visual moderno y consistente entre OS
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())