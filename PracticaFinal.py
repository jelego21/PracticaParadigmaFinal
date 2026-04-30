
import sys
import re
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton, QButtonGroup,
    QTextEdit, QTabWidget, QSplitter, QFrame, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPainter, QPen, QBrush, QColor, QFontMetrics

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#   NODO DEL ÁRBOL
# ─────────────────────────────────────────────
class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []

    def is_terminal(self):
        return len(self.children) == 0

    def __repr__(self):
        if self.is_terminal():
            return self.label
        return f"{self.label}({', '.join(repr(c) for c in self.children)})"


# ─────────────────────────────────────────────
#   TOKENIZADOR
# ─────────────────────────────────────────────
def tokenize(expr):
    """Convierte la expresión en una lista de tokens."""
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
#   PARSER RECURSIVO DESCENDENTE
#   Gramática:
#     E → E '+' T | E '-' T | T
#     T → T '*' F | T '/' F | F
#     F → '(' E ')' | identifier | number
# ─────────────────────────────────────────────
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected=None):
        tok = self.peek()
        if expected and tok != expected:
            raise SyntaxError(f"Se esperaba '{expected}', se encontró '{tok}'")
        self.pos += 1
        return tok

    def parse_E(self):
        """E → T (('+' | '-') T)*"""
        left = self.parse_T()
        while self.peek() in ('+', '-'):
            op = self.consume()
            right = self.parse_T()
            left = Node('E', [left, Node(op), right])
        return Node('E', [left])

    def parse_T(self):
        """T → F (('*' | '/') F)*"""
        left = self.parse_F()
        while self.peek() in ('*', '/'):
            op = self.consume()
            right = self.parse_F()
            left = Node('T', [left, Node(op), right])
        return Node('T', [left])

    def parse_F(self):
        """F → '(' E ')' | identifier | number"""
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
        tree = self.parse_E()
        if self.pos != len(self.tokens):
            raise SyntaxError(f"Token inesperado al final: '{self.peek()}'")
        return tree


# ─────────────────────────────────────────────
#   DERIVACIONES
# ─────────────────────────────────────────────
NT = {'E', 'T', 'F'}


def node_to_symbols(node):
    """Convierte un nodo en su lista de símbolos (terminales y no terminales)."""
    if node.is_terminal():
        return [node.label]
    if node.label in NT:
        return [node.label]
    return []


def get_flat_symbols(node):
    """Obtiene la forma sentencial actual expandiendo el árbol."""
    if node.is_terminal():
        return [node.label]
    if node.label in NT and len(node.children) == 0:
        return [node.label]
    return sum((get_flat_symbols(c) for c in node.children), [])


def left_derivation(tree):
    """Genera los pasos de derivación por la izquierda."""
    steps = [['E']]

    def expand(node, current):
        if node.is_terminal():
            return current
        if node.label not in NT:
            return current

        # Reemplaza la PRIMERA ocurrencia del no terminal
        rhs = sum((get_flat_symbols(c) for c in node.children), [])
        for i, sym in enumerate(current):
            if sym == node.label:
                new_current = current[:i] + rhs + current[i + 1:]
                steps.append(new_current[:])
                # Expandir hijos en orden izquierda → derecha
                result = new_current
                for child in node.children:
                    result = expand(child, result)
                return result
        return current

    expand(tree, ['E'])
    return steps


def right_derivation(tree):
    """Genera los pasos de derivación por la derecha."""
    steps = [['E']]

    def expand(node, current):
        if node.is_terminal():
            return current
        if node.label not in NT:
            return current

        rhs = sum((get_flat_symbols(c) for c in node.children), [])
        # Reemplaza la ÚLTIMA ocurrencia del no terminal
        last_idx = None
        for i, sym in enumerate(current):
            if sym == node.label:
                last_idx = i
        if last_idx is None:
            return current

        new_current = current[:last_idx] + rhs + current[last_idx + 1:]
        steps.append(new_current[:])
        result = new_current
        # Expandir hijos en orden derecha → izquierda
        for child in reversed(node.children):
            result = expand(child, result)
        return result

    expand(tree, ['E'])
    return steps


# ─────────────────────────────────────────────
#   CONSTRUCCIÓN DEL AST
# ─────────────────────────────────────────────
def build_ast(node):
    """Construye el AST eliminando nodos redundantes."""
    if node.is_terminal():
        return Node(node.label)

    if node.label == 'F':
        if len(node.children) == 3 and node.children[0].label == '(':
            # F → '(' E ')' → simplificar, el AST del subárbol E
            return build_ast(node.children[1])
        # F → identifier | number
        return build_ast(node.children[0])

    if node.label in ('E', 'T'):
        if len(node.children) == 1:
            return build_ast(node.children[0])
        if len(node.children) == 3:
            # E → E op T  o  T → T op F
            left = build_ast(node.children[0])
            op = node.children[1].label
            right = build_ast(node.children[2])
            return Node(op, [left, right])
        # Colapsar producciones unitarias anidadas
        return build_ast(node.children[0])

    return Node(node.label, [build_ast(c) for c in node.children])


# ─────────────────────────────────────────────
#   LAYOUT DEL ÁRBOL (ALGORITMO DE POSICIONAMIENTO)
# ─────────────────────────────────────────────
def layout_tree(node, depth=0, counter=None):
    """Asigna coordenadas x/y a cada nodo."""
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
    node.y = -depth


def collect_nodes(node, result=None):
    if result is None:
        result = []
    result.append(node)
    for c in node.children:
        collect_nodes(c, result)
    return result


# ─────────────────────────────────────────────
#   WIDGET DE MATPLOTLIB PARA EL ÁRBOL
# ─────────────────────────────────────────────
class TreeCanvas(FigureCanvas):
    def __init__(self, parent=None, is_ast=False):
        self.fig = Figure(figsize=(7, 5), facecolor='white')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.is_ast = is_ast
        self.setParent(parent)
        self.ax.set_axis_off()
        self.fig.tight_layout(pad=0.5)

    def draw_tree(self, root):
        self.ax.clear()
        self.ax.set_axis_off()
        if root is None:
            self.draw_idle()
            return

        layout_tree(root)
        nodes = collect_nodes(root)

        # Escala
        xs = [n.x for n in nodes]
        ys = [n.y for n in nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = max(0.5, (max_x - min_x) * 0.05)
        pad_y = 0.5

        self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
        self.ax.set_ylim(min_y - pad_y, max_y + pad_y)

        # Aristas
        for node in nodes:
            for child in node.children:
                self.ax.plot(
                    [node.x, child.x], [node.y, child.y],
                    color='#B4B2A9', linewidth=1.2, zorder=1
                )

        # Nodos
        for node in nodes:
            is_nt = node.label in NT
            if self.is_ast:
                fill = '#E6F1FB' if is_nt else '#EAF3DE'
                edge = '#185FA5' if is_nt else '#3B6D11'
                text_color = '#0C447C' if is_nt else '#27500A'
            else:
                fill = '#FAEEDA' if is_nt else '#EAF3DE'
                edge = '#BA7517' if is_nt else '#3B6D11'
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
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(8)
        root_layout.setContentsMargins(12, 12, 12, 12)

        # ── Título
        title = QLabel("Gramáticas Libres de Contexto — Árbol de Derivación y AST")
        title.setFont(QFont("Consolas", 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(title)

        # ── Gramática
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

        # ── Entrada
        input_box = QGroupBox("Expresión de entrada")
        input_layout = QHBoxLayout(input_box)

        self.expr_input = QLineEdit()
        self.expr_input.setFont(QFont("Consolas", 12))
        self.expr_input.setPlaceholderText("Ej: (5*x)+y  o  4+(a-b)*x")
        self.expr_input.setText("(5*x)+y")
        self.expr_input.returnPressed.connect(self.generate)
        input_layout.addWidget(self.expr_input, 3)

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

        # ── Ejemplos
        ex_layout = QHBoxLayout()
        ex_layout.addWidget(QLabel("Ejemplos:"))
        for expr in ["(5*x)+y", "x*y+z", "4+(a-b)*x", "2*4+8", "a+b*c-d"]:
            b = QPushButton(expr)
            b.setFont(QFont("Consolas", 10))
            b.clicked.connect(lambda _, e=expr: self._set_expr(e))
            ex_layout.addWidget(b)
        ex_layout.addStretch()
        root_layout.addLayout(ex_layout)

        # ── Error
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red; font-size: 12px;")
        root_layout.addWidget(self.error_label)

        # ── Tabs de resultados
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Consolas", 10))

        # Tab 1: Derivación
        deriv_widget = QWidget()
        deriv_layout = QVBoxLayout(deriv_widget)
        self.deriv_title = QLabel("Derivación por la izquierda:")
        self.deriv_title.setFont(QFont("Consolas", 11, QFont.Bold))
        deriv_layout.addWidget(self.deriv_title)
        self.deriv_text = QTextEdit()
        self.deriv_text.setFont(QFont("Consolas", 11))
        self.deriv_text.setReadOnly(True)
        deriv_layout.addWidget(self.deriv_text)
        self.tabs.addTab(deriv_widget, "Derivación")

        # Tab 2: Árbol de análisis
        self.parse_canvas = TreeCanvas(is_ast=False)
        scroll1 = QScrollArea()
        scroll1.setWidget(self.parse_canvas)
        scroll1.setWidgetResizable(True)
        self.tabs.addTab(scroll1, "Árbol de análisis sintáctico")

        # Tab 3: AST
        self.ast_canvas = TreeCanvas(is_ast=True)
        scroll2 = QScrollArea()
        scroll2.setWidget(self.ast_canvas)
        scroll2.setWidgetResizable(True)
        self.tabs.addTab(scroll2, "AST (Árbol de Sintaxis Abstracta)")

        root_layout.addWidget(self.tabs)

    def _set_expr(self, expr):
        self.expr_input.setText(expr)
        self.generate()

    def generate(self):
        self.error_label.setText("")
        raw = self.expr_input.text().strip()

        # Tokenizar
        try:
            tokens = tokenize(raw)
        except ValueError as e:
            self.error_label.setText(f"Error de tokenización: {e}")
            return

        if not tokens:
            self.error_label.setText("La expresión está vacía.")
            return

        # Parsear
        try:
            parser = Parser(tokens)
            tree = parser.parse()
        except SyntaxError as e:
            self.error_label.setText(f"Error sintáctico: {e}")
            return

        # Derivación
        side = 'left' if self.radio_left.isChecked() else 'right'
        if side == 'left':
            steps = left_derivation(tree)
            self.deriv_title.setText("Derivación por la IZQUIERDA:")
        else:
            steps = right_derivation(tree)
            self.deriv_title.setText("Derivación por la DERECHA:")

        lines = []
        for i, step in enumerate(steps):
            arrow = "  ⇒  " if i > 0 else "E  ⇒  " if len(steps) > 1 else ""
            sym = ' '.join(step)
            if i == 0:
                lines.append(sym)
            else:
                lines.append(f"  ⇒  {sym}")

        self.deriv_text.setPlainText("\n".join(lines))

        # Árbol de análisis
        self.parse_canvas.draw_tree(tree)

        # AST
        ast_root = build_ast(tree)
        self.ast_canvas.draw_tree(ast_root)

        self.tabs.setCurrentIndex(0)


# ─────────────────────────────────────────────
#   MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())