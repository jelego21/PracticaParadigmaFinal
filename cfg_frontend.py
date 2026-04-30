
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

# Importar toda la lógica del backend
from cfg_backend import (
    NT,
    tokenize,
    Parser,
    left_derivation,
    right_derivation,
    build_ast,
    layout_tree,
    collect_nodes,
    process_expression,
)


# ─────────────────────────────────────────────
#  CANVAS DE MATPLOTLIB EMBEBIDO EN QT
#
#  Widget reutilizable para dibujar cualquier árbol.
#    is_ast=False → paleta ámbar  (árbol de análisis completo)
#    is_ast=True  → paleta azul   (AST simplificado)
# ─────────────────────────────────────────────
class TreeCanvas(FigureCanvas):
    def __init__(self, parent=None, is_ast=False):
        self.fig = Figure(figsize=(7, 5), facecolor='white')
        self.ax  = self.fig.add_subplot(111)
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

        # Calcular posiciones x/y para cada nodo (viene del backend)
        layout_tree(root)
        nodes = collect_nodes(root)

        # Ajustar límites del gráfico con margen (padding)
        xs    = [n.x for n in nodes]
        ys    = [n.y for n in nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = max(0.6, (max_x - min_x) * 0.08)
        self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
        self.ax.set_ylim(min_y - 0.6,   max_y + 0.6)

        # Dibujar aristas (línea de cada padre a cada hijo)
        for node in nodes:
            for child in node.children:
                self.ax.plot(
                    [node.x, child.x], [node.y, child.y],
                    color='#B4B2A9', linewidth=1.2, zorder=1
                )

        # Dibujar nodos: círculo coloreado + etiqueta de texto
        for node in nodes:
            is_nt = node.label in NT   # ¿Es no terminal (E, T, F)?

            # Paleta según tipo de árbol y tipo de nodo
            if self.is_ast:
                fill, edge, text_color = (
                    ('#E6F1FB', '#185FA5', '#0C447C') if is_nt
                    else ('#EAF3DE', '#3B6D11', '#27500A')
                )
            else:
                fill, edge, text_color = (
                    ('#FAEEDA', '#BA7517', '#633806') if is_nt
                    else ('#EAF3DE', '#3B6D11', '#27500A')
                )

            self.ax.add_patch(plt.Circle(
                (node.x, node.y), 0.28,
                color=fill, ec=edge, linewidth=1.4, zorder=2
            ))
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
#  VENTANA PRINCIPAL
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generación de Árbol Sintáctico — Alexander Narváez")
        self.setMinimumSize(1000, 700)
        self._build_ui()

    def _build_ui(self):
        """Construye todos los widgets de la ventana."""
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

        # ── Gramática como referencia visual
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

        # ── Campo de entrada + opciones de derivación
        input_box = QGroupBox("Expresión de entrada")
        input_layout = QHBoxLayout(input_box)

        self.expr_input = QLineEdit()
        self.expr_input.setFont(QFont("Consolas", 12))
        self.expr_input.setPlaceholderText("Ej: (5*x)+y  o  4+(a-b)*x")
        self.expr_input.setText("(5*x)+y")
        self.expr_input.returnPressed.connect(self.generate)
        input_layout.addWidget(self.expr_input, 3)

        # Radio buttons: izquierda / derecha
        deriv_group = QButtonGroup(self)
        self.radio_left  = QRadioButton("Izquierda")
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

        # ── Botones de ejemplos rápidos
        ex_layout = QHBoxLayout()
        ex_layout.addWidget(QLabel("Ejemplos:"))
        for expr in ["(5*x)+y", "x*y+z", "4+(a-b)*x", "2*4+8", "a+b*c-d"]:
            b = QPushButton(expr)
            b.setFont(QFont("Consolas", 10))
            b.clicked.connect(lambda _, e=expr: self._set_expr(e))
            ex_layout.addWidget(b)
        ex_layout.addStretch()
        root_layout.addLayout(ex_layout)

        # ── Etiqueta de error
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
          1. Llama a process_expression() del backend.
          2. Formatea los pasos y los muestra en la pestaña de derivación.
          3. Dibuja el árbol de análisis y el AST en sus respectivos canvas.
        """
        self.error_label.setText("")
        raw  = self.expr_input.text().strip()
        side = 'left' if self.radio_left.isChecked() else 'right'

        # Delegar toda la lógica al backend
        try:
            parse_tree, ast_tree, steps = process_expression(raw, side)
        except ValueError as e:
            self.error_label.setText(f"Error de tokenización: {e}")
            return
        except SyntaxError as e:
            self.error_label.setText(f"Error sintáctico: {e}")
            return

        # Actualizar título de la pestaña de derivación
        if side == 'left':
            self.deriv_title.setText(
                "Derivación por la IZQUIERDA  "
                "(en cada paso se expande el NT más a la izquierda):"
            )
        else:
            self.deriv_title.setText(
                "Derivación por la DERECHA  "
                "(en cada paso se expande el NT más a la derecha):"
            )

        # Formatear los pasos como texto (formato del docente: E ⇒ ... ⇒ ...)
        lines = []
        for i, step in enumerate(steps):
            sym = ' '.join(step)
            lines.append(f"     {sym}" if i == 0 else f"  ⇒  {sym}")
        self.deriv_text.setPlainText("\n".join(lines))

        # Dibujar árboles en sus canvas
        self.parse_canvas.draw_tree(parse_tree)
        self.ast_canvas.draw_tree(ast_tree)

        # Mostrar pestaña de derivación al generar
        self.tabs.setCurrentIndex(0)


# ─────────────────────────────────────────────
#  MAIN — Punto de entrada
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
