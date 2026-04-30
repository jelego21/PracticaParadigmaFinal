import nltk
import string
from nltk import CFG # context free grammar (let us define rules)
from nltk.parse import ChartParser #breaks down the rules (code-hierarchical format)

letters = list(string.ascii_lowercase)
numbers = list(string.digits)

letter_rules = " | ".join(f"'{l}'" for l in letters)
number_rules = " | ".join(f"'{n}'" for n in numbers)

grammar_rules = f"""
E -> E '+' T | E '-' T | T
T-> T '*' F | T '/' F | F
F-> '(' E ')' | {letter_rules} | {number_rules} """

grammar = CFG.fromstring(grammar_rules)
parser = ChartParser(grammar)

user_input = input("Ingrese la expresión (sin espacios): ")

# Separate all input symbols properly
for symbol in ['(', ')', '+', '-', '*', '/']:
    user_input = user_input.replace(symbol, f' {symbol} ')

sentence = user_input.split()

print("\nTokens:", sentence)

trees = list(parser.parse(sentence))
if not trees:
    print("\n La expresión NO es válida según la gramática.")
else:
    print("\n La expresión es válida.\n")
    print(" Estructura (árbol de derivación):\n")

    for tree in trees:
        print(tree)
        tree.draw()