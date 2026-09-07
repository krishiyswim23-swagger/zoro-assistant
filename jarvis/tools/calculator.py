"""Stage 4: a calculator tool that evaluates arithmetic without using eval().

Only a whitelist of AST nodes (numbers, +-*/, **, %, unary +/-, and
parentheses) is allowed, so this can't be turned into arbitrary code
execution the way a bare eval(expression) could.
"""

from __future__ import annotations

import ast
import math
import operator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


class CalculatorError(Exception):
    pass


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        args = [_eval_node(arg) for arg in node.args]
        return _FUNCTIONS[node.func.id](*args)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    raise CalculatorError(f"unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except (SyntaxError, CalculatorError, TypeError, ZeroDivisionError, ValueError) as exc:
        raise CalculatorError(f"couldn't evaluate '{expression}': {exc}") from exc
    return str(result)
