from __future__ import annotations

import ast
import operator as op

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
_UN = {ast.USub: op.neg, ast.UAdd: op.pos}


def safe_calculate(expr: str) -> float:
    """Evaluate arithmetic via the AST (never eval()). Raises ValueError on anything else."""
    expr = expr.replace(",", "").replace("$", "").replace("×", "*").replace("^", "**")
    expr = expr.replace(" x ", " * ")
    if len(expr) > 200:
        raise ValueError("expression too long")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            l, r = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(r) > 16:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](l, r)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    try:
        return ev(ast.parse(expr.strip(), mode="eval"))
    except ZeroDivisionError as e:
        raise ValueError("division by zero") from e
    except SyntaxError as e:
        raise ValueError("invalid expression") from e


def fmt_number(x: float) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else f"{x:,}"
