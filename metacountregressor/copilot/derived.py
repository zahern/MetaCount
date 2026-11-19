"""Safe arithmetic expressions for derived columns (e.g. exposure offsets).

Only column names, numeric constants, arithmetic and a small function
whitelist are allowed.  No attribute access, indexing, imports, lambdas or
arbitrary calls.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

import numpy as np
import pandas as pd


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "log": np.log,
    "log1p": np.log1p,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "abs": np.abs,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
}
_CONSTANTS = {"pi": np.pi, "e": np.e}


def evaluate_expression(expression: str, df: pd.DataFrame) -> np.ndarray:
    """Evaluate a whitelisted arithmetic expression against dataframe columns."""
    text = str(expression).strip()
    if not text:
        raise ValueError("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc.msg}") from exc

    def col(name: str) -> np.ndarray:
        if name not in df.columns:
            raise ValueError(f"unknown column {name!r}")
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"column {name!r} contains non-finite values")
        return values

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(
                node.value, bool
            ):
                return float(node.value)
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.Name):
            if node.id in df.columns:
                return col(node.id)
            if node.id in _CONSTANTS:
                return float(_CONSTANTS[node.id])
            if node.id in _FUNCTIONS:
                return _FUNCTIONS[node.id]
            raise ValueError(
                f"unknown name {node.id!r}; use a column or a supported function"
            )
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError("operator not allowed")
            return op(ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError("unary operator not allowed")
            return op(ev(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only direct function calls are allowed")
            fn = _FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ValueError(f"function {node.func.id!r} is not allowed")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
            return fn(*[ev(arg) for arg in node.args])
        raise ValueError(
            f"expression element not allowed: {type(node).__name__}"
        )

    result = ev(tree)
    arr = np.asarray(result, dtype=float)
    if arr.ndim == 0:
        arr = np.full(len(df), float(arr), dtype=float)
    if arr.shape[0] != len(df):
        raise ValueError("expression did not produce one value per row")
    if not np.all(np.isfinite(arr)):
        raise ValueError("expression produced non-finite values")
    return arr


def is_log_offset(expression: str) -> bool:
    low = str(expression).lower()
    return "log(" in low or "log1p(" in low or "log10(" in low
