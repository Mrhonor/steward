"""Evaluate bounded Python calculations over a detached robot sensor packet."""

import ast
import json
import math
import operator
from collections.abc import Callable


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
            ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Is: operator.is_, ast.IsNot: operator.is_not}
_FUNCTIONS: dict[str, Callable] = {
    "abs": abs, "len": len, "sum": sum, "min": min, "max": max, "round": round,
    "sorted": sorted, "range": range, "list": list, "mean": lambda values: sum(values) / len(values),
}
_LIMIT = 12_000


class _Analysis:
    def __init__(self, packet: dict):
        self.values = {"sensor": packet}
        self.budget = 200_000

    def step(self, cost: int = 1) -> None:
        self.budget -= cost
        if self.budget < 0:
            raise ValueError("Python analysis exceeded its evaluation budget")

    def bind(self, target: ast.expr, value) -> None:
        if not isinstance(target, ast.Name) or target.id == "sensor" or target.id.startswith("_"):
            raise ValueError("Python analysis accepts assignments to local names only")
        self.values[target.id] = value

    def expression(self, node: ast.expr):
        self.step()
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, str, bool, type(None)):
            if isinstance(node.value, (int, float)) and (not math.isfinite(node.value) or abs(node.value) > 1e15):
                raise ValueError("numeric literal exceeds analysis bounds")
            return node.value
        if isinstance(node, ast.Name):
            return self.values[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.expression(value) for value in node.elts]
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise ValueError("Python analysis does not support dictionary unpacking")
            return {self.expression(key): self.expression(value)
                    for key, value in zip(node.keys, node.values, strict=True) if key is not None}
        if isinstance(node, ast.Subscript):
            value = self.expression(node.value)
            index = self.expression(node.slice)
            if isinstance(index, slice) and isinstance(value, (list, str, range)):
                self.step(len(range(*index.indices(len(value)))))
            return value[index]
        if isinstance(node, ast.Slice):
            return slice(*(self.expression(value) if value is not None else None
                           for value in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd, ast.Not):
            value = self.expression(node.operand)
            return {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}[type(node.op)](value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = self.expression(node.left), self.expression(node.right)
            if type(left) not in (int, float) or type(right) not in (int, float):
                raise ValueError("arithmetic requires two numbers")
            result = _BINARY[type(node.op)](left, right)
            if not math.isfinite(result) or abs(result) > 1e15:
                raise ValueError("numeric result exceeds analysis bounds")
            return result
        if isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
            left = self.expression(node.left)
            for op, value in zip(node.ops, node.comparators, strict=True):
                right = self.expression(value)
                if type(left) not in (int, float, str, bool, type(None)) or type(right) not in (int, float, str, bool, type(None)):
                    raise ValueError("comparison requires scalar values")
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            for value in node.values[:-1]:
                result = self.expression(value)
                if bool(result) == isinstance(node.op, ast.Or):
                    return result
            return self.expression(node.values[-1])
        if isinstance(node, ast.IfExp):
            return self.expression(node.body if self.expression(node.test) else node.orelse)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "keys") and not node.keywords):
            value = self.expression(node.func.value)
            if type(value) is not dict:
                raise ValueError("get and keys require a sensor dictionary")
            args = [self.expression(arg) for arg in node.args]
            if node.func.attr == "get":
                return value.get(*args)
            return list(value.keys(*args))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
            name = node.func.id
            args = [self.expression(arg) for arg in node.args]
            if name in ("sum", "min", "max", "sorted", "mean", "list"):
                if len(args) != 1 or not isinstance(args[0], (list, range, dict)) or len(args[0]) > _LIMIT:
                    raise ValueError(f"{name} requires one bounded numeric sequence")
                for value in args[0]:
                    self.step()
                    if name != "list" and type(value) not in ((int, float, str) if name == "sorted" else (int, float)):
                        raise ValueError(f"{name} requires numeric values")
            result = _FUNCTIONS[name](*args)
            if isinstance(result, range) and len(result) > _LIMIT:
                raise ValueError("range exceeds analysis bounds")
            return result
        if isinstance(node, ast.ListComp) and len(node.generators) == 1 and not node.generators[0].is_async:
            gen = node.generators[0]
            values = self.expression(gen.iter)
            if len(values) > _LIMIT:
                raise ValueError("comprehension exceeds analysis bounds")
            old = self.values.copy()
            result = []
            try:
                for value in values:
                    self.step()
                    self.bind(gen.target, value)
                    if all(self.expression(test) for test in gen.ifs):
                        result.append(self.expression(node.elt))
            finally:
                self.values = old
            return result
        raise ValueError(f"Python analysis does not support {type(node).__name__}")


def analyze_sensor(packet: dict, code: str) -> dict:
    """Run local assignments and expressions; return the final JSON expression."""
    if not code.strip() or len(code) > 4096:
        raise ValueError("provide between 1 and 4096 characters of Python")
    try:
        tree = ast.parse(code)
        analysis = _Analysis(packet)
        result = None
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                analysis.bind(stmt.targets[0], analysis.expression(stmt.value))
            elif isinstance(stmt, ast.Expr):
                result = analysis.expression(stmt.value)
            else:
                raise ValueError(f"Python analysis does not support {type(stmt).__name__}")
        encoder = json.JSONEncoder(allow_nan=False)
        size = 0
        for chunk in encoder.iterencode(result):
            size += len(chunk)
            if size > 4096:
                raise ValueError("analysis output exceeds 4096 characters")
        return {"value": result}
    except (SyntaxError, TypeError, LookupError, ArithmeticError, RecursionError) as exc:
        raise ValueError(f"Python analysis failed: {exc}") from exc
