"""Textual graph IR: the contract between the Python frontend and the C++ compiler (docs/ir-spec.md).

- export() turns a traced graph into an IRFunction. Its text is canonical: dlc-opt prints it back unchanged.
- parse_module() / parse_function() read IR text, such as dlc-opt output.
- execute() runs an IRFunction in the NumPy reference interpreter; comparing a function's outputs before and after a
  pass is how the graph passes are translation-validated (tests/python/test_translation_validation.py).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .tensor import Tensor

# The C++ op registry (lib/ir/OpRegistry.cpp), in its order: name -> (operand count, -1 if variadic;
# {attribute: (kind, required)}). test_translation_validation.py checks it against `dlc-opt --print-op-registry`.
OPS: Dict[str, Tuple[int, Dict[str, Tuple[str, bool]]]] = {
    "add": (2, {}),
    "sub": (2, {}),
    "mul": (2, {}),
    "div": (2, {}),
    "relu": (1, {}),
    "gelu": (1, {}),
    "exp": (1, {}),
    "log": (1, {}),
    "matmul": (2, {"transpose_a": ("int", False), "transpose_b": ("int", False)}),
    "softmax": (1, {"axis": ("int", False)}),
    "log_softmax": (1, {"axis": ("int", False)}),
    "cross_entropy": (2, {}),
    "layernorm": (3, {"eps": ("float", False)}),
    "sum": (1, {"axes": ("int-list", False), "keepdims": ("int", False)}),
    "mean": (1, {"axes": ("int-list", False), "keepdims": ("int", False)}),
    "transpose": (1, {"perm": ("int-list", False)}),
    "reshape": (1, {"shape": ("int-list", True)}),
    "embedding": (2, {}),
    "constant": (0, {"value": ("float", True)}),
    "onehot": (1, {"num_classes": ("int", True)}),
    "tanh": (1, {}),
    "sqrt": (1, {}),
    "rsqrt": (1, {}),
    "cmp_gt": (2, {}),
    "select": (3, {}),
    "broadcast_to": (1, {"shape": ("int-list", True)}),
    "reduce_max": (1, {"axes": ("int-list", False), "keepdims": ("int", False)}),
    "scatter_add": (2, {"num_rows": ("int", True)}),
    "cast": (1, {"to": ("string", True)}),
    "return": (-1, {}),
}

ROLES = ("input", "param", "const", "state")
DTYPES = ("f32", "i32")

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.$]*")


class IRParseError(ValueError):
    """Malformed IR text; the message starts with "line:col:"."""


@dataclass(frozen=True)
class IRType:
    dtype: str
    shape: Tuple[int, ...]

    def __str__(self) -> str:
        return f"{self.dtype}[{','.join(str(d) for d in self.shape)}]"


@dataclass
class IRArg:
    name: str
    type: IRType
    role: str = "input"
    init: Optional[float] = None  # the value every element of a state argument starts from


RESULT_KINDS = ("value", "grad", "donates")


@dataclass
class IROp:
    name: str
    operands: List[str]
    result: Optional[str] = None  # None for 'return'
    type: Optional[IRType] = None
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRFunction:
    name: str
    args: List[IRArg]
    result_types: List[IRType]
    ops: List[IROp]
    # Values of the param and const arguments, as exported; they are not part of the text.
    bindings: Dict[str, np.ndarray] = field(default_factory=dict)
    # Per result: ("value", None), ("grad", arg name) or ("donates", arg name). Empty means all plain values.
    result_info: List[Tuple[str, Optional[str]]] = field(default_factory=list)

    def result_kinds(self) -> List[Tuple[str, Optional[str]]]:
        return self.result_info or [("value", None)] * len(self.result_types)

    @property
    def returns(self) -> List[str]:
        if not self.ops or self.ops[-1].name != "return":
            raise ValueError(f"function @{self.name} does not end in 'return'")
        return list(self.ops[-1].operands)

    def arg(self, name: str) -> IRArg:
        for a in self.args:
            if a.name == name:
                return a
        raise KeyError(f"@{self.name} has no argument %{name}")

    def to_text(self) -> str:
        """The function as the C++ printer writes it: results renumbered %0, %1, ... and attributes sorted."""
        names = _slot_names(self)

        def ref(value: str) -> str:
            return "%" + names.get(value, value)

        params = []
        for a in self.args:
            role = ""
            if a.role != "input" or a.init is not None:
                init = "" if a.init is None else f", init = {_format_float(a.init)}"
                role = f" {{{a.role}{init}}}"
            params.append(f"{ref(a.name)}: {a.type}{role}")
        kinds = self.result_kinds()
        annotated = any(kind != "value" for kind, _ in kinds)
        if len(self.result_types) == 1 and not annotated:
            results = str(self.result_types[0])
        else:
            items = []
            for t, (kind, arg) in zip(self.result_types, kinds):
                items.append(str(t) if kind == "value" else f"{t} {{{kind} = {ref(arg)}}}")
            results = "(" + ", ".join(items) + ")"
        lines = [f"func @{self.name}({', '.join(params)}) -> {results} {{"]
        for op in self.ops:
            if op.name == "return":
                lines.append("  return" + "".join((", " if i else " ") + ref(v) for i, v in enumerate(op.operands)))
                continue
            text = f"  {ref(op.result)} = {op.name}({', '.join(ref(v) for v in op.operands)})"
            if op.type is not None:
                text += f" : {op.type}"
            if op.attrs:
                text += " {" + ", ".join(f"{k} = {_format_attr(op.attrs[k])}" for k in sorted(op.attrs)) + "}"
            lines.append(text)
        lines.append("}")
        return "\n".join(lines) + "\n"


def module_text(funcs: Sequence[IRFunction]) -> str:
    """A module as `dlc-opt` prints it."""
    return "// Module: parsed_module\n\n" + "".join(f.to_text() + "\n" for f in funcs)


def _slot_names(func: IRFunction) -> Dict[str, str]:
    # Same rules as the C++ SlotTracker: valid, unique argument names are kept, others become argN; results are
    # numbered in program order.
    names: Dict[str, str] = {}
    taken = set()
    named = [False] * len(func.args)
    for i, a in enumerate(func.args):
        if _NAME.fullmatch(a.name) and a.name not in taken:
            taken.add(a.name)
            names[a.name] = a.name
            named[i] = True
    for i, a in enumerate(func.args):
        if named[i]:
            continue
        fallback, k = f"arg{i}", 1
        while fallback in taken:
            fallback, k = f"arg{i}_{k}", k + 1
        taken.add(fallback)
        names[a.name] = fallback
    for n, op in enumerate(o for o in func.ops if o.result is not None):
        names[op.result] = str(n)
    return names


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(float(value))


def _format_attr(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _format_float(float(value))
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in value):
            return "[" + ", ".join(str(int(v)) for v in value) + "]"
        return "[" + ", ".join(_format_float(float(v)) for v in value) + "]"
    raise TypeError(f"cannot print attribute value {value!r}")


# ----------------------------------------------------------------------------------------------------------------
# Export


def export(
    outputs: Union[Tensor, Sequence[Tensor]],
    name: str = "main",
    module: Any = None,
    names: Optional[Mapping[Tensor, str]] = None,
) -> IRFunction:
    """Lowers the graph that computes `outputs` to an IRFunction.

    Placeholders become input arguments and nn.Parameters param arguments; tensor-valued constants (masks, position
    indices) become const arguments; f32 scalar constants become `constant` ops. Arguments are named after `names`,
    then `module.named_parameters()`, then the tensors' own names, made valid and unique. The data of param and const
    arguments is kept in `bindings`.
    """
    roots = [outputs] if isinstance(outputs, Tensor) else list(outputs)
    if not roots:
        raise ValueError("export needs at least one output")
    given: Dict[int, str] = {}
    if module is not None:
        given.update({id(p): n for n, p in module.named_parameters()})
    if names:
        given.update({id(t): n for t, n in names.items()})

    order = _topological_order(roots)
    leaves: List[Tuple[Tensor, str]] = []
    for t in order:
        if not t.inputs:
            role = _leaf_role(t)
            if role is not None:
                leaves.append((t, role))
    leaves.sort(key=lambda item: ROLES.index(item[1]))  # stable: encounter order within each role

    value_names: Dict[int, str] = {}
    taken = set()
    args: List[IRArg] = []
    bindings: Dict[str, np.ndarray] = {}
    for t, role in leaves:
        arg_name = _unique_name(given.get(id(t), t.name), taken)
        value_names[id(t)] = arg_name
        args.append(IRArg(arg_name, _ir_type(t), role))
        if role in ("param", "const"):
            bindings[arg_name] = np.asarray(t.data)

    ops: List[IROp] = []
    for t in order:
        if id(t) in value_names:
            continue
        result = str(len(ops))
        value_names[id(t)] = result
        if not t.inputs:  # an f32 scalar constant
            ops.append(IROp("constant", [], result, _ir_type(t), {"value": float(np.asarray(t.data))}))
            continue
        if t.op not in OPS or t.op in ("constant", "return"):
            raise ValueError(f"cannot export op '{t.op}' ({t!r})")
        operands = [value_names[id(i)] for i in t.inputs]
        ops.append(IROp(t.op, operands, result, _ir_type(t), _export_attrs(t)))
    ops.append(IROp("return", [value_names[id(r)] for r in roots]))
    return IRFunction(name, args, [_ir_type(r) for r in roots], ops, bindings)


def _topological_order(roots: Sequence[Tensor]) -> List[Tensor]:
    order: List[Tensor] = []
    seen = set()
    for root in roots:
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for inp in reversed(node.inputs):
                if id(inp) not in seen:
                    stack.append((inp, False))
    return order


def _leaf_role(t: Tensor) -> Optional[str]:
    """The argument role of a leaf, or None for a leaf that becomes a `constant` op."""
    if t.op == "placeholder":
        return "input"
    if t.data is None:
        raise ValueError(f"leaf {t!r} is neither a placeholder nor holds data")
    if t.op == "parameter":
        return "param"
    if t.dtype == "f32" and np.asarray(t.data).ndim == 0:
        return None
    return "const"


def _ir_type(t: Tensor) -> IRType:
    if t.dtype not in DTYPES:
        raise ValueError(f"{t!r}: the IR has no element type '{t.dtype}' (only {', '.join(DTYPES)})")
    if any(d < 0 for d in t.shape):
        raise ValueError(f"{t!r} has a negative dimension")
    return IRType(t.dtype, tuple(int(d) for d in t.shape))


def _unique_name(base: str, taken: set) -> str:
    name = re.sub(r"[^A-Za-z0-9_.$]", "_", base or "v")
    if not re.match(r"[A-Za-z_]", name):
        name = "v" + name
    unique, k = name, 1
    while unique in taken:
        unique, k = f"{name}_{k}", k + 1
    taken.add(unique)
    return unique


def _export_attrs(t: Tensor) -> Dict[str, Any]:
    rank = len(t.inputs[0].shape) if t.inputs else 0
    if t.op in ("softmax", "log_softmax"):
        return {"axis": int(t.attrs.get("axis", -1))}
    if t.op == "layernorm":
        return {"eps": float(t.attrs.get("eps", 1e-5))}
    if t.op in ("sum", "mean"):
        axis = t.attrs.get("axis")
        if axis is None:
            axes = list(range(rank))
        else:
            axes = sorted(int(a) % rank for a in (axis if isinstance(axis, (list, tuple)) else [axis]))
        return {"axes": axes, "keepdims": int(bool(t.attrs.get("keepdims", False)))}
    if t.op == "transpose":
        perm = t.attrs.get("perm")
        return {"perm": [int(p) for p in (perm if perm is not None else reversed(range(rank)))]}
    if t.op == "reshape":
        return {"shape": [int(d) for d in t.shape]}
    if t.op == "onehot":
        return {"num_classes": int(t.attrs["num_classes"])}
    return {}


# ----------------------------------------------------------------------------------------------------------------
# Parsing

_TOKEN = re.compile(
    r"""(?P<space>\s+)
      | (?P<comment>//[^\n]*)
      | (?P<value>%[A-Za-z0-9_.$]+)
      | (?P<func>@[A-Za-z0-9_.$]+)
      | (?P<arrow>->)
      | (?P<float>-inf|-?\d+(?:\.\d*(?:[eE][+-]?\d+)?|[eE][+-]?\d+))
      | (?P<int>-?\d+)
      | (?P<string>"(?:\\.|[^"\\\n])*")
      | (?P<ident>[A-Za-z_][A-Za-z0-9_.]*)
      | (?P<punct>[(){}\[\],:=])""",
    re.VERBOSE,
)


@dataclass
class _Tok:
    kind: str
    text: str
    line: int
    col: int


def _tokenize(text: str) -> List[_Tok]:
    tokens: List[_Tok] = []
    pos, line, line_start = 0, 1, 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise IRParseError(f"{line}:{pos - line_start + 1}: unexpected character {text[pos]!r}")
        kind = m.lastgroup
        if kind not in ("space", "comment"):
            value = m.group()
            if kind == "ident" and value in ("nan", "inf"):
                kind = "float"
            tokens.append(_Tok(kind, value, line, pos - line_start + 1))
        newlines = m.group().count("\n")
        if newlines:
            line += newlines
            line_start = m.start() + m.group().rindex("\n") + 1
        pos = m.end()
    tokens.append(_Tok("eof", "", line, pos - line_start + 1))
    return tokens


class _Parser:
    def __init__(self, text: str):
        self.toks = _tokenize(text)
        self.i = 0

    def peek(self) -> _Tok:
        return self.toks[self.i]

    def error(self, tok: _Tok, message: str) -> IRParseError:
        return IRParseError(f"{tok.line}:{tok.col}: {message}")

    def accept(self, kind: str, text: Optional[str] = None) -> Optional[_Tok]:
        tok = self.peek()
        if tok.kind == kind and (text is None or tok.text == text):
            self.i += 1
            return tok
        return None

    def expect(self, kind: str, text: Optional[str] = None, what: str = "") -> _Tok:
        tok = self.accept(kind, text)
        if tok is None:
            got = self.peek()
            raise self.error(got, f"expected {what or text or kind}, got {got.text or 'end of input'!r}")
        return tok

    def module(self) -> List[IRFunction]:
        funcs: List[IRFunction] = []
        while self.peek().kind != "eof":
            self.expect("ident", "func", "'func'")
            name = self.expect("func", what="a function name")
            if any(f.name == name.text[1:] for f in funcs):
                raise self.error(name, f"redefinition of function '{name.text}'")
            funcs.append(self.function(name.text[1:]))
        return funcs

    def type(self) -> IRType:
        dtype = self.expect("ident", what="a type")
        if dtype.text not in DTYPES:
            raise self.error(dtype, f"unknown element type '{dtype.text}'")
        self.expect("punct", "[")
        dims: List[int] = []
        if not self.accept("punct", "]"):
            while True:
                dims.append(int(self.expect("int", what="a dimension").text))
                if not self.accept("punct", ","):
                    break
            self.expect("punct", "]")
        return IRType(dtype.text, tuple(dims))

    def function(self, name: str) -> IRFunction:
        defined = set()
        args: List[IRArg] = []
        self.expect("punct", "(")
        if not self.accept("punct", ")"):
            while True:
                tok = self.expect("value", what="an argument")
                if tok.text[1:] in defined:
                    raise self.error(tok, f"redefinition of '{tok.text}'")
                defined.add(tok.text[1:])
                self.expect("punct", ":")
                arg_type = self.type()
                role = "input"
                init = None
                if self.accept("punct", "{"):
                    role_tok = self.expect("ident", what="an argument role")
                    if role_tok.text not in ROLES:
                        raise self.error(role_tok, f"unknown argument role '{role_tok.text}'")
                    role = role_tok.text
                    if self.accept("punct", ","):
                        self.expect("ident", "init", "'init'")
                        self.expect("punct", "=")
                        number = self.peek()
                        if not (self.accept("int") or self.accept("float")):
                            raise self.error(number, f"expected a number, got {number.text!r}")
                        init = float(number.text)
                    self.expect("punct", "}")
                args.append(IRArg(tok.text[1:], arg_type, role, init))
                if not self.accept("punct", ","):
                    break
            self.expect("punct", ")")
        self.expect("arrow", what="'->'")
        results: List[IRType] = []
        info: List[Tuple[str, Optional[str]]] = []
        if self.accept("punct", "("):
            if not self.accept("punct", ")"):
                while True:
                    results.append(self.type())
                    info.append(self.result_annotation(args))
                    if not self.accept("punct", ","):
                        break
                self.expect("punct", ")")
        else:
            results.append(self.type())
            info.append(("value", None))

        ops: List[IROp] = []
        self.expect("punct", "{")
        while not self.accept("punct", "}"):
            if self.peek().kind == "eof":
                raise self.error(self.peek(), f"expected '}}' to close function '@{name}'")
            ops.append(self.statement(defined))
        annotated = any(kind != "value" for kind, _ in info)
        return IRFunction(name, args, results, ops, result_info=info if annotated else [])

    def result_annotation(self, args: List[IRArg]) -> Tuple[str, Optional[str]]:
        if not self.accept("punct", "{"):
            return ("value", None)
        kind = self.expect("ident", what="'grad' or 'donates'")
        if kind.text not in ("grad", "donates"):
            raise self.error(kind, f"unknown result annotation '{kind.text}'")
        self.expect("punct", "=")
        ref = self.expect("value", what="an argument")
        if ref.text[1:] not in {a.name for a in args}:
            raise self.error(ref, f"'{ref.text}' is not an argument")
        self.expect("punct", "}")
        return (kind.text, ref.text[1:])

    def values(self, defined: set) -> List[str]:
        names: List[str] = []
        while True:
            tok = self.expect("value", what="a value")
            if tok.text[1:] not in defined:
                raise self.error(tok, f"use of undefined value '{tok.text}'")
            names.append(tok.text[1:])
            if not self.accept("punct", ","):
                return names

    def statement(self, defined: set) -> IROp:
        if self.accept("ident", "return"):
            operands: List[str] = []
            if self.accept("punct", "("):
                if not self.accept("punct", ")"):
                    operands = self.values(defined)
                    self.expect("punct", ")")
            elif self.peek().kind == "value":
                operands = self.values(defined)
            return IROp("return", operands)

        result = self.expect("value", what="'%name = op(...)' or 'return'")
        self.expect("punct", "=")
        op_tok = self.expect("ident", what="an op name")
        if op_tok.text not in OPS or op_tok.text == "return":
            raise self.error(op_tok, f"unknown op '{op_tok.text}'")
        arity, specs = OPS[op_tok.text]
        self.expect("punct", "(")
        operands = []
        if not self.accept("punct", ")"):
            operands = self.values(defined)
            self.expect("punct", ")")
        if arity >= 0 and len(operands) != arity:
            raise self.error(op_tok, f"'{op_tok.text}' expects {arity} operand(s), got {len(operands)}")
        op_type = self.type() if self.accept("punct", ":") else None
        attrs: Dict[str, Any] = {}
        if self.accept("punct", "{"):
            if not self.accept("punct", "}"):
                while True:
                    key = self.expect("ident", what="an attribute name")
                    if key.text not in specs and "." not in key.text:
                        raise self.error(key, f"unknown attribute '{key.text}' for '{op_tok.text}'")
                    if key.text in attrs:
                        raise self.error(key, f"duplicate attribute '{key.text}'")
                    self.expect("punct", "=")
                    value = self.attr_value()
                    if key.text in specs and specs[key.text][0] == "float" and isinstance(value, int):
                        value = float(value)
                    attrs[key.text] = value
                    if not self.accept("punct", ","):
                        break
                self.expect("punct", "}")
        if result.text[1:] in defined:
            raise self.error(result, f"redefinition of '{result.text}'")
        defined.add(result.text[1:])
        return IROp(op_tok.text, operands, result.text[1:], op_type, attrs)

    def attr_value(self) -> Any:
        tok = self.peek()
        if self.accept("int"):
            return int(tok.text)
        if self.accept("float"):
            return float(tok.text)
        if self.accept("string"):
            return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), tok.text[1:-1])
        if self.accept("ident", "true") or self.accept("ident", "false"):
            return int(tok.text == "true")
        if self.accept("punct", "["):
            items: List[Union[int, float]] = []
            if not self.accept("punct", "]"):
                while True:
                    item = self.peek()
                    if self.accept("int"):
                        items.append(int(item.text))
                    elif self.accept("float"):
                        items.append(float(item.text))
                    else:
                        raise self.error(item, f"expected a number in the list, got {item.text!r}")
                    if not self.accept("punct", ","):
                        break
                self.expect("punct", "]")
            if any(isinstance(v, float) for v in items):
                return [float(v) for v in items]
            return items
        raise self.error(tok, f"expected an attribute value, got {tok.text!r}")


def parse_module(text: str) -> List[IRFunction]:
    return _Parser(text).module()


def parse_function(text: str) -> IRFunction:
    funcs = parse_module(text)
    if not funcs:
        raise IRParseError("input contains no function")
    return funcs[0]


# ----------------------------------------------------------------------------------------------------------------
# Execution


def to_graph(func: IRFunction) -> Tuple[Dict[str, Tensor], List[Tensor]]:
    """The function as a frontend graph: placeholders for its arguments and the tensors it returns."""
    values: Dict[str, Tensor] = {}
    for a in func.args:
        values[a.name] = Tensor(shape=a.type.shape, dtype=a.type.dtype, op="placeholder", name=a.name)
    args = dict(values)
    for op in func.ops:
        if op.name == "return":
            continue
        a = op.attrs
        attrs: Dict[str, Any] = {}
        data = None
        shape = op.type.shape if op.type is not None else None
        if op.name == "constant":
            if op.type is None:
                raise ValueError(f"constant %{op.result} has no type")
            data = np.full(op.type.shape, a["value"], dtype=np.float64)
        elif op.name in ("softmax", "log_softmax"):
            attrs["axis"] = a.get("axis", -1)
        elif op.name == "layernorm":
            attrs["eps"] = a.get("eps", 1e-5)
        elif op.name in ("sum", "mean", "reduce_max"):
            axes = a.get("axes")
            attrs["axis"] = None if axes is None else tuple(axes)
            attrs["keepdims"] = bool(a.get("keepdims", 0))
        elif op.name == "matmul":
            attrs["transpose_a"] = bool(a.get("transpose_a", 0))
            attrs["transpose_b"] = bool(a.get("transpose_b", 0))
        elif op.name == "broadcast_to":
            shape = tuple(a["shape"])
        elif op.name in ("scatter_add", "cast"):
            attrs.update(a)
        elif op.name == "transpose":
            attrs["perm"] = a.get("perm")
        elif op.name == "reshape":
            shape = tuple(a["shape"])
        elif op.name == "onehot":
            attrs["num_classes"] = a["num_classes"]
        if shape is None:
            raise ValueError(f"%{op.result} = {op.name}(...) has no type")
        dtype = op.type.dtype if op.type is not None else "f32"
        values[op.result] = Tensor(
            shape=shape,
            dtype=dtype,
            op=op.name,
            inputs=[values[v] for v in op.operands],
            data=data,
            attrs=attrs,
            name=op.result,
        )
    return args, [values[v] for v in func.returns]


def execute(func: IRFunction, feeds: Mapping[str, np.ndarray], dtype: Any = np.float64) -> List[np.ndarray]:
    """Runs `func` in the reference interpreter. `feeds` maps argument names to arrays; arguments it leaves out take
    their exported `bindings`. Returns the values of the returned results."""
    from .reference import ReferenceInterpreter

    args, results = to_graph(func)
    feed: Dict[Tensor, np.ndarray] = {}
    for name, t in args.items():
        arg = func.arg(name)
        if name in feeds:
            value = feeds[name]
        elif name in func.bindings:
            value = func.bindings[name]
        elif arg.init is not None:
            value = np.full(arg.type.shape, arg.init)
        else:
            raise KeyError(f"no value for argument %{name} of @{func.name}")
        value = np.asarray(value)
        if tuple(value.shape) != t.shape:
            raise ValueError(f"%{name} expects shape {t.shape}, got {value.shape}")
        feed[t] = value
    values = ReferenceInterpreter(dtype=dtype).forward(results, feed)
    return [np.asarray(values[r]) for r in results]


class StepRunner:
    """Runs a training-step function (autodiff + optimizer lowering) repeatedly in the reference interpreter.

    Param, const and state arguments live here between steps: params and consts start from the function's bindings
    (or `bindings`), states from their `init` value, and each donated result replaces its argument after a step.
    Calling the runner with the input arguments returns the results that are not donated (the loss first).
    """

    def __init__(self, func: IRFunction, bindings: Optional[Mapping[str, np.ndarray]] = None, dtype: Any = np.float64):
        self.func = func
        self.dtype = dtype
        self.values: Dict[str, np.ndarray] = {}
        given = {**func.bindings, **(bindings or {})}
        for a in func.args:
            if a.role in ("param", "const"):
                if a.name not in given:
                    raise KeyError(f"no value for {a.role} argument %{a.name}")
                self.values[a.name] = np.array(given[a.name], dtype=dtype)
            elif a.role == "state":
                self.values[a.name] = np.full(a.type.shape, 0.0 if a.init is None else a.init, dtype=dtype)

    def __call__(self, feeds: Mapping[str, np.ndarray]) -> List[np.ndarray]:
        outputs = execute(self.func, {**self.values, **feeds}, self.dtype)
        plain = []
        for value, (kind, arg) in zip(outputs, self.func.result_kinds()):
            if kind == "donates":
                self.values[arg] = value
            else:
                plain.append(value)
        return plain
