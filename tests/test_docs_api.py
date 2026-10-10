"""Check that the docs and README only use public API that exists.

The top-level re-exports (``ewb.ERA5``, ``ewb.evaluation``, ``ewb.load_cases``,
``ewb.targets``, ...) were removed in favor of submodules (``ewb.inputs.ERA5``,
``ewb.evaluate.ExtremeWeatherBench``, ...). These tests make stale examples fail
CI instead of failing users.
"""

import ast
import importlib
import inspect
import pathlib
import re

import pytest

import extremeweatherbench as ewb

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC_FILES = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
DOC_FILES += sorted((REPO_ROOT / "docs").rglob("*.py"))

# ewb.a.b.c in code or prose, stopping at the first non-identifier character
EWB_REFERENCE = re.compile(r"(?<![\w.])ewb((?:\.[A-Za-z_]\w*)+)")
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _doc_id(path: pathlib.Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _python_snippets(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (first line number, source) for each Python snippet in a doc."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return [(1, text)]
    return [
        (_line(text, m.start()) + 1, m.group(1)) for m in PYTHON_BLOCK.finditer(text)
    ]


def _resolve(obj, dotted: str):
    """Follow a dotted attribute path from obj; return None if any part is missing."""
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


@pytest.mark.parametrize("doc", DOC_FILES, ids=_doc_id)
def test_docs_ewb_references_resolve(doc):
    """Every ``ewb.<...>`` reference, in code or prose, exists on the package."""
    text = doc.read_text(encoding="utf-8")
    missing = [
        f"{_doc_id(doc)}:{_line(text, m.start())}: ewb{m.group(1)}"
        for m in EWB_REFERENCE.finditer(text)
        if _resolve(ewb, m.group(1)[1:]) is None
    ]
    assert not missing, "Docs reference API that does not exist:\n" + "\n".join(missing)


@pytest.mark.parametrize("doc", DOC_FILES, ids=_doc_id)
def test_docs_python_snippets(doc):
    """Snippets parse, their EWB imports exist, and EWB calls use real kwargs."""
    errors = []
    for first_line, source in _python_snippets(doc):
        try:
            tree = ast.parse(source)
        except SyntaxError as err:
            errors.append(f"{first_line + (err.lineno or 1) - 1}: {err.msg}")
            continue

        # Names bound by `from extremeweatherbench[.mod] import name`
        bound = {"ewb": ewb}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "extremeweatherbench"
            ):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    obj = getattr(module, alias.name, None)
                    if obj is None:
                        errors.append(f"cannot import {alias.name} from {node.module}")
                    bound[alias.asname or alias.name] = obj

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = ast.unparse(node.func)
            root, _, rest = target.partition(".")
            func = bound.get(root)
            func = _resolve(func, rest) if func is not None and rest else func
            if not callable(func):
                continue
            try:
                params = inspect.signature(func).parameters
            except (TypeError, ValueError):
                continue
            if any(p.kind is p.VAR_KEYWORD for p in params.values()):
                continue
            errors += [
                f"{first_line + node.lineno - 1}: {target}({kw.arg}=...) "
                "is not a parameter"
                for kw in node.keywords
                if kw.arg is not None and kw.arg not in params
            ]
    assert not errors, f"{_doc_id(doc)}:\n" + "\n".join(errors)
