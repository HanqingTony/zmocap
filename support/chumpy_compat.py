"""Make the legacy ``chumpy`` package importable on modern Python / NumPy.

``chumpy`` (used by ``smplx`` to read MANO/SMPL ``.pkl`` models) predates
Python 3.11 and NumPy 2:

* it calls ``inspect.getargspec`` (removed in Python 3.11);
* it does ``from numpy import bool, int, float, ...`` (NumPy 2 removed those
  aliases).

Rather than editing the installed package, we provide the aliases chumpy
expects, import it, then drop the aliases again so the rest of ComfyUI keeps a
pristine NumPy. ``ensure()`` is idempotent and cheap.
"""
from __future__ import annotations

import builtins
import importlib
import inspect

__all__ = ["ensure"]

# Names chumpy imports from numpy; all of them are Python builtins.
_LEGACY_NAMES = {"bool": "bool", "int": "int", "float": "float", "complex": "complex",
                 "object": "object", "str": "str", "unicode": "str"}
_DONE = False


def ensure() -> bool:
    """Import ``chumpy`` with the removed APIs shimmed. False if unavailable."""
    global _DONE
    if _DONE:
        return True

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

    try:
        import numpy
    except ImportError:
        return False

    added = []
    for alias, builtin_name in _LEGACY_NAMES.items():
        if not hasattr(numpy, alias):
            setattr(numpy, alias, getattr(builtins, builtin_name))
            added.append(alias)

    try:
        importlib.import_module("chumpy")
    except ImportError:
        return False
    finally:
        for alias in added:
            try:
                delattr(numpy, alias)
            except AttributeError:
                pass

    _DONE = True
    return True
