from __future__ import annotations

from typing import Callable, Dict, TypeVar


T = TypeVar("T")
U = TypeVar("U")


def dict_apply(x: Dict[str, T], func: Callable[[T], U]) -> Dict[str, U]:
    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
