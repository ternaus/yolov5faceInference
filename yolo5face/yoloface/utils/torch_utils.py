from typing import Any


def copy_attr(a: Any, b: Any, include: tuple[str, ...] = (), exclude: tuple[str, ...] = ()) -> None:
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (include and k not in include) or k.startswith("_") or k in exclude:
            continue

        setattr(a, k, v)
