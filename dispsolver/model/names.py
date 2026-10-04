"""Abaqus name semantics for the model's name-keyed registries.

Abaqus labels (sets, surfaces, materials, sections, amplitudes, parts,
instances, ...) are case-insensitive: ``*NSET, NSET=Left`` and a later
``*BOUNDARY`` line on ``LEFT`` or ``left`` refer to the same set. Every
registry used to be a plain ``dict``, so a deck that merely spelled a name
in another case lost the reference -- and most consumers then skipped the
BC/load/section silently (deck audit 2026-10-04: section, BC and load
references dropped this way in dozens of the Abaqus verification decks).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


class CaseInsensitiveDict(dict):
    """dict whose ``str`` keys match regardless of case.

    The key keeps the spelling it was first stored with (iteration, repr and
    output names are unchanged); only lookups ignore case. An exact match is
    tried first, so the common path costs one ordinary dict lookup; the
    case-folding scan runs only on a miss. The class keeps no state beyond
    the dict itself, so it pickles and deep-copies like a plain dict.
    """

    __slots__ = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.update(*args, **kwargs)

    def _stored_key(self, key: Any) -> Any:
        """The key actually stored for `key`, or `key` itself if none."""
        if dict.__contains__(self, key) or not isinstance(key, str):
            return key
        folded = key.upper()
        for k in dict.keys(self):
            if isinstance(k, str) and k.upper() == folded:
                return k
        return key

    def __getitem__(self, key: Any) -> Any:
        return dict.__getitem__(self, self._stored_key(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        dict.__setitem__(self, self._stored_key(key), value)

    def __delitem__(self, key: Any) -> None:
        dict.__delitem__(self, self._stored_key(key))

    def __contains__(self, key: Any) -> bool:
        return dict.__contains__(self, self._stored_key(key))

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        k = self._stored_key(key)
        return dict.get(self, k, default)

    def pop(self, key: Any, *default: Any) -> Any:
        return dict.pop(self, self._stored_key(key), *default)

    def setdefault(self, key: Any, default: Optional[Any] = None) -> Any:
        return dict.setdefault(self, self._stored_key(key), default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        for src in args:
            items: Iterable = src.items() if hasattr(src, "items") else src
            for k, v in items:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def copy(self) -> "CaseInsensitiveDict":
        return CaseInsensitiveDict(self)

    def __reduce__(self):
        return (CaseInsensitiveDict, (dict(self),))
