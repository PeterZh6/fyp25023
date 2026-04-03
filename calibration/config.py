"""Shared calibration filtering config."""

from __future__ import annotations

from typing import Callable, Iterable, TypeVar

T = TypeVar("T")

EXCLUDED_BINARIES = {
    "998-specrand_base.arm32-gcc81-O3",
    "998-998-specrand_base.arm32-gcc81-O3",
    "999-specrand_base.arm32-gcc81-O3",
}


def filter_binary_names(binary_names: Iterable[str]) -> list[str]:
    """Filter excluded binary names while preserving input order."""
    names = list(binary_names)
    kept = set(names) - EXCLUDED_BINARIES
    filtered = [name for name in names if name in kept]
    print(
        f"[INFO] Excluded {len(names) - len(filtered)} duplicate specrand variants"
    )
    return filtered


def filter_binary_entries(
    entries: Iterable[T],
    name_getter: Callable[[T], str],
) -> list[T]:
    """Filter entry list by binary name while preserving input order."""
    items = list(entries)
    names = [name_getter(item) for item in items]
    kept = set(names) - EXCLUDED_BINARIES
    filtered = [item for item in items if name_getter(item) in kept]
    print(
        f"[INFO] Excluded {len(items) - len(filtered)} duplicate specrand variants"
    )
    return filtered
