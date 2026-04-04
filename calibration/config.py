"""Shared calibration filtering config."""

from __future__ import annotations

from typing import Callable, Iterable, TypeVar

T = TypeVar("T")

EXCLUDED_BINARIES = {
    "998-specrand_base.arm32-gcc81-O3",
    "998-998-specrand_base.arm32-gcc81-O3",
    "999-specrand_base.arm32-gcc81-O3",
}

_exclusion_logged = False


def filter_binary_names(binary_names: Iterable[str]) -> list[str]:
    """Filter excluded binary names while preserving input order."""
    global _exclusion_logged
    names = list(binary_names)
    excluded = [n for n in names if n in EXCLUDED_BINARIES]
    filtered = [n for n in names if n not in EXCLUDED_BINARIES]
    if excluded and not _exclusion_logged:
        print(f"Excluded {len(excluded)} binaries: {excluded}")
        _exclusion_logged = True
    return filtered


def filter_binary_entries(
    entries: Iterable[T],
    name_getter: Callable[[T], str],
) -> list[T]:
    """Filter entry list by binary name while preserving input order."""
    global _exclusion_logged
    items = list(entries)
    excluded = [
        name_getter(item) for item in items if name_getter(item) in EXCLUDED_BINARIES
    ]
    filtered = [
        item for item in items if name_getter(item) not in EXCLUDED_BINARIES
    ]
    if excluded and not _exclusion_logged:
        print(f"Excluded {len(excluded)} binaries: {excluded}")
        _exclusion_logged = True
    return filtered
