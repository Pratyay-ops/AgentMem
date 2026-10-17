"""The no-raw-ids write guard: a memory may carry a handle, never an identifier.

The blast radius is the reason. If every scope holds a private clone of the same
structure, a remembered raw id resolves to real content in the WRONG scope — and
with shared recall that wrong scope is reachable in normal use, not just in theory.
So store human-addressable handles ("4.2 Safety") and re-resolve them per turn.

REJECT on write, not filter on read. A filtered read hides the bug; a rejected
write surfaces it while the code that produced the bad memory can still be fixed.
"""

from __future__ import annotations

import re

# A uuid, or a long hex/opaque token that looks like a database id. Deliberately
# broad: a false positive on write is a clear error to fix; a false negative is a
# memory that silently points at another tenant's record.
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_LONG_HEX = re.compile(r"\b[0-9a-fA-F]{24,}\b")


class RawIdInMemory(ValueError):
    """A memory carried a raw identifier where only a handle is allowed."""


def assert_no_raw_ids(text: str) -> None:
    """Raise if ``text`` looks like it embeds a raw id. Store handles instead."""
    value = text or ""
    if _UUID.search(value) or _LONG_HEX.search(value):
        raise RawIdInMemory(
            "refusing to store a memory containing a raw identifier; store a "
            "human handle (e.g. '4.2 Safety') and re-resolve it per turn"
        )
