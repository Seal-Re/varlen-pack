"""varlen-pack: collator-time padding-free sequence packing.

The BFD packer has no third-party dependencies.  The collator requires
PyTorch and is imported lazily so that ``bfd_pack`` remains usable in
torch-free environments.
"""

from __future__ import annotations

from .bfd import bfd_pack

__all__ = ["bfd_pack", "PackedDataCollator", "Segment"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name in ("PackedDataCollator", "Segment"):
        from .collator import PackedDataCollator, Segment

        return {"PackedDataCollator": PackedDataCollator, "Segment": Segment}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
