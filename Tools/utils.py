"""Legacy compatibility wrapper for :mod:`Tools.Plot.utils`."""

import warnings

warnings.warn(
    "Tools.utils is deprecated; use Tools.Plot.utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

from Tools.Plot.utils import create_directory

__all__ = ["create_directory"]
