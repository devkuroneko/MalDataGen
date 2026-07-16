"""Legacy compatibility wrapper for :mod:`Tools.Plot.PlotHeatMap`."""

import warnings

warnings.warn(
    "Tools.PlotHeatMap is deprecated; use Tools.Plot.PlotHeatMap instead.",
    DeprecationWarning,
    stacklevel=2,
)

from Tools.Plot.PlotHeatMap import *  # noqa: F401,F403
