import importlib
import sys
import unittest


class LegacyPlotWrappersTest(unittest.TestCase):

    def _fresh_import(self, module_name):
        sys.modules.pop(module_name, None)
        with self.assertWarns(DeprecationWarning):
            return importlib.import_module(module_name)

    def test_tools_utils_wraps_canonical_plot_utils(self):
        legacy = self._fresh_import("Tools.utils")
        canonical = importlib.import_module("Tools.Plot.utils")

        self.assertIs(legacy.create_directory, canonical.create_directory)

    def test_plot_heatmap_wraps_canonical_plot_module(self):
        legacy = self._fresh_import("Tools.PlotHeatMap")
        canonical = importlib.import_module("Tools.Plot.PlotHeatMap")

        self.assertIs(legacy.HeatmapComparator, canonical.HeatmapComparator)


if __name__ == "__main__":
    unittest.main()
