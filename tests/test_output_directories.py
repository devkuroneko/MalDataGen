import unittest

from Engine.DataIO.DirectoryManager import DirectoryManager
from Engine.DataIO.DirectoryManager import OUTPUTS_ROOT


class OutputDirectoryTest(unittest.TestCase):

    def test_default_output_directory_is_timestamped_under_outputs(self):
        directory_manager = DirectoryManager()

        experiment_path = directory_manager._resolve_experiment_directory()

        self.assertEqual(experiment_path.parent, OUTPUTS_ROOT)
        self.assertRegex(experiment_path.name, r"^out_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

    def test_tmp_output_directory_is_redirected_to_outputs(self):
        directory_manager = DirectoryManager()

        experiment_path = directory_manager._resolve_experiment_directory("/tmp/maldatagen_run")

        self.assertEqual(experiment_path.parent, OUTPUTS_ROOT)
        self.assertRegex(experiment_path.name, r"^maldatagen_run_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

    def test_outputs_directory_without_date_gets_execution_date(self):
        directory_manager = DirectoryManager()

        experiment_path = directory_manager._resolve_experiment_directory("outputs/manual_run")

        self.assertEqual(experiment_path.parent, OUTPUTS_ROOT)
        self.assertRegex(experiment_path.name, r"^manual_run_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

    def test_outputs_directory_with_date_is_preserved(self):
        directory_manager = DirectoryManager()

        experiment_path = directory_manager._resolve_experiment_directory("outputs/manual_run_2026-07-10_12-30-00")

        self.assertEqual(experiment_path.parent, OUTPUTS_ROOT)
        self.assertEqual(experiment_path.name, "manual_run_2026-07-10_12-30-00")


if __name__ == "__main__":
    unittest.main()
