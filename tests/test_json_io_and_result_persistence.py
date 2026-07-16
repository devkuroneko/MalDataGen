import json
import math
import tempfile
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

import run_appclassnet_top200 as runner
from Engine.DataIO.JsonIO import JsonSerializationError
from Engine.DataIO.JsonIO import ResultArtifactCorruptionError
from Engine.DataIO.JsonIO import atomic_write_json
from Engine.DataIO.JsonIO import load_json_file
from Engine.DataIO.JsonIO import to_jsonable
from Engine.Evaluation.ExperimentProtocol import resolve_evaluation_protocol_plan


class Mode(Enum):
    VALUE = "value"


class Unknown:
    pass


def test_evaluation_protocol_plan_to_dict_is_json_serializable():
    plan = resolve_evaluation_protocol_plan(
        SimpleNamespace(evaluation_protocol="ts_tr", evaluation_mode="both", run_tr_tr=False)
    )

    payload = plan.to_dict()

    assert payload == {
        "protocol": "ts_tr",
        "run_tr_tr": False,
        "run_tr_ts": False,
        "run_ts_tr": True,
        "run_tr_plus_ts_tr": False,
        "required_result_keys": ["TS-TR"],
    }
    json.dumps(payload, allow_nan=False)


def test_to_jsonable_converts_common_project_types():
    payload = to_jsonable(
        {
            "path": Path("/tmp/example"),
            "enum": Mode.VALUE,
            "int": numpy.int64(3),
            "float": numpy.float32(1.5),
            "bool": numpy.bool_(True),
            "tuple": ("TS-TR",),
            "set": {"b", "a"},
            "array": numpy.asarray([1, 2, 3], dtype=numpy.int64),
        }
    )

    assert payload["path"] == "/tmp/example"
    assert payload["enum"] == "value"
    assert payload["int"] == 3
    assert payload["float"] == pytest.approx(1.5)
    assert payload["bool"] is True
    assert payload["tuple"] == ["TS-TR"]
    assert payload["set"] == ["a", "b"]
    assert payload["array"] == [1, 2, 3]


def test_unknown_object_reports_path_and_type():
    with pytest.raises(JsonSerializationError) as error:
        to_jsonable({"outer": {"bad": Unknown()}})

    assert error.value.object_path == "$.outer.bad"
    assert "Unknown" in error.value.object_type


def test_nan_is_rejected():
    with pytest.raises(JsonSerializationError):
        to_jsonable({"metric": math.nan})


def test_atomic_write_json_produces_valid_json():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "Results.json"

        atomic_write_json({"ok": True}, path)

        assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
        assert not path.with_name("Results.json.tmp").exists()


def test_failed_serialization_leaves_no_partial_file():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "Results.json"

        with pytest.raises(JsonSerializationError):
            atomic_write_json({"bad": Unknown()}, path)

        assert not path.exists()
        assert not path.with_name("Results.json.tmp").exists()


def test_failed_serialization_preserves_previous_valid_file():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "Results.json"
        atomic_write_json({"previous": True}, path)

        with pytest.raises(JsonSerializationError):
            atomic_write_json({"bad": Unknown()}, path)

        assert json.loads(path.read_text(encoding="utf-8")) == {"previous": True}


def test_corrupt_json_raises_corruption_error():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "Results.json"
        path.write_text("{", encoding="utf-8")

        with pytest.raises(ResultArtifactCorruptionError):
            load_json_file(path)


def test_corrupt_results_json_does_not_become_not_run():
    with tempfile.TemporaryDirectory() as directory:
        results_path = Path(directory) / "run" / "EvaluationResults" / "Results.json"
        results_path.parent.mkdir(parents=True)
        results_path.write_text("{", encoding="utf-8")

        with pytest.raises(ResultArtifactCorruptionError):
            runner.write_batches_metrics(
                [results_path],
                Path(directory) / "summary.json",
                SimpleNamespace(evaluation_protocol="ts_tr", evaluation_mode="ts_tr", run_tr_tr=False),
            )


def test_ts_tr_real_resample_registers_train_manifest_without_test_manifest_requirement():
    with tempfile.TemporaryDirectory() as directory:
        results_path = _write_results_fixture(Path(directory), include_ts_tr=True, include_tr_ts=False)

        _, payload = runner.write_batches_metrics(
            [results_path],
            Path(directory) / "summary.json",
            SimpleNamespace(evaluation_protocol="ts_tr", evaluation_mode="ts_tr", run_tr_tr=False),
        )

        assert payload["TS-TR"]["status"] == "completed"
        assert payload["TS-TR"]["synthetic_manifest"].endswith("synthetic_batches/train/manifest.json")
        assert payload["TR-TS"]["status"] == "not_applicable"
        assert payload["TR-TS"]["synthetic_manifest"] is None
        assert Path(directory, "summary.json").is_file()


def _write_results_fixture(root, *, include_ts_tr, include_tr_ts):
    run_dir = root / "dataset" / "campaign" / "combination_1"
    results_path = run_dir / "EvaluationResults" / "Results.json"
    results_path.parent.mkdir(parents=True)
    _write_manifest(run_dir / "DataGenerated" / "synthetic_batches" / "train" / "manifest.json", "train")
    payload = {
        "schema_version": "3.0",
        "run_id": "combination_1",
        "EvaluationProtocolPlan": resolve_evaluation_protocol_plan(
            SimpleNamespace(evaluation_protocol="ts_tr", evaluation_mode="ts_tr", run_tr_tr=False)
        ).to_dict(),
        "BatchClassifier": {"1-Fold": {}},
        "TR-TS": {"DecisionTreeSubset": {"1-Fold": {"status": "not_applicable", "Accuracy": None, "BalancedAccuracy": None}}},
        "TS-TR": {},
    }
    if include_ts_tr:
        payload["TS-TR"] = {
            "DecisionTreeSubset": {
                "1-Fold": {
                    "status": "completed",
                    "Accuracy": 0.9,
                    "BalancedAccuracy": 0.9,
                    "MacroF1": 0.9,
                    "WeightedF1": 0.9,
                }
            }
        }
        payload["BatchClassifier"]["1-Fold"]["TS-TR"] = {
            "classifier": "DecisionTreeSubset",
            "classifier_name": "DecisionTreeSubset",
            "effective_fit_rows": 10,
            "training_time_seconds": 0.01,
        }
    if include_tr_ts:
        payload["TR-TS"] = {
            "DecisionTreeSubset": {
                "1-Fold": {
                    "status": "completed",
                    "Accuracy": 0.9,
                    "BalancedAccuracy": 0.9,
                    "MacroF1": 0.9,
                    "WeightedF1": 0.9,
                }
            }
        }
    atomic_write_json(payload, results_path)
    return results_path


def _write_manifest(path, split):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        {
            "total_rows": 10,
            "num_classes": 10,
            "features": 20,
            "split": split,
            "batches_by_class": {},
        },
        path,
    )
