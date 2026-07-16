import math
from types import SimpleNamespace

import pytest

from Engine.Arguments.Arguments import _normalize_evaluation_protocol_arguments
from Engine.DataIO.SyntheticQualityAudit import assert_real_resample_control_metrics
from Engine.Evaluation.ExperimentProtocol import canonical_result_key
from Engine.Evaluation.ExperimentProtocol import legacy_evaluation_for_plan
from Engine.Evaluation.ExperimentProtocol import normalize_results_keys
from Engine.Evaluation.ExperimentProtocol import resolve_evaluation_protocol_plan
from Engine.Evaluation.ExperimentProtocol import selected_protocol_ids


def _args(protocol="legacy", mode="both", run_tr_tr=False):
    return SimpleNamespace(evaluation_protocol=protocol, evaluation_mode=mode, run_tr_tr=run_tr_tr)


def _plan(protocol="legacy", mode="both", run_tr_tr=False):
    return resolve_evaluation_protocol_plan(_args(protocol, mode, run_tr_tr))


def _metric_block(accuracy=0.9, balanced=0.9, status="completed"):
    return {
        "status": status,
        "reason": None,
        "Accuracy": accuracy,
        "BalancedAccuracy": balanced,
        "MacroF1": 0.9,
        "WeightedF1": 0.9,
    }


def _not_applicable_block(status="not_applicable"):
    return {
        "status": status,
        "reason": "evaluation_protocol=inactive",
        "Accuracy": None,
        "BalancedAccuracy": None,
    }


def _metadata(effective_fit_rows=100):
    return {
        "classifier": "DecisionTreeSubset",
        "classifier_name": "DecisionTreeSubset",
        "effective_classifier": "decision_tree_subset",
        "effective_fit_rows": effective_fit_rows,
    }


def _results(active=(), inactive=(), *, metric_overrides=None, metadata_overrides=None):
    metric_overrides = metric_overrides or {}
    metadata_overrides = metadata_overrides or {}
    payload = {"BatchClassifier": {"1-Fold": {}}}
    for mode in active:
        payload[mode] = {
            "DecisionTreeSubset": {
                "1-Fold": {**_metric_block(), **metric_overrides.get(mode, {})}
            }
        }
        payload["BatchClassifier"]["1-Fold"][mode] = {
            **_metadata(),
            **metadata_overrides.get(mode, {}),
        }
    for mode in inactive:
        payload[mode] = {"DecisionTreeSubset": {"1-Fold": _not_applicable_block()}}
    return payload


def _assert_ok(metrics, plan, number_classes=10):
    assert_real_resample_control_metrics(
        metrics,
        number_classes=number_classes,
        synthetic_control="real_resample",
        fold=1,
        protocol_plan=plan,
    )


def _assert_fails(metrics, plan, pattern="real_resample synthetic control failed"):
    with pytest.raises(RuntimeError, match=pattern):
        _assert_ok(metrics, plan)


def test_ts_tr_does_not_require_tr_ts():
    plan = _plan("ts_tr", "both")
    metrics = _results(active=("TS-TR",), inactive=("TR-TS",))

    _assert_ok(metrics, plan)


def test_tr_ts_does_not_require_ts_tr():
    plan = _plan("tr_ts", "both")
    metrics = _results(active=("TR-TS",), inactive=("TS-TR",))

    _assert_ok(metrics, plan)


def test_both_requires_tr_ts_and_ts_tr():
    plan = _plan("legacy", "both")
    metrics = _results(active=("TR-TS",), inactive=("TS-TR",))

    _assert_fails(metrics, plan, "TS-TR/DecisionTreeSubset status is not_applicable")


def test_inactive_protocol_not_applicable_passes():
    plan = _plan("ts_tr", "both")
    metrics = _results(active=("TS-TR",), inactive=("TR-TS",))

    _assert_ok(metrics, plan)


def test_active_protocol_without_accuracy_fails():
    plan = _plan("ts_tr", "both")
    metrics = _results(active=("TS-TR",), metric_overrides={"TS-TR": {"Accuracy": None}})

    _assert_fails(metrics, plan, "missing finite Accuracy")


def test_active_protocol_without_balanced_accuracy_fails():
    plan = _plan("ts_tr", "both")
    metrics = _results(active=("TS-TR",), metric_overrides={"TS-TR": {"BalancedAccuracy": None}})

    _assert_fails(metrics, plan, "missing finite BalancedAccuracy")


def test_active_protocol_with_nan_fails():
    plan = _plan("ts_tr", "both")
    metrics = _results(active=("TS-TR",), metric_overrides={"TS-TR": {"Accuracy": math.nan}})

    _assert_fails(metrics, plan, "missing finite Accuracy")


def test_active_protocol_with_failed_status_fails():
    plan = _plan("ts_tr", "both")
    metrics = _results(active=("TS-TR",), metric_overrides={"TS-TR": {"status": "failed"}})

    _assert_fails(metrics, plan, "status is failed")


def test_null_is_not_accepted_for_active_protocol():
    plan = _plan("ts_tr", "both")
    metrics = _results(active=("TS-TR",), metric_overrides={"TS-TR": {"Accuracy": None}})

    _assert_fails(metrics, plan, "missing finite Accuracy")


def test_null_is_accepted_for_inactive_protocol():
    plan = _plan("ts_tr", "both")
    metrics = _results(active=("TS-TR",), inactive=("TR-TS",))

    assert metrics["TR-TS"]["DecisionTreeSubset"]["1-Fold"]["Accuracy"] is None
    _assert_ok(metrics, plan)


def test_legacy_evaluation_list_is_derived_from_canonical_protocol():
    parsed = _normalize_evaluation_protocol_arguments(_args("tr_ts", "both"))

    assert parsed.evaluation == ["TrAs"]
    assert parsed.evaluation_protocol_plan.required_result_keys == ("TR-TS",)


def test_ts_tr_produces_only_tsar_legacy_evaluation():
    parsed = _normalize_evaluation_protocol_arguments(_args("ts_tr", "both"))

    assert parsed.evaluation == ["TsAr"]


def test_results_use_canonical_keys_after_alias_normalization():
    normalized = normalize_results_keys(
        {
            "TrAs": {"DecisionTreeSubset": {}},
            "ts_tr": {"DecisionTreeSubset": {}},
            "TR-TS": {"AnotherClassifier": {}},
        }
    )

    assert "TrAs" not in normalized
    assert "ts_tr" not in normalized
    assert set(normalized) == {"TR-TS", "TS-TR"}
    assert set(normalized["TR-TS"]) == {"DecisionTreeSubset", "AnotherClassifier"}


def test_real_resample_ten_classes_ts_tr_concludes():
    plan = _plan("ts_tr", "both")
    metrics = _results(active=("TS-TR",), inactive=("TR-TS",))

    _assert_ok(metrics, plan, number_classes=10)


def test_old_evaluation_mode_compatibility_still_selects_both():
    args = _args("appclassnet_strict", "both")

    assert selected_protocol_ids(args) == ["TR_TS", "TS_TR"]
    assert legacy_evaluation_for_plan(resolve_evaluation_protocol_plan(args)) == ["TrAs", "TsAr"]


def test_legacy_csv_results_without_status_or_batch_metadata_remain_compatible():
    plan = _plan("ts_tr", "both")
    legacy_metrics = {
        "TS-TR": {
            "DecisionTreeSubset": {
                "1-Fold": {
                    "Accuracy": 0.91,
                    "BalancedAccuracy": 0.9,
                }
            }
        },
        "TR-TS": {
            "DecisionTreeSubset": {
                "1-Fold": {
                    "Accuracy": None,
                    "BalancedAccuracy": None,
                    "status": "not_run",
                }
            }
        },
    }

    _assert_ok(legacy_metrics, plan)


def test_canonical_result_key_accepts_required_aliases():
    assert canonical_result_key("TrAs") == "TR-TS"
    assert canonical_result_key("TsAr") == "TS-TR"
    assert canonical_result_key("tr_ts") == "TR-TS"
    assert canonical_result_key("ts_tr") == "TS-TR"
