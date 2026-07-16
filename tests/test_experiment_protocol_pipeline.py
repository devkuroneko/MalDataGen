from types import SimpleNamespace

import numpy

from Engine.Classifiers.BatchClassifiers import train_batch_classifier
from Engine.Evaluation.ExperimentProtocol import selected_protocol_ids
from run_appclassnet_top200 import apply_execution_profile
from run_appclassnet_top200 import apply_experiment_budget_scenario
from run_appclassnet_top200 import annotate_explicit_cli_arguments
from run_appclassnet_top200 import build_parser


def test_canonical_evaluation_protocol_overrides_evaluation_mode_alias():
    args = SimpleNamespace(
        evaluation_protocol="tr_plus_ts_tr",
        evaluation_mode="both",
        run_tr_tr=True,
    )

    assert selected_protocol_ids(args) == ["TR_PLUS_TS_TR"]


def test_legacy_evaluation_mode_remains_supported():
    args = SimpleNamespace(
        evaluation_protocol="appclassnet_strict",
        evaluation_mode="both",
        run_tr_tr=False,
    )

    assert selected_protocol_ids(args) == ["TR_TS", "TS_TR"]


def test_budget_scenario_d_sets_effective_augmented_budget_unless_overridden():
    parser = build_parser()
    argv = ["--experiment_budget_scenario", "r50_s150_r500"]
    args = parser.parse_args(argv)
    annotate_explicit_cli_arguments(args, argv)

    apply_execution_profile(args)
    apply_experiment_budget_scenario(args)

    assert args.evaluation_protocol == "tr_plus_ts_tr"
    assert args.pipeline_effective == "augmentation"
    assert args.evaluation_mode == "tr_ts_tr"
    assert args.train_samples_per_class == 50
    assert args.synthetic_train_samples_per_class == 150
    assert args.generated_samples_per_class == 150
    assert args.test_samples_per_class == 500


def test_budget_scenario_allows_explicit_quota_override():
    parser = build_parser()
    argv = [
        "--experiment_budget_scenario",
        "r50_s150_r500",
        "--synthetic_train_samples_per_class",
        "17",
    ]
    args = parser.parse_args(argv)
    annotate_explicit_cli_arguments(args, argv)

    apply_execution_profile(args)
    apply_experiment_budget_scenario(args)

    assert args.synthetic_train_samples_per_class == 17
    assert args.train_samples_per_class == 50


def test_decision_tree_subset_reports_discarded_rows_and_hyperparameters():
    x_values = numpy.vstack([
        numpy.full((5, 2), class_id, dtype=numpy.float32)
        for class_id in range(3)
    ])
    y_values = numpy.concatenate([
        numpy.full(5, class_id, dtype=numpy.int64)
        for class_id in range(3)
    ])
    args = SimpleNamespace(
        random_state=7,
        train_samples_per_class=1,
        batch_classifier_subset_size=100,
        decision_tree_criterion="gini",
        decision_tree_max_depth=None,
        decision_tree_max_features=None,
        decision_tree_max_leaf_nodes=None,
        max_depth=None,
    )

    _, metadata = train_batch_classifier(
        "decision_tree_subset",
        [(x_values, y_values)],
        3,
        args,
    )

    assert metadata["requested_classifier"] == "decision_tree_subset"
    assert metadata["effective_classifier"] == "decision_tree_subset"
    assert metadata["effective_fit_rows"] == 3
    assert metadata["fit_rows_input"] == 15
    assert metadata["discarded_rows_for_fit"] is True
    assert metadata["discarded_rows"] == 12
    assert metadata["random_state"] == 7
