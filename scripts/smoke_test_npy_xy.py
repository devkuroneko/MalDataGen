#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Smoke test helper for MalDataGen npy_xy datasets.

The default path creates a small synthetic X/y dataset and validates the new
loader/schema/sample-plan flow without running the full generative pipeline.
Pass --run-pipeline to execute main.py with the generated or provided files.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Engine.DataIO.NpyXYLoader import NpyXYLoader
from Engine.DataIO.SamplePlanner import build_sample_plan_from_args
from Engine.DataIO.SamplePlanner import sample_plan_to_legacy_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or validate a small npy_xy dataset for MalDataGen."
    )
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes to generate or validate. Use 200 to simulate AppClassNet top-200.")
    parser.add_argument("--samples_per_class", type=int, default=2,
                        help="Rows per class for generated train split and balanced synthetic sample plan.")
    parser.add_argument("--valid_samples_per_class", type=int, default=1,
                        help="Rows per class for generated valid/test splits.")
    parser.add_argument("--num_features", type=int, default=20,
                        help="Feature count. AppClassNet uses 20.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic arrays.")
    parser.add_argument("--work_dir", type=Path, default=None,
                        help="Directory to write generated .npy files. Defaults to outputs/smoke_npy_xy_<date>/dataset.")

    parser.add_argument("--train_x_path", type=Path, default=None,
                        help="Optional real train_x.npy path. If supplied, all X/y paths must be supplied.")
    parser.add_argument("--train_y_path", type=Path, default=None)
    parser.add_argument("--valid_x_path", type=Path, default=None)
    parser.add_argument("--valid_y_path", type=Path, default=None)
    parser.add_argument("--test_x_path", type=Path, default=None)
    parser.add_argument("--test_y_path", type=Path, default=None)

    parser.add_argument("--sample_plan", choices=["balanced_per_class", "match_train_distribution", "total_rows"],
                        default="balanced_per_class",
                        help="Sample plan to validate and include in the recommended main.py command.")
    parser.add_argument("--total_synthetic_rows", type=int, default=None,
                        help="Total rows for match_train_distribution or total_rows sample plans.")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Validate parsing/loading/schema/sample-plan only. This is the default.")
    parser.add_argument("--run-pipeline", action="store_true",
                        help="Actually execute main.py after creating or validating the .npy files.")
    parser.add_argument("--model_type", default="random",
                        help="Model type to pass to main.py when --run-pipeline is used.")
    parser.add_argument("--classifier", default="DecisionTree",
                        help="Classifier to pass to main.py when --run-pipeline is used.")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Optional output directory to pass to main.py. Defaults to outputs/smoke_npy_xy_<date>/pipeline.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_default_output_paths(args)
    paths = resolve_dataset_paths(args)

    if not using_real_paths(args):
        create_synthetic_npy_dataset(paths, args)

    bundle = load_bundle(paths, args)
    sample_metadata = build_sample_metadata(bundle, args)
    command = build_main_command(paths, args)

    print_summary(paths, bundle, sample_metadata, command)
    print_real_appclassnet_example()

    if args.run_pipeline:
        print("\nRunning pipeline command...")
        return subprocess.call(command, cwd=str(REPO_ROOT))

    return 0


def apply_default_output_paths(args: argparse.Namespace) -> None:
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_run_dir = OUTPUTS_ROOT / f"smoke_npy_xy_{run_id}"

    if not using_real_paths(args) and args.work_dir is None:
        args.work_dir = default_run_dir / "dataset"

    if args.output_dir is None:
        args.output_dir = default_run_dir / "pipeline"


def using_real_paths(args: argparse.Namespace) -> bool:
    supplied = [
        args.train_x_path, args.train_y_path,
        args.valid_x_path, args.valid_y_path,
        args.test_x_path, args.test_y_path,
    ]
    if any(path is not None for path in supplied):
        if not all(path is not None for path in supplied):
            raise ValueError("When using real files, provide train/valid/test X and y paths.")
        return True
    return False


def resolve_dataset_paths(args: argparse.Namespace) -> dict[str, Path]:
    if using_real_paths(args):
        return {
            "train_x": args.train_x_path,
            "train_y": args.train_y_path,
            "valid_x": args.valid_x_path,
            "valid_y": args.valid_y_path,
            "test_x": args.test_x_path,
            "test_y": args.test_y_path,
        }

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    return {
        "train_x": work_dir / "train_x.npy",
        "train_y": work_dir / "train_y.npy",
        "valid_x": work_dir / "valid_x.npy",
        "valid_y": work_dir / "valid_y.npy",
        "test_x": work_dir / "test_x.npy",
        "test_y": work_dir / "test_y.npy",
    }


def create_synthetic_npy_dataset(paths: dict[str, Path], args: argparse.Namespace) -> None:
    rng = numpy.random.default_rng(args.seed)
    write_split(paths["train_x"], paths["train_y"], args.num_classes, args.samples_per_class, args.num_features, rng)
    write_split(paths["valid_x"], paths["valid_y"], args.num_classes, args.valid_samples_per_class, args.num_features, rng)
    write_split(paths["test_x"], paths["test_y"], args.num_classes, args.valid_samples_per_class, args.num_features, rng)


def write_split(
        x_path: Path,
        y_path: Path,
        num_classes: int,
        samples_per_class: int,
        num_features: int,
        rng: numpy.random.Generator) -> None:
    labels = numpy.repeat(numpy.arange(num_classes, dtype=numpy.int64), samples_per_class)
    features = rng.normal(loc=0.0, scale=1.0, size=(labels.shape[0], num_features)).astype(numpy.float32)
    features += labels.reshape(-1, 1).astype(numpy.float32) / max(num_classes, 1)
    numpy.save(x_path, features)
    numpy.save(y_path, labels)


def load_bundle(paths: dict[str, Path], args: argparse.Namespace):
    loader = NpyXYLoader(
        train_x_path=paths["train_x"],
        train_y_path=paths["train_y"],
        valid_x_path=paths["valid_x"],
        valid_y_path=paths["valid_y"],
        test_x_path=paths["test_x"],
        test_y_path=paths["test_y"],
        mmap_mode="r",
        target_type="multiclass",
        feature_type="continuous",
        num_classes=args.num_classes,
    )
    return loader.load()


def build_sample_metadata(bundle, args: argparse.Namespace) -> dict:
    planner_args = SimpleNamespace(
        data_format="npy_xy",
        sample_plan=args.sample_plan,
        number_samples_per_class="default-not-explicit",
        samples_per_class=args.samples_per_class,
        total_synthetic_rows=args.total_synthetic_rows,
        _legacy_number_samples_per_class_explicit=False,
    )
    plan = build_sample_plan_from_args(
        planner_args,
        bundle.train.y,
        number_classes=bundle.schema.num_classes,
        data_type=bundle.schema.feature_type,
    )
    return sample_plan_to_legacy_metadata(plan, data_type=bundle.schema.feature_type)


def build_main_command(paths: dict[str, Path], args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "--data_format", "npy_xy",
        "--split_mode", "provided",
        "--train_x_path", str(paths["train_x"]),
        "--train_y_path", str(paths["train_y"]),
        "--valid_x_path", str(paths["valid_x"]),
        "--valid_y_path", str(paths["valid_y"]),
        "--test_x_path", str(paths["test_x"]),
        "--test_y_path", str(paths["test_y"]),
        "--target_type", "multiclass",
        "--feature_type", "continuous",
        "--num_classes", str(args.num_classes),
        "--sample_plan", args.sample_plan,
        "--model_type", args.model_type,
        "--classifier", args.classifier,
    ]
    if args.sample_plan == "balanced_per_class":
        command.extend(["--samples_per_class", str(args.samples_per_class)])
    else:
        total_rows = args.total_synthetic_rows
        if total_rows is None:
            total_rows = args.num_classes * args.samples_per_class
        command.extend(["--total_synthetic_rows", str(total_rows)])
    command.extend(["--output_dir", str(args.output_dir)])
    return command


def print_summary(paths: dict[str, Path], bundle, sample_metadata: dict, command: list[str]) -> None:
    print("npy_xy smoke test inputs:")
    for key, path in paths.items():
        print(f"  {key}: {path}")

    print("\nLoaded DatasetBundle:")
    print(f"  schema.source_format: {bundle.schema.source_format}")
    print(f"  schema.feature_type: {bundle.schema.feature_type}")
    print(f"  schema.target_type: {bundle.schema.target_type}")
    print(f"  schema.num_classes: {bundle.schema.num_classes}")
    for split_name, split in bundle.splits.items():
        print(f"  {split_name}: X={split.X.shape}, y={split.y.shape}")

    preview = dict(list(sample_metadata["classes"].items())[:10])
    print("\nSample plan:")
    print(f"  mode: {sample_metadata['sample_plan']}")
    print(f"  number_classes: {sample_metadata['number_classes']}")
    print(f"  first_counts: {preview}")
    print(f"  total_rows: {sum(sample_metadata['classes'].values())}")

    print("\nRecommended command:")
    print(" ".join(shlex.quote(part) for part in command))


def print_real_appclassnet_example() -> None:
    print("\nReal AppClassNet top-200 example:")
    print("Replace /absolute/path/to/appclassnet_top200 with the directory that contains the real .npy files.")
    print(
        "python main.py \\\n"
        "  --data_format npy_xy \\\n"
        "  --split_mode provided \\\n"
        "  --train_x_path /absolute/path/to/appclassnet_top200/train_x.npy \\\n"
        "  --train_y_path /absolute/path/to/appclassnet_top200/train_y.npy \\\n"
        "  --valid_x_path /absolute/path/to/appclassnet_top200/valid_x.npy \\\n"
        "  --valid_y_path /absolute/path/to/appclassnet_top200/valid_y.npy \\\n"
        "  --test_x_path /absolute/path/to/appclassnet_top200/test_x.npy \\\n"
        "  --test_y_path /absolute/path/to/appclassnet_top200/test_y.npy \\\n"
        "  --target_type multiclass \\\n"
        "  --feature_type continuous \\\n"
        "  --num_classes 200 \\\n"
        "  --sample_plan balanced_per_class \\\n"
        "  --samples_per_class 1000"
    )


if __name__ == "__main__":
    raise SystemExit(main())
