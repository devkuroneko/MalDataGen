#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run paired AppClassNet real/synthetic significance experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy

from Engine.Evaluation.SignificanceExperiment import SignificanceConfig
from Engine.Evaluation.SignificanceExperiment import SignificanceExperimentRunner


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_x_path", required=True)
    parser.add_argument("--train_y_path", required=True)
    parser.add_argument("--test_x_path", required=True)
    parser.add_argument("--test_y_path", required=True)
    parser.add_argument("--synthetic_x_path")
    parser.add_argument("--synthetic_y_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--classifiers", default="decision_tree_subset")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--class_labels")
    parser.add_argument("--test_samples_per_class", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--max_samples", type=float)
    parser.add_argument("--class_weight")
    parser.add_argument("--noninferiority_margin", type=float)
    parser.add_argument("--noninferiority_margin_justification")
    parser.add_argument("--mmap_mode", default="r")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seeds = _parse_int_list(args.seeds)
    classifiers = [item.strip() for item in args.classifiers.split(",") if item.strip()]
    class_labels = _parse_int_list(args.class_labels) if args.class_labels else None

    train_x = _load(args.train_x_path, args.mmap_mode)
    train_y = _load(args.train_y_path, args.mmap_mode)
    test_x = _load(args.test_x_path, args.mmap_mode)
    test_y = _load(args.test_y_path, args.mmap_mode)
    synthetic_x = _load(args.synthetic_x_path, args.mmap_mode) if args.synthetic_x_path else None
    synthetic_y = _load(args.synthetic_y_path, args.mmap_mode) if args.synthetic_y_path else None

    config = SignificanceConfig(
        output_dir=args.output_dir,
        classifiers=classifiers,
        seeds=seeds,
        test_samples_per_class=args.test_samples_per_class,
        num_classes=args.num_classes,
        class_labels=class_labels,
        eval_batch_size=args.eval_batch_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        class_weight=args.class_weight,
        noninferiority_margin=args.noninferiority_margin,
        noninferiority_margin_justification=args.noninferiority_margin_justification,
        command=[sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        resolved_config=vars(args),
        paths={
            "train_x_path": str(args.train_x_path),
            "train_y_path": str(args.train_y_path),
            "test_x_path": str(args.test_x_path),
            "test_y_path": str(args.test_y_path),
            "synthetic_x_path": str(args.synthetic_x_path) if args.synthetic_x_path else None,
            "synthetic_y_path": str(args.synthetic_y_path) if args.synthetic_y_path else None,
            "output_dir": str(args.output_dir),
        },
    )
    paths = SignificanceExperimentRunner(
        train_x,
        train_y,
        test_x,
        test_y,
        synthetic_x,
        synthetic_y,
        config,
    ).run()
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))
    return 0


def _load(path, mmap_mode):
    return numpy.load(path, mmap_mode=mmap_mode, allow_pickle=False)


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).replace(" ", ",").split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
