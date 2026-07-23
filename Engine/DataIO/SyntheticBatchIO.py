#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Incremental synthetic batch persistence for low-memory experiments."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy

from Engine.DataIO.JsonIO import atomic_write_json
from Engine.DataIO.JsonIO import load_json_file


class SyntheticBatchWriter:
    """Persist synthetic samples incrementally and maintain a manifest."""

    def __init__(
            self,
            root_dir,
            num_classes,
            num_features,
            seed,
            model_name,
            execution_mode,
            output_format="npy_batches",
            data_space="source",
            transform_id=None,
            transform_history=None,
            feature_names=None,
            feature_dtype=None,
            schema_hash=None,
            split_name=None,
            fold_number=None,
            requested_classes=None,
            generation_plan=None,
            label_mapping=None,
            batch_sizes=None,
            training=None,
            code_version=None,
            data_hashes=None):
        self.root_dir = Path(root_dir)
        self.num_classes = int(num_classes)
        self.num_features = int(num_features)
        self.seed = seed
        self.model_name = model_name
        self.execution_mode = execution_mode
        self.output_format = output_format
        self.data_space = data_space
        self.transform_id = transform_id
        self.transform_history = list(transform_history or [])
        self.feature_names = list(feature_names or [f"f{index}" for index in range(self.num_features)])
        self.feature_dtype = str(feature_dtype) if feature_dtype is not None else None
        self.schema_hash = schema_hash
        self.split_name = split_name
        self.fold_number = fold_number
        self.requested_classes = None if requested_classes is None else [int(label) for label in requested_classes]
        self.generation_plan = dict(generation_plan or {})
        self.label_mapping = label_mapping
        self.batch_sizes = dict(batch_sizes or {})
        self.training = training
        self.code_version = code_version
        self.data_hashes = dict(data_hashes or {})
        self.batch_dir = self.root_dir / "synthetic_batches"
        if self.split_name:
            self.batch_dir = self.batch_dir / str(self.split_name)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self._single_npy_path = self.batch_dir / "synthetic.npy"
        self._single_npy = None
        self._single_npy_offset = 0
        self._total_expected_rows = None
        self._xy_batch_index = 0
        self._global_min = None
        self._global_max = None
        self._nan_count = 0
        self._inf_count = 0
        self._generated_by_class = {}
        self.manifest = {
            "total_rows": 0,
            "num_classes": self.num_classes,
            "features": self.num_features,
            "batches_by_class": {},
            "seed": self.seed,
            "model": self.model_name,
            "execution_mode": self.execution_mode,
            "format": self.output_format,
            "data_space": self.data_space,
            "transform_id": self.transform_id,
            "transform_history": self.transform_history,
            "feature_names": self.feature_names,
            "feature_order": self.feature_names,
            "feature_dtype": self.feature_dtype,
            "schema_hash": self.schema_hash,
            "split": self.split_name,
            "fold": self.fold_number,
            "requested_classes": self.requested_classes,
            "generation_plan": self.generation_plan,
            "label_mapping": self.label_mapping,
            "batch_sizes": self.batch_sizes,
            "training": self.training,
            "code_version": self.code_version,
            "data_hashes": self.data_hashes,
            "synthetic_manifest_version": 2,
            "dataset_origin": self.data_hashes,
            "training_split": self.split_name or "train",
            "generator": self.model_name,
            "generator_parameters": self.generation_plan.get("generator_parameters", {}),
            "transformation": {
                "data_space": self.data_space,
                "transform_id": self.transform_id,
                "transform_history": self.transform_history,
            },
            "classes": self.requested_classes,
            "requested_per_class": self._requested_per_class_payload(),
            "generated_per_class": {},
            "batches": [],
            "feature_min": None,
            "feature_max": None,
            "nan_count": 0,
            "inf_count": 0,
            "paths": {
                "root_dir": str(self.root_dir),
                "batch_dir": str(self.batch_dir),
                "manifest": str(self.batch_dir / "manifest.json"),
                "synthetic_manifest": str(self.batch_dir / "synthetic_manifest.json"),
            },
        }

    def initialize_single_npy(self, total_rows, dtype=numpy.float32):
        self._total_expected_rows = int(total_rows)
        self._single_npy = numpy.lib.format.open_memmap(
            self._single_npy_path,
            mode="w+",
            dtype=dtype,
            shape=(self._total_expected_rows, self.num_features),
        )
        self.manifest["single_npy_path"] = str(self._single_npy_path)

    def write_batch(self, class_label, batch_index, x_batch):
        if int(class_label) < 0 or int(class_label) >= self.num_classes:
            raise ValueError(
                f"Synthetic batch class label must be within [0, {self.num_classes - 1}]. "
                f"Got {class_label}."
            )
        x_batch = numpy.asarray(x_batch)
        if x_batch.shape[0] == 0:
            raise ValueError(f"Synthetic batch for class={class_label} batch={batch_index} is empty.")
        if x_batch.ndim != 2 or x_batch.shape[1] != self.num_features:
            raise ValueError(
                f"Synthetic batch must have shape (n, {self.num_features}). Got {x_batch.shape}."
            )
        nan_count = int(numpy.isnan(x_batch).sum()) if numpy.issubdtype(x_batch.dtype, numpy.number) else 0
        inf_count = int(numpy.isinf(x_batch).sum()) if numpy.issubdtype(x_batch.dtype, numpy.number) else 0
        if nan_count or inf_count:
            raise ValueError(
                f"Synthetic batch class={class_label} batch={batch_index} contains non-finite values: "
                f"nan_count={nan_count} inf_count={inf_count}."
            )

        class_key = str(int(class_label))
        class_dir = self.batch_dir / f"class_{int(class_label):03d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        if self.output_format == "npy_batches":
            file_path = class_dir / f"batch_{batch_index:06d}.npy"
            numpy.save(file_path, x_batch)
            batch_entry = {
                "path": str(file_path),
                "shape": [int(x_batch.shape[0]), int(x_batch.shape[1])],
                "dtype": str(x_batch.dtype),
            }
        elif self.output_format == "csv_batches":
            file_path = class_dir / f"batch_{batch_index:06d}.csv"
            with file_path.open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(x_batch.tolist())
            batch_entry = {
                "path": str(file_path),
                "shape": [int(x_batch.shape[0]), int(x_batch.shape[1])],
                "dtype": str(x_batch.dtype),
            }
        elif self.output_format == "single_npy":
            if self._single_npy is None:
                raise ValueError("initialize_single_npy must be called before writing single_npy batches.")
            offset_start = self._single_npy_offset
            end = self._single_npy_offset + x_batch.shape[0]
            self._single_npy[self._single_npy_offset:end] = x_batch
            self._single_npy_offset = end
            file_path = self._single_npy_path
            batch_entry = {
                "path": str(file_path),
                "shape": [int(x_batch.shape[0]), int(x_batch.shape[1])],
                "dtype": str(x_batch.dtype),
                "offset_start": int(offset_start),
                "offset_end": int(end),
            }
        else:
            raise ValueError(f"Unsupported synthetic output format: {self.output_format}")

        xy_x_path = self.batch_dir / f"x_{self._xy_batch_index:05d}.npy"
        xy_y_path = self.batch_dir / f"y_{self._xy_batch_index:05d}.npy"
        y_batch = numpy.full(int(x_batch.shape[0]), int(class_label), dtype=numpy.int64)
        numpy.save(xy_x_path, x_batch)
        numpy.save(xy_y_path, y_batch)
        xy_entry = {
            "batch_index": int(self._xy_batch_index),
            "class_id": int(class_label),
            "x_path": str(xy_x_path),
            "y_path": str(xy_y_path),
            "x_shape": [int(x_batch.shape[0]), int(x_batch.shape[1])],
            "y_shape": [int(y_batch.shape[0])],
            "x_dtype": str(x_batch.dtype),
            "y_dtype": str(y_batch.dtype),
            "min": float(numpy.min(x_batch)),
            "max": float(numpy.max(x_batch)),
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
        self._xy_batch_index += 1
        self.manifest["batches"].append(xy_entry)
        self._generated_by_class[class_key] = int(self._generated_by_class.get(class_key, 0)) + int(x_batch.shape[0])
        self.manifest["generated_per_class"] = dict(self._generated_by_class)
        self._global_min = xy_entry["min"] if self._global_min is None else min(self._global_min, xy_entry["min"])
        self._global_max = xy_entry["max"] if self._global_max is None else max(self._global_max, xy_entry["max"])
        self._nan_count += nan_count
        self._inf_count += inf_count
        self.manifest["feature_min"] = self._global_min
        self.manifest["feature_max"] = self._global_max
        self.manifest["nan_count"] = int(self._nan_count)
        self.manifest["inf_count"] = int(self._inf_count)

        self.manifest["batches_by_class"].setdefault(class_key, []).append(batch_entry)
        self.manifest["total_rows"] += int(x_batch.shape[0])
        return file_path

    def close(self):
        if self._single_npy is not None:
            self._single_npy.flush()
            if self._total_expected_rows is not None and self._single_npy_offset != self._total_expected_rows:
                logging.warning(
                    "single_npy expected %d rows but wrote %d rows.",
                    self._total_expected_rows,
                    self._single_npy_offset,
                )
        self._validate_manifest_before_close()
        manifest_path = self.batch_dir / "manifest.json"
        synthetic_manifest_path = self.batch_dir / "synthetic_manifest.json"
        self.manifest["paths"]["manifest"] = str(manifest_path)
        self.manifest["paths"]["synthetic_manifest"] = str(synthetic_manifest_path)
        atomic_write_json(self.manifest, manifest_path, indent=2, model=self.model_name)
        atomic_write_json(self.manifest, synthetic_manifest_path, indent=2, model=self.model_name)
        return SyntheticBatchReader(manifest_path)

    def _requested_per_class_payload(self):
        classes = self.generation_plan.get("classes", {})
        return {str(int(class_id)): int(count) for class_id, count in classes.items()}

    def _validate_manifest_before_close(self):
        if int(self.manifest["total_rows"]) <= 0:
            raise ValueError("Synthetic generation produced no rows; refusing to write an empty manifest.")
        if not self.manifest["batches"]:
            raise ValueError("Synthetic generation produced no non-empty batches.")
        for batch in self.manifest["batches"]:
            x_path = Path(batch["x_path"])
            y_path = Path(batch["y_path"])
            if not x_path.is_file() or not y_path.is_file():
                raise ValueError(f"Synthetic batch files are missing: x={x_path} y={y_path}.")
            if int(batch["x_shape"][1]) != self.num_features:
                raise ValueError(
                    f"Synthetic batch {batch['batch_index']} has {batch['x_shape'][1]} features; "
                    f"expected {self.num_features}."
                )
            if int(batch["x_shape"][0]) != int(batch["y_shape"][0]):
                raise ValueError(
                    f"Synthetic batch {batch['batch_index']} X/y row mismatch: "
                    f"X rows={batch['x_shape'][0]} y rows={batch['y_shape'][0]}."
                )
        requested = self.manifest.get("requested_per_class", {})
        generated = self.manifest.get("generated_per_class", {})
        for class_id, requested_count in requested.items():
            generated_count = int(generated.get(str(class_id), 0))
            if generated_count != int(requested_count):
                raise ValueError(
                    f"Synthetic generation count mismatch for class={class_id}: "
                    f"requested={int(requested_count)} generated={generated_count}."
                )


class SyntheticBatchReader:
    """Iterate synthetic batches from a manifest without loading all samples."""

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        self.manifest = load_json_file(self.manifest_path)

    @property
    def total_rows(self):
        return int(self.manifest.get("total_rows", 0))

    @property
    def num_classes(self):
        return int(self.manifest.get("num_classes", 0))

    @property
    def num_features(self):
        return int(self.manifest.get("features", 0))

    def items(self):
        for class_label, batches in self.manifest.get("batches_by_class", {}).items():
            for batch in batches:
                yield int(class_label), self._load_batch(batch)

    def iter_batches(self):
        yield from self.items()

    def iter_xy_batches(self):
        for batch in self.manifest.get("batches", []):
            yield (
                numpy.load(batch["x_path"], mmap_mode="r", allow_pickle=False),
                numpy.load(batch["y_path"], mmap_mode="r", allow_pickle=False),
                batch,
            )

    def _load_batch(self, batch):
        path = Path(batch["path"])
        if path.suffix == ".npy":
            values = numpy.load(path, mmap_mode="r", allow_pickle=False)
            if "offset_start" in batch:
                return values[int(batch["offset_start"]):int(batch["offset_end"])]
            return values
        if path.suffix == ".csv":
            return numpy.loadtxt(path, delimiter=",", dtype=numpy.float32)
        raise ValueError(f"Unsupported synthetic batch file: {path}")


class SyntheticSplitBatchReaders:
    """Container for separate train/test synthetic batch manifests."""

    def __init__(self, train_reader, test_reader):
        self.train_reader = train_reader
        self.test_reader = test_reader

    @property
    def manifest_path(self):
        return {
            "train": str(self.train_reader.manifest_path),
            "test": str(self.test_reader.manifest_path),
        }

    @property
    def manifest(self):
        return {
            "train": self.train_reader.manifest,
            "test": self.test_reader.manifest,
        }
