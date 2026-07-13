#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Incremental synthetic batch persistence for low-memory experiments."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy


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
            transform_history=None):
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
        self.batch_dir = self.root_dir / "synthetic_batches"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self._single_npy_path = self.batch_dir / "synthetic.npy"
        self._single_npy = None
        self._single_npy_offset = 0
        self._total_expected_rows = None
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
        x_batch = numpy.asarray(x_batch)
        if x_batch.ndim != 2 or x_batch.shape[1] != self.num_features:
            raise ValueError(
                f"Synthetic batch must have shape (n, {self.num_features}). Got {x_batch.shape}."
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
            }
        elif self.output_format == "csv_batches":
            file_path = class_dir / f"batch_{batch_index:06d}.csv"
            with file_path.open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(x_batch.tolist())
            batch_entry = {
                "path": str(file_path),
                "shape": [int(x_batch.shape[0]), int(x_batch.shape[1])],
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
                "offset_start": int(offset_start),
                "offset_end": int(end),
            }
        else:
            raise ValueError(f"Unsupported synthetic output format: {self.output_format}")

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
        manifest_path = self.batch_dir / "manifest.json"
        with manifest_path.open("w") as manifest_file:
            json.dump(self.manifest, manifest_file, indent=2)
        return SyntheticBatchReader(manifest_path)


class SyntheticBatchReader:
    """Iterate synthetic batches from a manifest without loading all samples."""

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open() as manifest_file:
            self.manifest = json.load(manifest_file)

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
