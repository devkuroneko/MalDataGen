#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Batch iterator for separated X/y NumPy dataset files.

This module is intentionally independent from the legacy CSV loader and from
NpyXYLoader. It keeps `.npy` arrays memory-mapped when requested and yields
small batches for lower-memory AppClassNet execution paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy

from Engine.DataIO.StratifiedNpySelection import get_last_stratified_selection_report
from Engine.DataIO.StratifiedNpySelection import select_stratified_indices_from_npy


class BatchNpyDataset:
    """Iterate over separated X/y `.npy` files without loading X in RAM.

    Shuffle policy:
        For small effective datasets this class uses a full int64 permutation of
        row indices. For larger datasets it shuffles one index chunk at a time.
        Chunk shuffle avoids allocating a permutation proportional to the full
        dataset, but it is not a perfect global shuffle because rows do not move
        across chunk boundaries.
    """

    _MAX_FULL_SHUFFLE_INDICES = 1_000_000
    _SHUFFLE_CHUNK_MULTIPLIER = 64

    def __init__(
            self,
            x_path,
            y_path,
            batch_size,
            mmap_mode="r",
            shuffle=False,
            seed=42,
            dtype=None,
            max_samples=None,
            max_samples_per_class=None,
            num_classes=None):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.batch_size = self._validate_positive_int(batch_size, "batch_size")
        self.mmap_mode = mmap_mode
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.dtype = dtype
        self.max_samples = self._validate_optional_non_negative_int(max_samples, "max_samples")
        self.max_samples_per_class = self._validate_optional_non_negative_int(
            max_samples_per_class,
            "max_samples_per_class",
        )
        self.configured_num_classes = num_classes
        self.selection_report = None

        self._x = self._load_array(self.x_path)
        self._y = self._normalize_y(self._load_array(self.y_path))
        self._validate_arrays()

        self._all_classes = numpy.unique(self._y)
        self._indices = self._build_indices()
        self._num_samples = self._get_effective_num_samples()
        self._classes = numpy.unique(self._labels_for_metadata())

    @staticmethod
    def _validate_positive_int(value, field_name):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer.")
        return value

    @staticmethod
    def _validate_optional_non_negative_int(value, field_name):
        if value is None:
            return None
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer when provided.")
        return value

    def _load_array(self, path):
        if not path.is_file():
            raise FileNotFoundError(f"NumPy dataset file not found: {path}")
        return numpy.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)

    @staticmethod
    def _normalize_y(y_values):
        y_array = y_values if isinstance(y_values, numpy.ndarray) else numpy.asarray(y_values)
        if y_array.ndim == 0:
            raise ValueError("y must be 1D or safely convertible with reshape(-1).")
        if y_array.ndim == 1:
            return y_array
        return y_array.reshape(-1)

    def _validate_arrays(self):
        if self._x.ndim != 2:
            raise ValueError(f"X must be a 2D matrix. Got shape {self._x.shape}.")
        if self._x.shape[0] != self._y.shape[0]:
            raise ValueError(
                f"X/y row mismatch: X has {self._x.shape[0]} rows, y has {self._y.shape[0]} rows."
            )

    def _build_indices(self):
        if self.max_samples_per_class is not None:
            return self._select_max_samples_per_class()

        if self.max_samples is not None and self.max_samples < self._x.shape[0]:
            return numpy.arange(self.max_samples, dtype=numpy.int64)

        return None

    def _select_max_samples_per_class(self):
        logging.warning(
            "max_samples_per_class uses mmap y scan and builds a compact selected-index array; "
            "memory is proportional to selected rows, not to X."
        )
        if self.max_samples is not None:
            logging.warning(
                "Ignoring max_samples=%s while selecting max_samples_per_class=%s; "
                "class coverage is guaranteed before any total cap.",
                self.max_samples,
                self.max_samples_per_class,
            )
        num_classes = self._configured_or_inferred_num_classes()
        indices = select_stratified_indices_from_npy(
            self.y_path,
            self.max_samples_per_class,
            num_classes,
            self.seed,
            mmap_mode=self.mmap_mode,
        )
        self.selection_report = get_last_stratified_selection_report()
        return indices

    def _configured_or_inferred_num_classes(self):
        if self.configured_num_classes is not None:
            return int(self.configured_num_classes)
        if self._all_classes.size == 0:
            return 1
        return int(numpy.max(self._all_classes)) + 1

    def _get_effective_num_samples(self):
        if self._indices is not None:
            return int(self._indices.shape[0])
        if self.max_samples is not None:
            return min(int(self.max_samples), int(self._x.shape[0]))
        return int(self._x.shape[0])

    def _labels_for_metadata(self):
        if self._indices is not None:
            return self._y[self._indices]
        return self._y[:self._num_samples]

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def num_features(self):
        return int(self._x.shape[1])

    @property
    def classes_(self):
        return self._classes

    @property
    def num_classes(self):
        return int(self._classes.shape[0])

    @property
    def shape(self):
        return self.num_samples, self.num_features

    @property
    def X(self):
        return self._x

    @property
    def y(self):
        return self._y

    def head(self, n):
        n = self._validate_optional_non_negative_int(n, "n")
        end = min(n, self.num_samples)
        if self._indices is None:
            return self._format_batch(self._x[:end], self._y[:end])
        indices = self._indices[:end]
        return self._format_batch(self._x[indices], self._y[indices])

    def iter_batches(self):
        if self.num_samples == 0:
            return

        if self.shuffle:
            yield from self._iter_shuffled_batches()
            return

        if self._indices is None:
            yield from self._iter_contiguous_batches()
            return

        yield from self._iter_indexed_batches(self._indices)

    def _iter_contiguous_batches(self):
        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            yield self._format_batch(self._x[start:end], self._y[start:end])

    def _iter_indexed_batches(self, indices):
        for start in range(0, indices.shape[0], self.batch_size):
            end = min(start + self.batch_size, indices.shape[0])
            batch_indices = indices[start:end]
            yield self._format_batch(self._x[batch_indices], self._y[batch_indices])

    def _iter_shuffled_batches(self):
        random_generator = numpy.random.default_rng(self.seed)

        if self._indices is not None:
            indices = numpy.array(self._indices, copy=True)
            random_generator.shuffle(indices)
            yield from self._iter_indexed_batches(indices)
            return

        if self.num_samples <= self._MAX_FULL_SHUFFLE_INDICES:
            indices = random_generator.permutation(self.num_samples)
            yield from self._iter_indexed_batches(indices)
            return

        chunk_size = max(self.batch_size, self.batch_size * self._SHUFFLE_CHUNK_MULTIPLIER)
        logging.warning(
            "Dataset has %d rows; using chunk-local shuffle with chunk_size=%d to avoid a full permutation.",
            self.num_samples,
            chunk_size,
        )
        for start in range(0, self.num_samples, chunk_size):
            end = min(start + chunk_size, self.num_samples)
            chunk_indices = numpy.arange(start, end, dtype=numpy.int64)
            random_generator.shuffle(chunk_indices)
            yield from self._iter_indexed_batches(chunk_indices)

    def _format_batch(self, x_batch, y_batch):
        if self.dtype is not None:
            x_batch = x_batch.astype(self.dtype, copy=False)
        return x_batch, numpy.asarray(y_batch)
