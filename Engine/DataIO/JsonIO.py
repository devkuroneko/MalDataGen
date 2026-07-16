#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Canonical JSON conversion, validation, and atomic persistence helpers."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
from pathlib import Path

import numpy


MAX_JSON_ARRAY_ELEMENTS = 1000


class JsonSerializationError(TypeError):
    def __init__(self, message, *, object_path="$", object_type=None, original_error=None):
        self.object_path = object_path
        self.object_type = object_type
        self.original_error = original_error
        super().__init__(message)


class ResultSerializationError(JsonSerializationError):
    def __init__(
            self,
            message,
            *,
            path,
            object_path="$",
            object_type=None,
            run_id=None,
            protocol=None,
            model=None,
            original_error=None):
        self.path = Path(path)
        self.run_id = run_id
        self.protocol = protocol
        self.model = model
        super().__init__(
            (
                f"{message}; path={self.path}; object_path={object_path}; "
                f"object_type={object_type}; run_id={run_id}; protocol={protocol}; model={model}"
            ),
            object_path=object_path,
            object_type=object_type,
            original_error=original_error,
        )


class ResultArtifactMissingError(FileNotFoundError):
    pass


class ResultArtifactCorruptionError(ValueError):
    def __init__(self, path, original_error):
        self.path = Path(path)
        self.original_error = original_error
        super().__init__(f"Malformed JSON artifact: {self.path}: {original_error}")


class ResultSchemaValidationError(ValueError):
    pass


def to_jsonable(value, *, object_path="$", max_array_elements=MAX_JSON_ARRAY_ELEMENTS):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise JsonSerializationError(
                "Non-finite float is not valid JSON.",
                object_path=object_path,
                object_type=type(value).__module__ + "." + type(value).__qualname__,
            )
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value, object_path=object_path, max_array_elements=max_array_elements)
    if isinstance(value, numpy.bool_):
        return bool(value)
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, numpy.floating):
        return to_jsonable(float(value), object_path=object_path, max_array_elements=max_array_elements)
    if isinstance(value, numpy.ndarray):
        if value.size > max_array_elements:
            raise JsonSerializationError(
                f"NumPy array is too large for JSON serialization ({value.size} elements).",
                object_path=object_path,
                object_type=type(value).__module__ + "." + type(value).__qualname__,
            )
        return to_jsonable(value.tolist(), object_path=object_path, max_array_elements=max_array_elements)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict(), object_path=object_path, max_array_elements=max_array_elements)
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value), object_path=object_path, max_array_elements=max_array_elements)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            json_key = to_jsonable(key, object_path=f"{object_path}.<key>", max_array_elements=max_array_elements)
            if not isinstance(json_key, str):
                json_key = str(json_key)
            result[json_key] = to_jsonable(
                item,
                object_path=f"{object_path}.{json_key}",
                max_array_elements=max_array_elements,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(item, object_path=f"{object_path}[{index}]", max_array_elements=max_array_elements)
            for index, item in enumerate(value)
        ]
    if isinstance(value, set):
        return [
            to_jsonable(item, object_path=f"{object_path}[{index}]", max_array_elements=max_array_elements)
            for index, item in enumerate(sorted(value, key=lambda item: str(item)))
        ]
    raise JsonSerializationError(
        "Unsupported object for JSON serialization.",
        object_path=object_path,
        object_type=type(value).__module__ + "." + type(value).__qualname__,
    )


def prepare_json_payload(payload, *, path=None, run_id=None, protocol=None, model=None):
    try:
        serializable_payload = to_jsonable(payload)
        json.dumps(serializable_payload, allow_nan=False)
        return serializable_payload
    except JsonSerializationError as error:
        if path is not None:
            raise ResultSerializationError(
                "Result payload is not JSON serializable",
                path=path,
                object_path=error.object_path,
                object_type=error.object_type,
                run_id=run_id,
                protocol=protocol,
                model=model,
                original_error=error,
            ) from error
        raise
    except (TypeError, ValueError) as error:
        if path is not None:
            raise ResultSerializationError(
                "Result payload failed JSON validation",
                path=path,
                object_path="$",
                object_type=type(payload).__module__ + "." + type(payload).__qualname__,
                run_id=run_id,
                protocol=protocol,
                model=model,
                original_error=error,
            ) from error
        raise JsonSerializationError(
            f"Payload failed JSON validation: {error}",
            object_path="$",
            object_type=type(payload).__module__ + "." + type(payload).__qualname__,
            original_error=error,
        ) from error


def atomic_write_json(payload, path, *, indent=2, sort_keys=True, run_id=None, protocol=None, model=None):
    path = Path(path)
    serializable_payload = prepare_json_payload(
        payload,
        path=path,
        run_id=run_id,
        protocol=protocol,
        model=model,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as output_file:
            json.dump(serializable_payload, output_file, indent=indent, sort_keys=sort_keys, allow_nan=False)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return path


def load_json_file(path, *, normalize=None, missing_error=ResultArtifactMissingError):
    path = Path(path)
    if not path.is_file():
        raise missing_error(f"JSON artifact is missing: {path}")
    try:
        with path.open(encoding="utf-8") as json_file:
            payload = json.load(json_file)
    except json.JSONDecodeError as error:
        raise ResultArtifactCorruptionError(path, error) from error
    if normalize is not None:
        payload = normalize(payload)
    return payload


def move_corrupt_json(path, *, suffix=".corrupt"):
    path = Path(path)
    try:
        load_json_file(path)
        return {"path": str(path), "status": "valid", "moved_to": None, "reason": None}
    except ResultArtifactMissingError as error:
        return {"path": str(path), "status": "missing", "moved_to": None, "reason": str(error)}
    except ResultArtifactCorruptionError as error:
        target = path.with_name(path.name + suffix)
        os.replace(path, target)
        return {"path": str(path), "status": "corrupt", "moved_to": str(target), "reason": str(error.original_error)}
