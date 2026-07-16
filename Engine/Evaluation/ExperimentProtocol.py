#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Canonical AppClassNet experimental protocol helpers."""

from __future__ import annotations

CANONICAL_PROTOCOLS = {
    "tr_tr": {
        "protocol_id": "TR_TR",
        "legacy_name": "TR-TR",
        "train_sources": ["real_train"],
        "test_source": "real_test",
    },
    "tr_ts": {
        "protocol_id": "TR_TS",
        "legacy_name": "TR-TS",
        "train_sources": ["real_train"],
        "test_source": "synthetic_test",
    },
    "ts_tr": {
        "protocol_id": "TS_TR",
        "legacy_name": "TS-TR",
        "train_sources": ["synthetic_train"],
        "test_source": "real_test",
    },
    "tr_plus_ts_tr": {
        "protocol_id": "TR_PLUS_TS_TR",
        "legacy_name": "TR+TS-TR",
        "train_sources": ["real_train", "synthetic_train"],
        "test_source": "real_test",
    },
}

CANONICAL_PROTOCOL_CHOICES = tuple(CANONICAL_PROTOCOLS) + ("all",)
LEGACY_EVALUATION_PROTOCOL_CHOICES = ("legacy", "appclassnet_strict")
EVALUATION_PROTOCOL_CHOICES = LEGACY_EVALUATION_PROTOCOL_CHOICES + CANONICAL_PROTOCOL_CHOICES
LEGACY_TO_CANONICAL = {
    metadata["legacy_name"]: metadata["protocol_id"]
    for metadata in CANONICAL_PROTOCOLS.values()
}


def normalize_protocol_selector(value):
    if value is None:
        return "legacy"
    return str(value).strip().lower().replace("-", "_").replace("+", "_plus_")


def is_canonical_protocol_selector(value):
    return normalize_protocol_selector(value) in CANONICAL_PROTOCOL_CHOICES


def protocol_ids_for_selector(value):
    selector = normalize_protocol_selector(value)
    if selector == "all":
        return [metadata["protocol_id"] for metadata in CANONICAL_PROTOCOLS.values()]
    if selector in CANONICAL_PROTOCOLS:
        return [CANONICAL_PROTOCOLS[selector]["protocol_id"]]
    return []


def protocol_ids_from_legacy_mode(evaluation_mode, run_tr_tr=False):
    selected = []
    if run_tr_tr:
        selected.append("TR_TR")
    mode = str(evaluation_mode or "both").lower()
    if mode in {"tr_ts", "both", "all"}:
        selected.append("TR_TS")
    if mode in {"ts_tr", "both", "all"}:
        selected.append("TS_TR")
    if mode in {"tr_ts_tr", "all"}:
        selected.append("TR_PLUS_TS_TR")
    return selected


def selected_protocol_ids(arguments):
    protocol = getattr(arguments, "evaluation_protocol", "legacy")
    if is_canonical_protocol_selector(protocol):
        return protocol_ids_for_selector(protocol)
    return protocol_ids_from_legacy_mode(
        getattr(arguments, "evaluation_mode", "both"),
        run_tr_tr=bool(getattr(arguments, "run_tr_tr", False)),
    )


def legacy_name_for_protocol_id(protocol_id):
    for metadata in CANONICAL_PROTOCOLS.values():
        if metadata["protocol_id"] == protocol_id:
            return metadata["legacy_name"]
    return None


def protocol_metadata(protocol_id):
    for metadata in CANONICAL_PROTOCOLS.values():
        if metadata["protocol_id"] == protocol_id:
            return dict(metadata)
    raise KeyError(protocol_id)
