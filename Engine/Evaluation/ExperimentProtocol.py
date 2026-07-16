#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Canonical AppClassNet experimental protocol helpers."""

from __future__ import annotations

from dataclasses import dataclass

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
PROTOCOL_ID_TO_RESULT_KEY = {
    metadata["protocol_id"]: metadata["legacy_name"]
    for metadata in CANONICAL_PROTOCOLS.values()
}
RESULT_KEY_TO_PROTOCOL_ID = {
    metadata["legacy_name"]: metadata["protocol_id"]
    for metadata in CANONICAL_PROTOCOLS.values()
}
SYNTHETIC_RESULT_KEYS = ("TR-TS", "TS-TR", "TR+TS-TR")
LEGACY_EVALUATION_NAMES = {
    "TR-TS": "TrAs",
    "TS-TR": "TsAr",
}
RESULT_KEY_ALIASES = {
    "trtr": "TR-TR",
    "tr_tr": "TR-TR",
    "tr-tr": "TR-TR",
    "TR_TR": "TR-TR",
    "TR-TR": "TR-TR",
    "trts": "TR-TS",
    "tr_ts": "TR-TS",
    "tr-ts": "TR-TS",
    "TR_TS": "TR-TS",
    "TR-TS": "TR-TS",
    "tras": "TR-TS",
    "TrAs": "TR-TS",
    "tstr": "TS-TR",
    "ts_tr": "TS-TR",
    "ts-tr": "TS-TR",
    "TS_TR": "TS-TR",
    "TS-TR": "TS-TR",
    "tsar": "TS-TR",
    "TsAr": "TS-TR",
    "trpluststr": "TR+TS-TR",
    "tr_plus_ts_tr": "TR+TS-TR",
    "tr+ts-tr": "TR+TS-TR",
    "TR_PLUS_TS_TR": "TR+TS-TR",
    "TR+TS-TR": "TR+TS-TR",
}


@dataclass(frozen=True)
class EvaluationProtocolPlan:
    protocol: str
    run_tr_tr: bool
    run_tr_ts: bool
    run_ts_tr: bool
    run_tr_plus_ts_tr: bool
    required_result_keys: tuple[str, ...]

    @property
    def active_result_keys(self):
        keys = []
        if self.run_tr_tr:
            keys.append("TR-TR")
        if self.run_tr_ts:
            keys.append("TR-TS")
        if self.run_ts_tr:
            keys.append("TS-TR")
        if self.run_tr_plus_ts_tr:
            keys.append("TR+TS-TR")
        return tuple(keys)

    @property
    def active_protocol_ids(self):
        return tuple(
            RESULT_KEY_TO_PROTOCOL_ID[key]
            for key in self.active_result_keys
        )

    @property
    def legacy_evaluation(self):
        return tuple(
            LEGACY_EVALUATION_NAMES[key]
            for key in self.required_result_keys
            if key in LEGACY_EVALUATION_NAMES
        )

    def to_dict(self):
        payload = {
            "protocol": self.protocol,
            "run_tr_tr": bool(self.run_tr_tr),
            "run_tr_ts": bool(self.run_tr_ts),
            "run_ts_tr": bool(self.run_ts_tr),
            "run_tr_plus_ts_tr": bool(self.run_tr_plus_ts_tr),
            "required_result_keys": list(self.required_result_keys),
        }
        return payload

    def as_dict(self):
        payload = self.to_dict()
        payload["active_result_keys"] = list(self.active_result_keys)
        payload["active_protocol_ids"] = list(self.active_protocol_ids)
        payload["legacy_evaluation"] = list(self.legacy_evaluation)
        return payload


def normalize_protocol_selector(value):
    if value is None:
        return "legacy"
    return str(value).strip().lower().replace("-", "_").replace("+", "_plus_")


def is_canonical_protocol_selector(value):
    return normalize_protocol_selector(value) in CANONICAL_PROTOCOL_CHOICES


def _normalize_result_key_token(value):
    return str(value or "").strip().replace(" ", "").replace("+", "plus").lower()


def canonical_result_key(value):
    if value in RESULT_KEY_TO_PROTOCOL_ID:
        return value
    token = _normalize_result_key_token(value)
    if token in RESULT_KEY_ALIASES:
        return RESULT_KEY_ALIASES[token]
    selector = normalize_protocol_selector(value)
    if selector in CANONICAL_PROTOCOLS:
        return CANONICAL_PROTOCOLS[selector]["legacy_name"]
    return value


def normalize_results_keys(results):
    if not isinstance(results, dict):
        return results
    normalized = {}
    for key, value in results.items():
        canonical_key = canonical_result_key(key)
        if canonical_key in normalized and isinstance(normalized[canonical_key], dict) and isinstance(value, dict):
            normalized[canonical_key].update(value)
        elif canonical_key in normalized:
            continue
        else:
            normalized[canonical_key] = value
    return normalized


def protocol_ids_for_selector(value):
    selector = normalize_protocol_selector(value)
    if selector == "all":
        return [metadata["protocol_id"] for metadata in CANONICAL_PROTOCOLS.values()]
    if selector in CANONICAL_PROTOCOLS:
        return [CANONICAL_PROTOCOLS[selector]["protocol_id"]]
    return []


def protocol_ids_from_legacy_mode(evaluation_mode, run_tr_tr=False):
    plan = resolve_evaluation_protocol_plan_from_values(
        protocol="legacy",
        evaluation_mode=evaluation_mode,
        run_tr_tr=run_tr_tr,
    )
    return list(plan.active_protocol_ids)


def selected_protocol_ids(arguments):
    return list(resolve_evaluation_protocol_plan(arguments).active_protocol_ids)


def _required_keys_for_flags(run_tr_ts, run_ts_tr, run_tr_plus_ts_tr):
    keys = []
    if run_tr_ts:
        keys.append("TR-TS")
    if run_ts_tr:
        keys.append("TS-TR")
    if run_tr_plus_ts_tr:
        keys.append("TR+TS-TR")
    return tuple(keys)


def resolve_evaluation_protocol_plan_from_values(protocol="legacy", evaluation_mode="both", run_tr_tr=False):
    selector = normalize_protocol_selector(protocol)
    mode = str(evaluation_mode or "both").strip().lower()

    if selector in CANONICAL_PROTOCOLS or selector == "all":
        active_ids = set(protocol_ids_for_selector(selector))
        plan_protocol = selector
        return EvaluationProtocolPlan(
            protocol=plan_protocol,
            run_tr_tr="TR_TR" in active_ids,
            run_tr_ts="TR_TS" in active_ids,
            run_ts_tr="TS_TR" in active_ids,
            run_tr_plus_ts_tr="TR_PLUS_TS_TR" in active_ids,
            required_result_keys=_required_keys_for_flags(
                "TR_TS" in active_ids,
                "TS_TR" in active_ids,
                "TR_PLUS_TS_TR" in active_ids,
            ),
        )

    run_tr_ts = mode in {"tr_ts", "both", "all"}
    run_ts_tr = mode in {"ts_tr", "both", "all"}
    run_tr_plus_ts_tr = mode in {"tr_ts_tr", "all"}
    run_real_real = bool(run_tr_tr) or mode == "none"
    if mode in {"tr_ts", "ts_tr", "tr_ts_tr", "both", "all"}:
        plan_protocol = mode if mode != "tr_ts_tr" else "tr_plus_ts_tr"
    elif run_real_real:
        plan_protocol = "tr_tr"
    else:
        plan_protocol = "legacy"
    return EvaluationProtocolPlan(
        protocol=plan_protocol,
        run_tr_tr=run_real_real,
        run_tr_ts=run_tr_ts,
        run_ts_tr=run_ts_tr,
        run_tr_plus_ts_tr=run_tr_plus_ts_tr,
        required_result_keys=_required_keys_for_flags(run_tr_ts, run_ts_tr, run_tr_plus_ts_tr),
    )


def resolve_evaluation_protocol_plan(arguments):
    existing = getattr(arguments, "evaluation_protocol_plan", None)
    if isinstance(existing, EvaluationProtocolPlan):
        return existing
    return resolve_evaluation_protocol_plan_from_values(
        protocol=getattr(arguments, "evaluation_protocol", "legacy"),
        evaluation_mode=getattr(arguments, "evaluation_mode", "both"),
        run_tr_tr=bool(
            getattr(arguments, "run_tr_tr", False)
            or getattr(arguments, "run_tr_tr_effective", False)
        ),
    )


def legacy_evaluation_for_plan(plan):
    if not isinstance(plan, EvaluationProtocolPlan):
        plan = resolve_evaluation_protocol_plan(plan)
    return list(plan.legacy_evaluation)


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
