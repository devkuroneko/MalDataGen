#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Small resource monitoring helpers with optional psutil support."""

from __future__ import annotations

import logging
import time

NOT_AVAILABLE = "not_available"

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None


def _round_mb(value):
    if value == NOT_AVAILABLE:
        return NOT_AVAILABLE
    return round(float(value), 3)


def get_current_memory_mb():
    """Return current process RSS in MB, or not_available."""
    if psutil is not None:
        return _round_mb(psutil.Process().memory_info().rss / (1024 * 1024))
    return NOT_AVAILABLE


def get_peak_memory_mb():
    """Return peak process memory in MB, or not_available."""
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports ru_maxrss in KiB. macOS reports bytes, but this project
        # targets Linux for the fallback path.
        return _round_mb(usage.ru_maxrss / 1024)

    if psutil is not None:
        return _round_mb(psutil.Process().memory_info().rss / (1024 * 1024))

    return NOT_AVAILABLE


class Timer:
    """Context manager that records elapsed seconds."""

    def __init__(self, stage_name=None):
        self.stage_name = stage_name
        self.start_time = None
        self.elapsed_seconds = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed_seconds = time.perf_counter() - self.start_time
        return False


def log_resource_usage(stage_name):
    """Log and return current/peak process memory for a stage."""
    current_memory_mb = get_current_memory_mb()
    peak_memory_mb = get_peak_memory_mb()
    logging.info(
        "Resource usage [%s]: current_memory_mb=%s peak_memory_mb=%s",
        stage_name,
        current_memory_mb,
        peak_memory_mb,
    )
    return {
        "current_memory_mb": current_memory_mb,
        "peak_memory_mb": peak_memory_mb,
    }
