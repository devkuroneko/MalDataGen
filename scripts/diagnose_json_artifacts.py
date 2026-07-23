#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Diagnose malformed JSON artifacts and optionally move them aside."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Engine.DataIO.JsonIO import ResultArtifactCorruptionError
from Engine.DataIO.JsonIO import ResultArtifactMissingError
from Engine.DataIO.JsonIO import load_json_file
from Engine.DataIO.JsonIO import move_corrupt_json


def diagnose(path, *, move=False):
    path = Path(path)
    if move:
        return move_corrupt_json(path)
    try:
        load_json_file(path)
        return {"path": str(path), "status": "valid", "reason": None, "moved_to": None}
    except ResultArtifactMissingError as error:
        return {"path": str(path), "status": "missing", "reason": str(error), "moved_to": None}
    except ResultArtifactCorruptionError as error:
        return {"path": str(path), "status": "corrupt", "reason": str(error.original_error), "moved_to": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="JSON artifact paths to diagnose.")
    parser.add_argument("--move-corrupt", action="store_true", help="Move malformed JSON files to *.corrupt.")
    args = parser.parse_args(argv)
    payload = [diagnose(path, move=args.move_corrupt) for path in args.paths]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if any(item["status"] == "corrupt" for item in payload) else 0


if __name__ == "__main__":
    raise SystemExit(main())
