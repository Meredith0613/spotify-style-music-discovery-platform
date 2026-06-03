"""Utility helpers for reading and writing local project files."""

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file and return its dictionary payload.

    Args:
        path: Filesystem path to the JSON document.

    Returns:
        The parsed JSON payload as a dictionary.
    """

    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}, but found {type(payload).__name__}.")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dictionary payload to a JSON file.

    Args:
        path: Destination filesystem path for the JSON document.
        payload: Dictionary payload to serialize.
    """

    # Creating parent directories here keeps collection code focused on
    # business workflow orchestration rather than filesystem setup.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Centralizing JSON IO makes it easier to add validation or richer
    # serialization behavior later without touching business modules.
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
