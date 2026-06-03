"""CLI entrypoint that demonstrates Spotify data collection and preprocessing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

# Adding the repository's src directory to `sys.path` allows the CLI module
# to run directly from the repo root before editable installation.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.settings import ProjectSettings
from data.data_collector import DataCollector
from data.preprocessor import Preprocessor
from data.spotify_client import (
    SpotifyAPIClient,
    SpotifyAPIClientError,
    SpotifyAuthenticationError,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the sample data collection flow.

    Returns:
        Configured argument parser for the sample CLI.
    """

    parser = argparse.ArgumentParser(
        description="Collect a sample Spotify playlist bundle and preprocess it.",
    )
    parser.add_argument(
        "--playlist-id",
        required=True,
        help="Spotify playlist ID to collect.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix used for saved raw and processed files.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Collect raw payloads only and skip DataFrame normalization.",
    )
    return parser


def main() -> None:
    """Run the sample Spotify collection workflow from the command line."""

    parser = build_argument_parser()
    arguments = parser.parse_args()

    settings = ProjectSettings.from_env()
    client = SpotifyAPIClient.from_settings(settings)
    collector = DataCollector(client=client, settings=settings)
    preprocessor = Preprocessor(settings=settings)

    try:
        raw_paths = collector.collect_playlist_bundle(
            playlist_id=arguments.playlist_id,
            output_prefix=arguments.output_prefix,
        )
    except SpotifyAuthenticationError as error:
        parser.exit(
            status=1,
            message=f"{error}\n",
        )
    except SpotifyAPIClientError as error:
        parser.exit(status=1, message=f"{error}\n")

    print("Raw files saved:")
    _print_saved_paths(asdict(raw_paths))

    if arguments.skip_preprocess:
        return

    processed_paths = preprocessor.preprocess_collection_bundle(
        raw_paths=raw_paths,
        output_prefix=arguments.output_prefix or arguments.playlist_id,
    )
    print("Processed files saved:")
    _print_saved_paths(asdict(processed_paths))


def _print_saved_paths(path_mapping: dict[str, Any]) -> None:
    """Print a human-readable mapping of generated artifact paths.

    Args:
        path_mapping: Mapping of logical labels to raw or processed file paths.
    """

    for label, path in path_mapping.items():
        if path is not None:
            print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
