"""Configuration objects for the Spotify hybrid recommender project."""

from dataclasses import dataclass
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback depends on local environment
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        """Fallback no-op when python-dotenv is not installed."""

        return False


@dataclass(slots=True)
class ProjectSettings:
    """Store reusable configuration values for local development.

    Attributes:
        project_root: Root directory of the repository.
        raw_data_dir: Directory used for untouched source data.
        interim_data_dir: Directory used for partially processed data.
        processed_data_dir: Directory used for model-ready datasets.
        artifacts_dir: Directory used for model outputs and reports.
        spotify_client_id: Optional Spotify client ID from the environment.
        spotify_client_secret: Optional Spotify client secret from the environment.
        spotify_redirect_uri: Optional redirect URI for OAuth flows.
        spotify_oauth_scopes: OAuth scopes requested during PKCE login.
        spotify_api_base_url: Base URL for Spotify Web API resources.
        spotify_accounts_base_url: Base URL for Spotify authentication.
        spotify_request_timeout_seconds: Timeout in seconds for Spotify requests.
        spotify_default_market: Default Spotify market used by future fetches.
        default_recommendation_count: Default number of recommendations to return.
    """

    project_root: Path
    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path
    artifacts_dir: Path
    spotify_client_id: str
    spotify_client_secret: str
    spotify_redirect_uri: str
    spotify_api_base_url: str
    spotify_accounts_base_url: str
    spotify_request_timeout_seconds: int
    spotify_default_market: str
    spotify_oauth_scopes: tuple[str, ...] = ("user-read-recently-played",)
    default_recommendation_count: int = 10

    @classmethod
    def from_env(cls) -> "ProjectSettings":
        """Build settings from the current repository layout and environment."""

        # The settings object computes paths once so other modules do not
        # need to duplicate path-building logic.
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(dotenv_path=project_root / ".env")
        return cls(
            project_root=project_root,
            raw_data_dir=project_root / "data" / "raw",
            interim_data_dir=project_root / "data" / "interim",
            processed_data_dir=project_root / "data" / "processed",
            artifacts_dir=project_root / "artifacts",
            spotify_client_id=os.getenv("SPOTIFY_CLIENT_ID", ""),
            spotify_client_secret=os.getenv("SPOTIFY_CLIENT_SECRET", ""),
            spotify_redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", ""),
            spotify_oauth_scopes=cls._parse_scope_list(
                os.getenv("SPOTIFY_OAUTH_SCOPES", "user-read-recently-played")
            ),
            spotify_api_base_url=os.getenv("SPOTIFY_API_BASE_URL", "https://api.spotify.com/v1"),
            spotify_accounts_base_url=os.getenv(
                "SPOTIFY_ACCOUNTS_BASE_URL",
                "https://accounts.spotify.com",
            ),
            spotify_request_timeout_seconds=int(os.getenv("SPOTIFY_REQUEST_TIMEOUT_SECONDS", "30")),
            spotify_default_market=os.getenv("SPOTIFY_DEFAULT_MARKET", "US"),
        )

    def ensure_project_directories(self) -> None:
        """Create key project directories when they do not yet exist."""

        # Creating directories centrally makes local setup more predictable
        # across scripts, notebooks, tests, and the Streamlit demo.
        for directory in (
            self.raw_data_dir,
            self.interim_data_dir,
            self.processed_data_dir,
            self.artifacts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def spotify_credentials_available(self) -> bool:
        """Return whether Spotify credentials are available in the environment."""

        return all([self.spotify_client_id, self.spotify_client_secret])

    def spotify_oauth_available(self) -> bool:
        """Return whether the app has the minimum PKCE OAuth configuration."""

        return all([self.spotify_client_id, self.spotify_redirect_uri])

    @staticmethod
    def _parse_scope_list(raw_scope_value: str) -> tuple[str, ...]:
        """Normalize a comma- or space-delimited scope string into a tuple."""

        normalized_value = raw_scope_value.replace(",", " ")
        scopes = tuple(scope.strip() for scope in normalized_value.split() if scope.strip())
        if scopes:
            return scopes
        return ("user-read-recently-played",)
