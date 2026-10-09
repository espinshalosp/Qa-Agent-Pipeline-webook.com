"""
config/settings.py
==================
Centralised settings for the QA Agent Platform.

All values are loaded from:
  1. Environment variables (highest priority)
  2. .env file in the project root
  3. Defaults defined in this file

Usage
-----
    from config.settings import get_settings
    settings = get_settings()
    print(settings.jira_base_url)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env path relative to this file's location (project root/config/../.env)
_PROJECT_ROOT = Path(__file__).parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables / .env file.
    Field names map directly to environment variable names (case-insensitive).
    E.g., field `jira_base_url` loads from env var `JIRA_BASE_URL`.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --------------------------------------------------------------------------
    # Jira — Connection
    # --------------------------------------------------------------------------
    jira_base_url: str = Field(..., description="Jira instance base URL (no trailing slash)")
    jira_email: str = Field(..., description="Jira account email address")
    jira_api_token: str = Field(..., description="Jira API token")

    # --------------------------------------------------------------------------
    # Jira — Query
    # --------------------------------------------------------------------------
    jira_jql: str = Field(
        default='status = "Ready for QA"',
        description="JQL query used to discover QA tickets",
    )
    jira_project_key: str = Field(default="PROJ", description="Jira project key")
    jira_max_results_per_page: int = Field(default=100, description="Page size for JQL search")
    jira_acceptance_criteria_field: str = Field(
        default="customfield_10045",
        description="Custom field ID storing acceptance criteria (instance-specific)",
    )

    # --------------------------------------------------------------------------
    # Jira — Retry
    # --------------------------------------------------------------------------
    jira_max_retries: int = Field(default=3)
    jira_initial_delay_seconds: float = Field(default=1.0)
    jira_backoff_factor: int = Field(default=2)

    # --------------------------------------------------------------------------
    # Pipeline — General
    # --------------------------------------------------------------------------
    log_level: str = Field(default="INFO")
    dry_run: bool = Field(default=False)
    pipeline_name: str = Field(default="QA Agent Platform")

    # --------------------------------------------------------------------------
    # Pipeline — Paths (resolved against project root)
    # --------------------------------------------------------------------------
    artifact_dir: Path = Field(default=_PROJECT_ROOT / "artifacts")
    report_dir: Path = Field(default=_PROJECT_ROOT / "artifacts" / "reports")
    screenshot_dir: Path = Field(default=_PROJECT_ROOT / "artifacts" / "screenshots")
    log_dir: Path = Field(default=_PROJECT_ROOT / "logs")
    attachment_dir: Path = Field(default=_PROJECT_ROOT / "artifacts" / "attachments")
    execution_dir: Path = Field(default=_PROJECT_ROOT / "artifacts" / "execution")
    defect_dir: Path = Field(default=_PROJECT_ROOT / "artifacts" / "defects")

    # --------------------------------------------------------------------------
    # Coverage Thresholds
    # --------------------------------------------------------------------------
    coverage_minimum_overall: float = Field(default=0.80)
    coverage_critical_requirements: float = Field(default=1.0)

    # --------------------------------------------------------------------------
    # LLM (for future Agents 2-7)
    # --------------------------------------------------------------------------
    llm_provider: str | None = Field(default=None)
    llm_api_key: str | None = Field(default=None)
    llm_model: str | None = Field(default=None)
    llm_temperature: float = Field(default=0.2)
    llm_max_tokens: int = Field(default=4096)
    llm_vision_prompt: str = Field(
        default="Describe the layout, buttons, fields, elements, and overall UI intent of this image for a QA test generation agent. Focus on structural details, validations, or functional flows represented.",
        description="Default prompt for LLM vision analysis of attachments"
    )

    # --------------------------------------------------------------------------
    # Application Configuration
    # --------------------------------------------------------------------------
    application_name: str = Field(default="Webook", description="Application name")
    application_base_url: str = Field(default="https://webook.com", description="Application base URL")

    # --------------------------------------------------------------------------
    # Allure Report Auto-open Configuration
    # --------------------------------------------------------------------------
    allure_auto_open: bool = Field(default=False, description="Automatically open Allure report after execution")
    allure_open_browser: str = Field(default="default", description="Browser to use for opening ('default' or 'chrome')")

    # --------------------------------------------------------------------------
    # Validators
    # --------------------------------------------------------------------------
    @field_validator("jira_base_url", "application_base_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/") if v else v


    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    def ensure_dirs(self) -> None:
        """Create all required output directories if they do not exist."""
        for path in (
            self.artifact_dir,
            self.report_dir,
            self.screenshot_dir,
            self.log_dir,
            self.attachment_dir,
            self.execution_dir,
            self.defect_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.
    Raises ValidationError on first call if required env vars are missing.
    """
    return Settings()
