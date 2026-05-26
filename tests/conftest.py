"""Shared test fixtures for Repository Manager."""

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("REPOSITORY_URL", "https://test.example.com")
    monkeypatch.setenv("REPOSITORY_TOKEN", "test-token-12345")
    monkeypatch.setenv("REPOSITORY_SSL_VERIFY", "False")
