"""Tests for the artifact layout.

The project must work for anyone who hands over an export, so nothing may be
keyed to one particular person. These tests pin the split: per-user tables are
isolated from each other, and film-level facts stay shared.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lbrec.config import DEFAULT_USER, Settings, normalize_user


def make_settings(tmp_path: Path, *, default_user: str = DEFAULT_USER) -> Settings:
    return Settings(
        export_dir=tmp_path / "Data",
        artifacts_dir=tmp_path / "artifacts",
        tmdb_token=None,
        tmdb_api_key=None,
        tmdb_rps=1.0,
        default_user=default_user,
    )


@pytest.mark.parametrize("label", ["me", "volodya", "friend-1", "a_b", "x" * 64])
def test_valid_user_labels_are_accepted(label: str):
    assert normalize_user(label) == label.lower()


def test_user_labels_are_case_folded():
    """Otherwise 'Alice' and 'alice' become two directories for one person."""
    assert normalize_user("Alice") == "alice"


@pytest.mark.parametrize("label", ["", "..", "../etc", "a/b", "-lead", "x" * 65, "sp ace"])
def test_unsafe_user_labels_are_rejected(label: str):
    """The label becomes a directory name, so it must not escape the tree."""
    with pytest.raises(ValueError, match="invalid user label"):
        normalize_user(label)


def test_each_user_gets_their_own_tables(tmp_path: Path):
    settings = make_settings(tmp_path)
    mine = settings.user("me")
    theirs = settings.user("friend")
    assert mine.films_local != theirs.films_local
    assert mine.interactions != theirs.interactions
    assert mine.film_status != theirs.film_status
    assert theirs.root.name == "friend"


def test_film_level_tables_are_shared_across_users(tmp_path: Path):
    """A film's identity does not depend on who watched it."""
    settings = make_settings(tmp_path)
    assert "users" not in settings.film_map_path.parts
    assert "users" not in settings.films_tmdb_path.parts
    assert "users" not in settings.overrides_path.parts
    assert "users" not in settings.movielens_dir.parts


def test_default_user_is_configurable(tmp_path: Path):
    assert make_settings(tmp_path).user().user == DEFAULT_USER
    assert make_settings(tmp_path, default_user="volodya").user().user == "volodya"


def test_known_users_lists_only_ingested_exports(tmp_path: Path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    assert settings.known_users() == []

    for name in ("me", "friend"):
        paths = settings.user(name).ensure()
        paths.films_local.write_bytes(b"")
    # A directory without the table is not an ingested export.
    settings.user("halfway").ensure()

    assert settings.known_users() == ["friend", "me"]
