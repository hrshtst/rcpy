# ruff: noqa: D100
from __future__ import annotations

import nox

nox.needs_version = ">=2024.10.9"
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["lint", "tests"]

PYPROJECT = nox.project.load_toml("pyproject.toml")

ALL_PYTHONS = [
    c.split()[-1] for c in PYPROJECT["project"]["classifiers"] if c.startswith("Programming Language :: Python :: 3.")
]


@nox.session(python="3.12", reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run ruff linting."""
    session.install("ruff")
    session.run("ruff", "check", *session.posargs)


@nox.session(python=ALL_PYTHONS, reuse_venv=True)
def tests(session: nox.Session) -> None:
    """Run test suite with pytest."""
    session.install(*PYPROJECT["dependency-groups"]["dev"], "uv")
    session.install("-e.")
    session.run("pytest", *session.posargs)


# Local Variables:
# jinx-local-words: "dev noqa pyproject pytest uv virtualenv"
# End:
