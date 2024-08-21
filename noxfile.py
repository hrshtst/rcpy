from __future__ import annotations

import os
from pathlib import Path

import nox

nox.needs_version = ">=2024.4.15"
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.sessions = ["fetch", "tests", "lint"]

PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]


# https://gist.github.com/MtkN1/41f36491dbf7162043d89d73a9d87dec
def ensurepath() -> None:
    rye_home = os.getenv("RYE_HOME")
    rye_py = Path(rye_home) if rye_home else Path.home() / ".rye" / "py"

    for py_dir in rye_py.iterdir():
        bin_dir = py_dir / "bin"
        os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"


@nox.session()
def fetch(session: nox.Session) -> None:
    for require_version in PYTHON_VERSIONS:
        session.run("rye", "fetch", require_version, external=True)
    ensurepath()


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    session.install("-r", "requirements-dev.lock")
    session.run("ruff", "check", *session.posargs)


@nox.session(
    python=PYTHON_VERSIONS,
    reuse_venv=True,
)
def tests(session: nox.Session) -> None:
    session.install("-r", "requirements-dev.lock")
    session.run("pytest", *session.posargs)


# Local Variables:
# jinx-local-words: "dev pytest uv virtualenv"
# End:
