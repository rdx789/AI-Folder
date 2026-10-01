"""Project layout and the one configurable NovaOps dataset root: DATA_PATH.

    maya/            PROJECT_DIR: .env, pyproject.toml, docs
      code/          CODE_DIR: the `maya` package, its tests and eval fixtures
      data/          default DATA_PATH (NovaOps documents and database)
      results/       RESULTS_DIR: eval reports, logs and replay artifacts

Precedence: environment variable DATA_PATH, then `DATA_PATH=` in Maya's .env, then the
legacy NOVAOPS_DATA_ROOT variable, then this project's `data/`. A relative value is
resolved against this project folder, not the current directory. Only this one value
is read from .env, so importing Maya still never loads credentials into the process.
"""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
ENV_FILE = PROJECT_DIR / '.env'
RESULTS_DIR = PROJECT_DIR / 'results'


def _dotenv_value(name):
    try:
        from dotenv import dotenv_values
    except ImportError:
        return None
    try:
        return dotenv_values(ENV_FILE).get(name)
    except (OSError, UnicodeDecodeError):
        return None  # an unreadable .env falls back to the default root


def data_path() -> Path:
    value = (os.environ.get('DATA_PATH') or _dotenv_value('DATA_PATH')
             or os.environ.get('NOVAOPS_DATA_ROOT'))
    path = Path(value.strip()).expanduser() if value and value.strip() else PROJECT_DIR / 'data'
    return path if path.is_absolute() else PROJECT_DIR / path


DATA_PATH = data_path()
