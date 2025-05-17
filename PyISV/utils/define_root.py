# Common project paths
from pathlib import Path

# Find the project root by searching upwards for a marker file or folder (e.g., .env)

# Find the project root by searching upwards for a marker file (e.g., .env)
def find_project_root(current: Path, marker: str = "setup.cfg") -> Path:
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Project root with marker '{marker}' not found.")

# Usage: always resolves to the folder containing the setup.cfg file
PROJECT_ROOT = find_project_root(Path(__file__).resolve(), marker="setup.cfg")

