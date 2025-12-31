from pathlib import Path

def find_project_root(marker_file="pyproject.toml"):
    """
    Traverses up from the current file to find the directory containing the marker file.
    """
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker_file).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root containing '{marker_file}'")

# 1. Establish the reliable Root
ROOT_DIR = find_project_root()

# 2. Define standard directories relative to Root
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
SCRIPTS_DIR = ROOT_DIR / "scripts"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# 3. Path Resolver Helper
def resolve_path(path_str):
    """
    Smartly resolves a path from the config.
    - If it's absolute (e.g., "C:/data"), returns it as-is.
    - If it's relative (e.g., "data/real"), appends it to ROOT_DIR.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path