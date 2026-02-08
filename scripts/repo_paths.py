from pathlib import Path

def get_repo_paths():
    """
    Returns a dictionary containing absolute paths to key directories.
    Based on the structure:
    repo_root/
      ├── src/
      ├── notebooks/
      └── dataset/
    """
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent
    repo_root = src_dir.parent
    data_dir = repo_root / "dataset"
    notebooks_dir = repo_root / "notebooks"
    config_dir = repo_root / "config"
    runs_dir = repo_root / "runs"

    # Folder names
    patches_root_name = "patches"
    pos_class_name = "positive"
    neg_class_name = "negative"

    return {
        # System Paths
        "repo_root": repo_root,
        "src_dir": src_dir,
        "data_dir": data_dir,
        "config_dir": config_dir,
        "notebooks_dir": notebooks_dir,
        "runs_dir": runs_dir,

        # Source Images (Full sized)
        "train_images": data_dir / "full_sized" / "images",
        "train_labels": data_dir / "full_sized" / "labels",
        "backgrounds": data_dir / "backgrounds",

        # Patch Output Directories
        "patches_dir": data_dir / patches_root_name,
        "patches_pos": data_dir / patches_root_name / pos_class_name,
        "patches_neg": data_dir / patches_root_name / neg_class_name,

        # Class names
        "class_name_pos": pos_class_name,
        "class_name_neg": neg_class_name

    }

if __name__ == "__main__":
    # Quick test to verify paths when running this file directly
    paths = get_repo_paths()
    print("Repository Paths Configuration:")
    for key, value in paths.items():
        print(f"{key:<15}: {value}")