from pathlib import Path
import shutil

# Root folder
root = Path("joblib")

# Step 1: Move all files up to root
for path in root.rglob("*"):
    if path.is_file():
        # Move to root, keeping only filename
        destination = root / path.name
        if destination.exists():
            print(f"Warning: {destination} already exists. Skipping {path}")
            continue
        print(f"Moving {path} -> {destination}")
        shutil.move(str(path), str(destination))

# Step 2: Remove empty folders
for path in sorted(root.glob('**/*'), key=lambda p: -len(str(p))):  # Sort deeper folders first
    if path.is_dir() and not any(path.iterdir()):
        print(f"Removing empty folder {path}")
        path.rmdir()
