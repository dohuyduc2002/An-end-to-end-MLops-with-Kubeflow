#!/bin/bash

set -e

# -----------------------------
# CONFIGURATION
# -----------------------------
DRIVE_FOLDER_ID="1HCoHY7N0GGCIqFouF3mx9cVKY35Z-p44"
MINIO_ALIAS="localMinio"
MINIO_BUCKET="sample-data"
MINIO_TARGET_PATH="data"
LOCAL_DATA_PATH="data"

# -----------------------------
# 1. Clean old data
# -----------------------------
echo "🧹 Cleaning local & remote DVC data..."
rm -rf "$LOCAL_DATA_PATH" .dvc/cache
mc rm --recursive --force "$MINIO_ALIAS/$MINIO_BUCKET"

# -----------------------------
# 2. Download Google Drive folder using gdown
# -----------------------------
echo "⬇️ Downloading folder from Google Drive to '$LOCAL_DATA_PATH/'..."
gdown --folder "https://drive.google.com/drive/folders/$DRIVE_FOLDER_ID" -O "$LOCAL_DATA_PATH"

# -----------------------------
# 3. Upload to MinIO for permanent storage
# -----------------------------
echo "☁️ Uploading '$LOCAL_DATA_PATH/' to MinIO → $MINIO_BUCKET/$MINIO_TARGET_PATH/"
mc cp --recursive "$LOCAL_DATA_PATH" "$MINIO_ALIAS/$MINIO_BUCKET/$MINIO_TARGET_PATH/"

# -----------------------------
# 4. Track with DVC
# -----------------------------
echo "📦 Tracking with DVC..."
dvc add "$LOCAL_DATA_PATH"

# -----------------------------
# 5. Commit DVC metadata
# -----------------------------
echo "📝 Committing to Git..."
git add "$LOCAL_DATA_PATH.dvc" .gitignore
git commit -m "Fresh DVC tracking for Google Drive → MinIO sync"

# -----------------------------
# 6. Push to DVC remote (MinIO)
# -----------------------------
echo "☁️ Pushing tracked data to DVC remote..."
dvc push

# -----------------------------
# 7. Cleanup local folder
# -----------------------------
echo "🧹 Cleaning local data to save space..."
rm -rf "$LOCAL_DATA_PATH"

echo "✅ All done! Your data is now tracked and stored in MinIO via DVC."
