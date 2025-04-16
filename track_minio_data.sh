#!/bin/bash

# -----------------------------
# CONFIGURATION (edit if needed)
# -----------------------------
MINIO_ALIAS="localMinio"
MINIO_BUCKET="sample-data"
REMOTE_PATH="data"
LOCAL_PATH="data"

# -----------------------------
# 1. Download latest data from MinIO
# -----------------------------
echo "🔁 Syncing data from MinIO: s3://$MINIO_BUCKET/$REMOTE_PATH"
mc cp --recursive "$MINIO_ALIAS/$MINIO_BUCKET/$REMOTE_PATH" "$LOCAL_PATH"

# -----------------------------
# 2. Track with DVC
# -----------------------------
echo "📦 Tracking with DVC: $LOCAL_PATH"
dvc add "$LOCAL_PATH"

# -----------------------------
# 3. Commit metadata to Git
# -----------------------------
echo "📝 Committing DVC metadata to Git"
git add "$LOCAL_PATH.dvc" .gitignore
git commit -m "Track new version of '$REMOTE_PATH' from MinIO"

# -----------------------------
# 4. Push data to MinIO DVC remote
# -----------------------------
echo "☁️ Pushing DVC data to remote (MinIO)"
dvc push

# -----------------------------
# 5. Remove local copy to save space
# -----------------------------
echo "🧹 Removing local data to save space"
rm -rf "$LOCAL_PATH"

echo "✅ Done! Tracked and pushed '$REMOTE_PATH' via DVC, only MinIO stores the data now."

# Kubeflow → MinIO (sample-data/data/)
#             ↓
#     ./track_minio_data.sh
#             ↓
#  Git: data.dvc       MinIO: DVC cache