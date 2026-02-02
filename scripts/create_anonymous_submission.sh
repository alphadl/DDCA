#!/usr/bin/env bash
# Create a zip archive of the repository for anonymous submission (e.g. ACL 2026).
# Excludes .git and common non-source files so reviewers get a clean code snapshot.
# The archive has one top-level folder so unzipping yields ./efficient_reason_DCA/...

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(dirname "$REPO_ROOT")"
FOLDER_NAME="$(basename "$REPO_ROOT")"
OUT_NAME="$REPO_ROOT/efficient_reason_DCA_submission.zip"

# Remove previous archive if present
rm -f "$OUT_NAME"

# Create zip from parent so the archive contains one root folder (efficient_reason_DCA)
cd "$PARENT"
zip -r "$OUT_NAME" "$FOLDER_NAME" \
  -x "$FOLDER_NAME/.git/*" \
  -x "$FOLDER_NAME/*.pyc" \
  -x "$FOLDER_NAME/*__pycache__*" \
  -x "$FOLDER_NAME/*.zip" \
  -x "$FOLDER_NAME/.DS_Store" \
  -x "$FOLDER_NAME/*.egg-info/*" \
  -x "$FOLDER_NAME/.eggs/*"

echo "Created $OUT_NAME ($(du -h "$OUT_NAME" | cut -f1))"
