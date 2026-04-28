#!/usr/bin/env bash
# download_data.sh — fetch PTB-XL v1.0.3 from the PhysioNet open-data S3 mirror.
#
# Requirements:
#   - AWS CLI v2 installed (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
#   - No AWS account or credentials needed: --no-sign-request enables anonymous access.
#
# Usage:
#   bash scripts/download_data.sh            # default: data/ptbxl/
#   DATA_DIR=/custom/path bash scripts/download_data.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/ptbxl}"
S3_URI="s3://physionet-open/ptb-xl/1.0.3/"

echo ">>> Checking AWS CLI..."
if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI not found. Install AWS CLI v2 and re-run." >&2
  exit 1
fi

echo ">>> Target directory: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"

echo ">>> Syncing PTB-XL v1.0.3 from ${S3_URI}"
echo "    (anonymous access, ~3 GB, may take several minutes)"

aws s3 sync \
  --no-sign-request \
  --only-show-errors \
  "${S3_URI}" \
  "${DATA_DIR}/"

echo ">>> Verifying expected files..."
REQUIRED=(
  "ptbxl_database.csv"
  "scp_statements.csv"
  "records100"
  "records500"
)
missing=0
for item in "${REQUIRED[@]}"; do
  if [[ ! -e "${DATA_DIR}/${item}" ]]; then
    echo "  MISSING: ${DATA_DIR}/${item}"
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "ERROR: One or more expected PTB-XL files are missing after sync." >&2
  exit 1
fi

echo ">>> Done. PTB-XL is available at: ${DATA_DIR}"
