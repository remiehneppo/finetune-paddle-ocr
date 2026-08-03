#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./download_pretrained_models.sh [all|rec|det] [output-directory]

Download official PP-OCRv6 medium training checkpoints (.pdparams).

Examples:
  ./download_pretrained_models.sh all
  ./download_pretrained_models.sh rec
  ./download_pretrained_models.sh det /data/paddle-models
EOF
}

case "${1:-all}" in
  -h|--help)
    usage
    exit 0
    ;;
  all|rec|det)
    selection="${1:-all}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if (( $# > 2 )); then
  usage >&2
  exit 2
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${2:-${script_dir}/models}"
mkdir -p -- "${output_dir}"

download() {
  local filename="$1"
  local url="$2"
  local destination="${output_dir}/${filename}"
  local temporary="${destination}.part"

  if [[ -s "${destination}" ]]; then
    echo "SKIP ${destination} (already exists)"
    return
  fi

  echo "DOWNLOAD ${filename}"
  curl --fail --location --retry 3 --retry-all-errors \
    --output "${temporary}" "${url}"
  if [[ ! -s "${temporary}" ]]; then
    echo "ERROR: downloaded file is empty: ${temporary}" >&2
    exit 2
  fi
  mv -- "${temporary}" "${destination}"
  echo "OK ${destination}"
}

if [[ "${selection}" == "all" || "${selection}" == "rec" ]]; then
  download \
    "PP-OCRv6_medium_rec_pretrained.pdparams" \
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams"
fi

if [[ "${selection}" == "all" || "${selection}" == "det" ]]; then
  download \
    "PP-OCRv6_medium_det_pretrained.pdparams" \
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams"
fi
