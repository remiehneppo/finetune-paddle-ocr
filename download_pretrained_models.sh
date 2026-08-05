#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./download_pretrained_models.sh [all|rec|det|vl] [output-directory] [--revision REV]

Download official PP-OCRv6 checkpoints and/or the complete PaddleOCR-VL-1.6 snapshot.

Examples:
  ./download_pretrained_models.sh all
  ./download_pretrained_models.sh rec
  ./download_pretrained_models.sh det /data/paddle-models
  ./download_pretrained_models.sh vl /data/paddle-models --revision main
EOF
}

case "${1:-all}" in
  -h|--help)
    usage
    exit 0
    ;;
  all|rec|det|vl)
    selection="${1:-all}"
    shift $(( $# > 0 ? 1 : 0 ))
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${script_dir}/models"
vl_revision="14e49e712dfa8ac6ff89aa88f1a21c8f30e0cf29"

if (( $# > 0 )) && [[ "${1}" != --* ]]; then
  output_dir="${1}"
  shift
fi
while (( $# > 0 )); do
  case "${1}" in
    --revision)
      if (( $# < 2 )); then
        echo "ERROR: --revision requires a value" >&2
        exit 2
      fi
      vl_revision="${2}"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: ${1}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

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
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required for PP-OCRv6 checkpoints" >&2
    exit 2
  fi
  download \
    "PP-OCRv6_medium_rec_pretrained.pdparams" \
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams"
fi

if [[ "${selection}" == "all" || "${selection}" == "det" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required for PP-OCRv6 checkpoints" >&2
    exit 2
  fi
  download \
    "PP-OCRv6_medium_det_pretrained.pdparams" \
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams"
fi

if [[ "${selection}" == "all" || "${selection}" == "vl" ]]; then
  if command -v hf >/dev/null 2>&1; then
    hf_cli=(hf download)
    resume_args=()
  elif command -v huggingface-cli >/dev/null 2>&1; then
    hf_cli=(huggingface-cli download)
    resume_args=(--resume-download)
  else
    echo "ERROR: hf or huggingface-cli is required for PaddleOCR-VL-1.6" >&2
    echo "Install it with: python -m pip install huggingface_hub" >&2
    exit 2
  fi
  vl_dir="${output_dir}/PaddleOCR-VL-1.6"
  echo "DOWNLOAD PaddlePaddle/PaddleOCR-VL-1.6 revision ${vl_revision}"
  "${hf_cli[@]}" \
    "PaddlePaddle/PaddleOCR-VL-1.6" \
    --local-dir "${vl_dir}" \
    --revision "${vl_revision}" \
    "${resume_args[@]}"
  if [[ ! -s "${vl_dir}/config.json" ]]; then
    echo "ERROR: PaddleOCR-VL-1.6 snapshot is incomplete: ${vl_dir}" >&2
    exit 2
  fi
  echo "OK ${vl_dir}"
fi
