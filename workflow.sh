#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOKS_PATH="${SCRIPT_DIR}/_notebooks"
DRAFTS_PATH="${SCRIPT_DIR}/_drafts"
POSTS_PATH="${SCRIPT_DIR}/_posts"

usage() {
  cat <<'EOF'
Usage: ./workflow.sh {new|convert|review|publish} [args]

Commands:
  new [name]                    Open Jupyter Lab in _notebooks (optionally with <name>.ipynb)
  convert <source> [output]     Convert notebook source to a draft
  review                         Start Jekyll server with drafts
  publish <draft_slug_or_file>  Publish a draft into _posts
EOF
}

ensure_directories() {
  mkdir -p "$NOTEBOOKS_PATH" "$DRAFTS_PATH" "$POSTS_PATH"
}

create_notebook() {
  local notebook_name="${1:-}"
  if ! command -v jupyter >/dev/null 2>&1; then
    echo "Error: jupyter is not installed or not on PATH."
    exit 1
  fi

  cd "$NOTEBOOKS_PATH"
  if [[ -n "$notebook_name" ]]; then
    jupyter lab "${notebook_name%.ipynb}.ipynb"
  else
    jupyter lab
  fi
}

convert_notebook() {
  local notebook_source="${1:-}"
  local output_name="${2:-}"

  if [[ -z "$notebook_source" ]]; then
    read -r -p "Enter notebook source (path/url/repo): " notebook_source
  fi

  if [[ -n "$output_name" ]]; then
    bash "$SCRIPT_DIR/notebook_to_draft.sh" "$notebook_source" "$output_name"
  else
    bash "$SCRIPT_DIR/notebook_to_draft.sh" "$notebook_source"
  fi
}

review_draft() {
  cd "$SCRIPT_DIR"
  bundle exec jekyll serve --drafts
}

publish_draft() {
  local draft_name="${1:-}"
  if [[ -z "$draft_name" ]]; then
    read -r -p "Enter draft slug or filename: " draft_name
  fi
  bash "$SCRIPT_DIR/publish_draft.sh" "$draft_name"
}

ensure_directories

case "${1:-}" in
  new)
    create_notebook "${2:-}"
    ;;
  convert)
    convert_notebook "${2:-}" "${3:-}"
    ;;
  review)
    review_draft
    ;;
  publish)
    publish_draft "${2:-}"
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
