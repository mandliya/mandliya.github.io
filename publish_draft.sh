#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./publish_draft.sh <draft_slug_or_filename>

Examples:
  ./publish_draft.sh my-post
  ./publish_draft.sh my-post.md
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAFTS_DIR="${SCRIPT_DIR}/_drafts"
POSTS_DIR="${SCRIPT_DIR}/_posts"

input_name="$1"
slug="${input_name%.md}"

draft_file="${DRAFTS_DIR}/${slug}.md"
post_file="${POSTS_DIR}/$(date +%Y-%m-%d)-${slug}.md"

if [[ ! -f "$draft_file" ]]; then
  echo "Error: draft not found: $draft_file"
  exit 1
fi

mkdir -p "$POSTS_DIR"

if [[ -f "$post_file" ]]; then
  echo "Error: post already exists: $post_file"
  exit 1
fi

mv "$draft_file" "$post_file"
echo "Published draft: $post_file"
