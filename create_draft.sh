#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./create_draft.sh [title]

Creates a new draft in _drafts using the provided title.
If title is omitted, you'll be prompted for it.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAFTS_DIR="${SCRIPT_DIR}/_drafts"

title="${*:-}"
if [[ -z "$title" ]]; then
  read -r -p "Enter the title: " title
fi

title="$(echo "$title" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
if [[ -z "$title" ]]; then
  echo "Error: title cannot be empty."
  exit 1
fi

slug="$(echo "$title" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
if [[ -z "$slug" ]]; then
  echo "Error: could not generate a valid slug from title '$title'."
  exit 1
fi

mkdir -p "$DRAFTS_DIR"
filename="${DRAFTS_DIR}/${slug}.md"

if [[ -f "$filename" ]]; then
  echo "Error: draft already exists: $filename"
  exit 1
fi

today="$(date +"%Y-%m-%d %H:%M:%S %z")"

cat > "$filename" <<EOF
---
layout: post
title: "$title"
date: $today
categories: [Category1, Category2]
tags: [Tag1, Tag2]
description: "Add a brief description here"
image: /assets/img/sample.jpg
image_alt: "$title"
math: true
mermaid: true
pin: false
---

Write your content here.
EOF

echo "Draft created: $filename"
