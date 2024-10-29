#!/bin/bash
# Usage: ./publish_draft.sh <draft_filename>

DRAFT_FILE="_drafts/$1.md"
POST_FILE="_posts/$(date +%Y-%m-%d)-$1.md"

if [ -f "$DRAFT_FILE" ]; then
  mv "$DRAFT_FILE" "$POST_FILE"
  echo "Draft '$1' has been published as a post."
else
  echo "Draft '$1' does not exist in _drafts."
fi