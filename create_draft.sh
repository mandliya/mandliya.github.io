#!/bin/bash

# Usage: ./create_new_draft.sh
# This script creates a new draft file in the _drafts folder with placeholders for initial metadata.

# Prompt for the title to generate a short slug
read -p "Enter the title: " title

# Convert title to a slug (lowercase, hyphens instead of spaces)
slug=$(echo "$title" | tr '[:upper:]' '[:lower:]' | tr -s ' ' '-' | tr -cd '[:alnum:]-')

# Set the filename as the slug in _drafts
filename="_drafts/$slug.md"

# Get today's date in the preferred format
today=$(date +"%Y-%m-%d %H:%M:%S %z")

# Create the draft file with metadata placeholders
echo "---" > "$filename"
echo "title: \"$title\"" >> "$filename"
echo "date: $today" >> "$filename"  # Default to today's date
echo "categories: [Category1, Category2]" >> "$filename"  # Placeholder categories
echo "tags: [Tag1, Tag2]" >> "$filename"                  # Placeholder tags
echo "description: \"Add a brief description here\"" >> "$filename"
echo "image: /assets/img/sample.jpg" >> "$filename"       # Placeholder image path
echo "image_alt: \"$title\"" >> "$filename"
echo "math: true" >> "$filename"
echo "mermaid: true" >> "$filename"
echo "pin: false" >> "$filename"
echo "---" >> "$filename"
echo "" >> "$filename"
echo "Write your content here." >> "$filename"

# Add a message for the user
echo "Draft created at $filename with today's date. You can now edit the file to fill in the details."