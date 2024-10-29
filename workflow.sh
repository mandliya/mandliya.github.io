#!/bin/bash

# Define paths
VENV_PATH="$HOME/.pyenv/versions/3.11.7/envs/blog"
BLOG_REPO_PATH="$HOME/p/blog/mandliya.github.io"
NOTEBOOKS_PATH="$BLOG_REPO_PATH/_notebooks"
DRAFTS_PATH="$BLOG_REPO_PATH/_drafts"
POSTS_PATH="$BLOG_REPO_PATH/_posts"

# Function to activate the virtual environment
activate_venv() {
    if [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate"
        echo "Virtual environment 'blog' activated."
    else
        echo "Error: Virtual environment at $VENV_PATH not found."
        exit 1
    fi
}

# Function to deactivate the virtual environment
deactivate_venv() {
    deactivate
    echo "Virtual environment deactivated."
}

# Function to ensure required directories exist
ensure_directories() {
    for dir in "$NOTEBOOKS_PATH" "$DRAFTS_PATH" "$POSTS_PATH"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}

# Function to create a new Jupyter notebook
create_notebook() {
    echo "Opening Jupyter Notebook..."
    cd "$NOTEBOOKS_PATH" || exit
    jupyter lab
}

# Function to convert notebook to Markdown
convert_notebook() {
    local title="$1"
    local notebook_file="$NOTEBOOKS_PATH/${title}.ipynb"
    local markdown_file="$NOTEBOOKS_PATH/${title}.md"

    # Check if the notebook file exists
    if [ ! -f "$notebook_file" ]; then
        echo "Error: Notebook '$title.ipynb' not found in the notebooks folder ('$notebook_file')."
        exit 1
    fi

    # Convert notebook to Markdown
    jupyter nbconvert --to markdown "$notebook_file" --output "$markdown_file"
    echo "Notebook converted to Markdown: $markdown_file"
}

# Function to add front matter and move to drafts
add_frontmatter() {
    local title="$1"
    local draft_file="$DRAFTS_PATH/${title}.md"
    local markdown_file="$NOTEBOOKS_PATH/${title}.md"
    local date=$(date +"%Y-%m-%d %H:%M:%S %z")

    # Check if Markdown file exists
    if [ ! -f "$markdown_file" ]; then
        echo "Error: Markdown file for notebook '$title' not found."
        exit 1
    fi

    # Add front matter to the Markdown file
    cat <<EOF > temp.md
---
layout: post
title: "$title"
date: $date
categories: [Learning, Notes]
tags: [Jupyter, Draft]
math: true
---
EOF

    # Append the notebook content to the front matter and move to drafts
    cat "$markdown_file" >> temp.md
    mv temp.md "$draft_file"
    echo "Draft created at $draft_file"
}

# Function to review the draft locally with Jekyll
review_draft() {
    cd "$BLOG_REPO_PATH" || exit
    echo "Starting Jekyll server to review drafts..."
    bundle exec jekyll serve --drafts
}

# Function to publish the draft by moving it to _posts
publish_draft() {
    local title="$1"
    local draft_file="$DRAFTS_PATH/${title}.md"
    local today=$(date +"%Y-%m-%d")
    local post_file="$POSTS_PATH/${today}-${title}.md"

    # Check if the draft file exists
    if [ ! -f "$draft_file" ]; then
        echo "Error: Draft '$title.md' not found in the drafts folder."
        exit 1
    fi

    # Move the draft to _posts
    mv "$draft_file" "$post_file"
    echo "Draft published as $post_file"
}

# Main script workflow
activate_venv
ensure_directories

case "$1" in
    new)
        create_notebook
        ;;
    convert)
        read -p "Enter the notebook title to convert to Markdown: " title
        convert_notebook "$title"
        add_frontmatter "$title"
        ;;
    review)
        review_draft
        ;;
    publish)
        read -p "Enter the title of the draft to publish: " title
        publish_draft "$title"
        ;;
    *)
        echo "Usage: $0 {new|convert|review|publish}"
        echo "  new      - Create a new Jupyter notebook"
        echo "  convert  - Convert notebook to markdown, add frontmatter, and save as draft"
        echo "  review   - Start Jekyll server to review drafts"
        echo "  publish  - Publish the reviewed draft"
        ;;
esac
