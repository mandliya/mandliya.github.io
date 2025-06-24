#!/bin/bash

# Notebook to Draft Converter Script
# Usage: ./notebook_to_draft.sh <notebook_path_or_repo_or_url> [output_name]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "${CYAN}📓 Notebook to Draft Converter${NC}"
    echo -e "${CYAN}================================${NC}"
}

show_help() {
    print_header
    echo ""
    echo "Usage: $0 <notebook_source> [output_name]"
    echo ""
    echo "Supported notebook sources:"
    echo "  📁 Local file:           my_notebook.ipynb"
    echo "  🌐 Direct URL:           https://raw.githubusercontent.com/user/repo/main/notebook.ipynb"
    echo "  📋 GitHub blob URL:      https://github.com/user/repo/blob/main/notebook.ipynb"
    echo "  📦 GitHub repository:    username/repo-name"
    echo "  📂 Specific notebook:    username/repo-name::path/to/notebook.ipynb"
    echo ""
    echo "Examples:"
    echo "  $0 my_notebook.ipynb"
    echo "  $0 my_notebook.ipynb my-custom-slug"
    echo "  $0 fastai/fastbook::01_intro.ipynb"
    echo "  $0 microsoft/ML-For-Beginners"
    echo "  $0 https://github.com/user/repo/blob/main/notebook.ipynb"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo ""
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" || $# -eq 0 ]]; then
    show_help
    exit 0
fi

# Check if Python script exists
if [ ! -f "notebook_to_draft.py" ]; then
    print_error "notebook_to_draft.py not found in current directory"
    exit 1
fi

NOTEBOOK_SOURCE="$1"
OUTPUT_NAME="$2"

print_header
print_info "Processing notebook source: $NOTEBOOK_SOURCE"

# Determine the source type
if [[ "$NOTEBOOK_SOURCE" == http* ]]; then
    print_info "Detected URL source"
elif [[ "$NOTEBOOK_SOURCE" == *::* ]]; then
    print_info "Detected repository with specific notebook path"
elif [[ "$NOTEBOOK_SOURCE" == */* && "$NOTEBOOK_SOURCE" != *".ipynb" && ! -f "$NOTEBOOK_SOURCE" ]]; then
    print_info "Detected repository source"
elif [[ "$NOTEBOOK_SOURCE" == *.ipynb ]]; then
    if [ -f "$NOTEBOOK_SOURCE" ]; then
        print_info "Detected local notebook file"
    else
        print_error "Local notebook file not found: $NOTEBOOK_SOURCE"
        exit 1
    fi
else
    print_warning "Source type unclear, will attempt to process as-is"
fi

# Check if Python dependencies are installed
print_info "Checking dependencies..."
if ! python3 -c "import nbformat, nbconvert, yaml" 2>/dev/null; then
    print_warning "Required Python packages not found. Installing..."
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
    else
        pip3 install nbformat nbconvert PyYAML
    fi
fi

# Check if git is available for repository operations
if [[ "$NOTEBOOK_SOURCE" == */* && "$NOTEBOOK_SOURCE" != *".ipynb" && "$NOTEBOOK_SOURCE" != http* ]]; then
    if ! command -v git &> /dev/null; then
        print_error "Git is required for repository operations but not found"
        print_info "Please install git or use a direct URL instead"
        exit 1
    fi
fi

print_info "Converting notebook..."

# Run the Python script
if [ -n "$OUTPUT_NAME" ]; then
    python3 notebook_to_draft.py "$NOTEBOOK_SOURCE" --output "$OUTPUT_NAME"
else
    python3 notebook_to_draft.py "$NOTEBOOK_SOURCE"
fi

print_success "Conversion completed!"
echo ""
print_info "Next steps:"
echo "  1. Review the draft in the _drafts directory"
echo "  2. Add a cover image if needed"
echo "  3. Edit the front matter as needed"
echo "  4. Run ./publish_draft.sh when ready to publish"
echo ""
print_info "To preview with Jekyll:"
echo "  bundle exec jekyll serve --drafts"
