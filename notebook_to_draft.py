#!/usr/bin/env python3
"""
Jupyter Notebook to Draft Post Converter

This script converts Jupyter notebooks to draft posts for a Jekyll blog using the Chirpy theme.
It handles:
- Converting notebook cells to markdown
- Extracting and processing images
- Generating appropriate front matter
- Creating proper asset structure
- Pulling notebooks from external repositories (GitHub, local paths, URLs)
"""

import os
import sys
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple
import nbformat
from nbconvert import MarkdownExporter
import yaml

class NotebookToDraftConverter:
    def __init__(self, blog_root: str):
        self.blog_root = Path(blog_root)
        self.drafts_dir = self.blog_root / "_drafts"
        self.assets_dir = self.blog_root / "assets"
        self.img_dir = self.assets_dir / "img"

        # Ensure directories exist
        self.drafts_dir.mkdir(exist_ok=True)
        self.img_dir.mkdir(exist_ok=True)

    def is_github_url(self, path: str) -> bool:
        """Check if the path is a GitHub URL."""
        return path.startswith(('https://github.com/', 'https://raw.githubusercontent.com/'))

    def is_url(self, path: str) -> bool:
        """Check if the path is a URL."""
        return path.startswith(('http://', 'https://'))

    def is_git_repo(self, path: str) -> bool:
        """Check if the path contains a git repository reference."""
        return '::' in path or path.count('/') >= 2 and not path.startswith('/')

    def download_from_url(self, url: str, temp_dir: Path) -> Path:
        """Download a notebook from a URL."""
        print(f"Downloading notebook from: {url}")

        # Convert GitHub blob URLs to raw URLs
        if 'github.com' in url and '/blob/' in url:
            url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')

        filename = url.split('/')[-1]
        if not filename.endswith('.ipynb'):
            filename += '.ipynb'

        local_path = temp_dir / filename

        try:
            urllib.request.urlretrieve(url, local_path)
            return local_path
        except Exception as e:
            raise ValueError(f"Failed to download notebook from {url}: {e}")

    def clone_and_extract(self, repo_path: str, temp_dir: Path) -> Path:
        """Clone a repository and extract the notebook."""
        print(f"Processing repository path: {repo_path}")

        # Parse different formats:
        # - username/repo::path/to/notebook.ipynb
        # - https://github.com/username/repo::path/to/notebook.ipynb
        # - username/repo (will look for notebooks in common locations)

        if '::' in repo_path:
            repo_part, notebook_path = repo_path.split('::', 1)
        else:
            repo_part = repo_path
            notebook_path = None

        # Handle GitHub URLs
        if repo_part.startswith('https://github.com/'):
            repo_url = repo_part
            repo_name = repo_part.split('/')[-1]
        else:
            # Assume it's username/repo format
            if repo_part.count('/') == 1:
                repo_url = f"https://github.com/{repo_part}.git"
                repo_name = repo_part.split('/')[-1]
            else:
                raise ValueError(f"Invalid repository format: {repo_part}")

        # Clone the repository
        clone_dir = temp_dir / repo_name
        print(f"Cloning {repo_url} to {clone_dir}")

        try:
            subprocess.run(['git', 'clone', '--depth', '1', repo_url, str(clone_dir)],
                         check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to clone repository {repo_url}: {e.stderr}")

        # Find the notebook
        if notebook_path:
            notebook_file = clone_dir / notebook_path
            if not notebook_file.exists():
                raise FileNotFoundError(f"Notebook not found: {notebook_path} in {repo_url}")
        else:
            # Search for notebooks in common locations
            search_paths = [
                "*.ipynb",
                "notebooks/*.ipynb",
                "examples/*.ipynb",
                "demos/*.ipynb",
                "tutorials/*.ipynb",
                "*/*.ipynb"
            ]

            notebook_file = None
            for pattern in search_paths:
                notebooks = list(clone_dir.glob(pattern))
                if notebooks:
                    if len(notebooks) == 1:
                        notebook_file = notebooks[0]
                        break
                    else:
                        print(f"Found multiple notebooks matching {pattern}:")
                        for i, nb in enumerate(notebooks):
                            print(f"  {i+1}. {nb.relative_to(clone_dir)}")
                        choice = input("Enter the number of the notebook to convert (or 'q' to quit): ")
                        if choice.lower() == 'q':
                            sys.exit(0)
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(notebooks):
                                notebook_file = notebooks[idx]
                                break
                        except ValueError:
                            continue

            if not notebook_file:
                raise FileNotFoundError(f"No notebooks found in repository {repo_url}")

        print(f"Found notebook: {notebook_file.relative_to(clone_dir)}")
        return notebook_file

    def get_notebook_path(self, notebook_input: str) -> Tuple[Path, Optional[Path]]:
        """
        Get the notebook path, handling various input formats:
        - Local file path
        - URL (direct download)
        - GitHub repository (clone and extract)
        - Repository with specific path

        Returns:
            Tuple of (notebook_path, temp_dir) where temp_dir is None for local files
        """
        if self.is_url(notebook_input):
            # Direct URL download
            temp_dir = Path(tempfile.mkdtemp())
            notebook_path = self.download_from_url(notebook_input, temp_dir)
            return notebook_path, temp_dir

        elif self.is_git_repo(notebook_input):
            # Repository reference
            temp_dir = Path(tempfile.mkdtemp())
            notebook_path = self.clone_and_extract(notebook_input, temp_dir)
            return notebook_path, temp_dir

        else:
            # Local file path
            notebook_path = Path(notebook_input)
            if not notebook_path.exists():
                raise FileNotFoundError(f"Notebook not found: {notebook_path}")
            return notebook_path, None

    def extract_title_from_notebook(self, notebook_path: Path) -> str:
        """Extract title from notebook metadata or first markdown cell."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)

            # Try to get title from metadata
            if 'metadata' in nb and 'title' in nb['metadata']:
                return nb['metadata']['title']

            # Look for title in first markdown cell
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'markdown':
                    content = ''.join(cell.get('source', []))
                    # Look for # Title pattern
                    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                    if title_match:
                        return title_match.group(1).strip()

            # Fallback to filename
            return notebook_path.stem.replace('_', ' ').title()

        except Exception as e:
            print(f"Warning: Could not extract title from notebook: {e}")
            return notebook_path.stem.replace('_', ' ').title()

    def generate_slug(self, title: str) -> str:
        """Generate a URL-friendly slug from title."""
        # Convert to lowercase and replace spaces with hyphens
        slug = re.sub(r'[^\w\s-]', '', title.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug.strip('-')

    def extract_categories_and_tags(self, notebook_path: Path) -> Tuple[List[str], List[str]]:
        """Extract categories and tags from notebook metadata or content."""
        categories = []
        tags = []

        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)

            # Try to get from metadata
            if 'metadata' in nb:
                metadata = nb['metadata']
                if 'categories' in metadata:
                    categories = metadata['categories']
                if 'tags' in metadata:
                    tags = metadata['tags']

            # If not found in metadata, try to extract from content
            if not categories and not tags:
                for cell in nb.get('cells', []):
                    if cell.get('cell_type') == 'markdown':
                        content = ''.join(cell.get('source', []))
                        # Look for categories/tags in markdown comments
                        cat_match = re.search(r'categories?:\s*\[(.*?)\]', content, re.IGNORECASE)
                        if cat_match:
                            categories = [cat.strip().strip('"\'') for cat in cat_match.group(1).split(',')]

                        tag_match = re.search(r'tags?:\s*\[(.*?)\]', content, re.IGNORECASE)
                        if tag_match:
                            tags = [tag.strip().strip('"\'') for tag in tag_match.group(1).split(',')]

            # Default categories based on notebook name
            if not categories:
                notebook_name = notebook_path.name.lower()
                if any(word in notebook_name for word in ['ml', 'ai', 'machine', 'learning']):
                    categories = ['Machine Learning', 'AI']
                elif any(word in notebook_name for word in ['data', 'analysis', 'science']):
                    categories = ['Data Science']
                elif any(word in notebook_name for word in ['deep', 'neural', 'network']):
                    categories = ['Deep Learning', 'AI']
                else:
                    categories = ['Technology']

            # Default tags
            if not tags:
                tags = ['Jupyter', 'Notebook']

        except Exception as e:
            print(f"Warning: Could not extract categories/tags: {e}")
            categories = ['Technology']
            tags = ['Jupyter', 'Notebook']

        return categories, tags

    def generate_description(self, content: str) -> str:
        """Generate a description from the first paragraph of content."""
        # Remove markdown formatting and get first paragraph
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('!['):
                # Remove markdown formatting
                clean_line = re.sub(r'[*_`]', '', line)
                if len(clean_line) > 50:
                    return clean_line[:150] + "..." if len(clean_line) > 150 else clean_line

        return "A Jupyter notebook converted to a blog post."

    def process_images(self, content: str, notebook_name: str, notebook_dir: Path, temp_dir: Optional[Path] = None) -> str:
        """
        Process and move images to the assets directory.

        Expected image structure in repositories:
        - Images should be in an 'images/' directory relative to the notebook
        - Use markdown syntax: ![alt text](images/filename.png)
        - Generated cell output images (output_*.png) will be skipped with a note
        """
        # Create a subdirectory for this notebook's images
        notebook_img_dir = self.img_dir / notebook_name
        notebook_img_dir.mkdir(exist_ok=True)

        def replace_image(match):
            alt_text = match.group(1)
            image_path = match.group(2)

            # Handle external URLs - keep as is
            if image_path.startswith('http'):
                return match.group(0)

            # Skip generated cell output images with informative message
            if image_path.startswith('output_') and (image_path.endswith('.png') or image_path.endswith('.svg')):
                print(f"Info: Skipping generated cell output image: {image_path}")
                print("      (Cell output images are generated when notebook is executed)")
                # Remove the image reference since it won't render
                return f"*[Generated plot output - run notebook to see visualization]*"

            # Handle local images - expect them to be in images/ directory
            if image_path.startswith('./'):
                image_path = image_path[2:]

            # Primary search locations (following our rules)
            search_paths = []

            if temp_dir:
                # For remote repositories, look in the cloned directory
                search_paths = [
                    temp_dir / image_path,  # Direct path from notebook
                    temp_dir / notebook_dir.name / image_path,  # If notebook is in subdirectory
                ]
            else:
                # For local files
                search_paths = [
                    notebook_dir / image_path,  # Relative to notebook
                ]

            # Try to find and copy the image
            for search_path in search_paths:
                if search_path.exists() and search_path.is_file():
                    try:
                        # Copy to assets directory
                        new_path = notebook_img_dir / Path(image_path).name
                        shutil.copy2(search_path, new_path)
                        print(f"✅ Copied image: {search_path.name} -> /assets/img/{notebook_name}/")
                        # Update the reference
                        return f'![{alt_text}](/assets/img/{notebook_name}/{Path(image_path).name})'
                    except Exception as e:
                        print(f"Warning: Could not copy image {search_path}: {e}")
                        continue

            # If image not found, provide helpful guidance
            print(f"❌ Image not found: {image_path}")
            print(f"   Expected location: images/{Path(image_path).name}")
            print(f"   Please ensure your repository follows the image structure rules:")
            print(f"   - Place images in an 'images/' directory next to your notebook")
            print(f"   - Use markdown syntax: ![alt text](images/filename.png)")

            # Keep original reference but add a note
            return f'![{alt_text} - IMAGE NOT FOUND]({image_path})'

        # Process markdown image syntax: ![alt](path)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        content = re.sub(image_pattern, replace_image, content)

        return content

    def convert_notebook_to_markdown(self, notebook_path: Path) -> str:
        """Convert notebook to markdown using nbconvert."""
        try:
            # Load the notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            # Configure the markdown exporter
            md_exporter = MarkdownExporter()
            md_exporter.exclude_input_prompt = True
            md_exporter.exclude_output_prompt = True

            # Convert to markdown
            (body, resources) = md_exporter.from_notebook_node(nb)

            return body

        except Exception as e:
            print(f"Error converting notebook: {e}")
            return ""

    def generate_front_matter(self, title: str, categories: List[str], tags: List[str],
                            description: str, notebook_name: str) -> str:
        """Generate Jekyll front matter for the Chirpy theme."""
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")

        front_matter = {
            'layout': 'post',
            'title': title,
            'date': today,
            'categories': categories,
            'tags': tags,
            'description': description,
            'image': f'/assets/img/{notebook_name}/cover.png',  # Default cover image
            'image_alt': title,
            'math': True,  # Enable math rendering
            'mermaid': True,  # Enable mermaid diagrams
            'pin': False,
            'toc': True,  # Enable table of contents
            'comments': True
        }

        return yaml.dump(front_matter, default_flow_style=False, sort_keys=False)

    def convert_notebook(self, notebook_input: str, output_name: Optional[str] = None) -> str:
        """Convert a notebook to a draft post."""
        # Get the notebook path (handles URLs, repos, local files)
        notebook_path, temp_dir = self.get_notebook_path(notebook_input)
        notebook_dir = notebook_path.parent

        try:
            # Extract metadata
            title = self.extract_title_from_notebook(notebook_path)
            categories, tags = self.extract_categories_and_tags(notebook_path)

            # Generate output filename
            if output_name:
                slug = self.generate_slug(output_name)
            else:
                slug = self.generate_slug(title)

            # Convert notebook to markdown
            content = self.convert_notebook_to_markdown(notebook_path)
            if not content:
                raise ValueError("Failed to convert notebook to markdown")

            # Generate description
            description = self.generate_description(content)

            # Process images (pass temp_dir for remote repositories)
            notebook_name = slug
            content = self.process_images(content, notebook_name, notebook_dir, temp_dir)

            # Generate front matter
            front_matter = self.generate_front_matter(title, categories, tags, description, notebook_name)

            # Create the draft file
            draft_path = self.drafts_dir / f"{slug}.md"

            with open(draft_path, 'w', encoding='utf-8') as f:
                f.write("---\n")
                f.write(front_matter)
                f.write("---\n\n")
                f.write(content)

            print(f"Draft created: {draft_path}")
            print(f"Title: {title}")
            print(f"Categories: {', '.join(categories)}")
            print(f"Tags: {', '.join(tags)}")
            print(f"Images directory: /assets/img/{notebook_name}/")

            return str(draft_path)

        finally:
            # Clean up temporary files if notebook was downloaded/cloned
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(
        description='Convert Jupyter notebook to draft post',
        epilog="""
Examples:
  # Local notebook
  %(prog)s my_notebook.ipynb

  # GitHub repository (will search for notebooks)
  %(prog)s username/repo-name

  # Specific notebook in GitHub repo
  %(prog)s username/repo-name::path/to/notebook.ipynb

  # Direct URL
  %(prog)s https://raw.githubusercontent.com/user/repo/main/notebook.ipynb

  # GitHub blob URL (will be converted to raw URL)
  %(prog)s https://github.com/user/repo/blob/main/notebook.ipynb
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('notebook', help='Path to notebook, GitHub repo, or URL')
    parser.add_argument('--output', '-o', help='Output name for the draft (optional)')
    parser.add_argument('--blog-root', default='.', help='Path to blog root directory')

    args = parser.parse_args()

    try:
        converter = NotebookToDraftConverter(args.blog_root)
        draft_path = converter.convert_notebook(args.notebook, args.output)
        print(f"\n✅ Successfully created draft: {draft_path}")
        print("\nNext steps:")
        print("1. Review and edit the draft in the _drafts directory")
        print("2. Add a cover image to /assets/img/[slug]/cover.png")
        print("3. Run the publish script when ready")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
