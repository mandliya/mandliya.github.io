# Jupyter Notebook to Blog Post Converter

This system automatically converts Jupyter notebooks to draft blog posts for your Jekyll blog using the Chirpy theme. It handles all the formatting, metadata generation, and asset management automatically, and can pull notebooks from **local files, GitHub repositories, or direct URLs**.

## Features

- ✅ **Automatic conversion** of `.ipynb` files to markdown with proper front matter
- ✅ **Remote repository support** - clone and convert from GitHub repositories
- ✅ **URL download support** - convert notebooks from direct URLs
- ✅ **Image handling** - automatically copies and organizes images to the assets directory
- ✅ **Metadata extraction** - extracts title, categories, and tags from notebook content
- ✅ **GitHub Actions integration** - automatically converts notebooks when pushed to the repository
- ✅ **Local development** - works both locally and in CI/CD
- ✅ **Chirpy theme compatibility** - generates proper front matter for the Chirpy Jekyll theme

## Quick Start

### Option 1: Local Conversion

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Convert a notebook:**
   ```bash
   # Local notebook
   ./notebook_to_draft.sh my_notebook.ipynb
   
   # GitHub repository (searches for notebooks automatically)
   ./notebook_to_draft.sh fastai/fastbook
   
   # Specific notebook in a repository
   ./notebook_to_draft.sh microsoft/ML-For-Beginners::1-Introduction/04-techniques-of-ML/notebook.ipynb
   
   # Direct URL
   ./notebook_to_draft.sh https://raw.githubusercontent.com/user/repo/main/notebook.ipynb
   
   # GitHub blob URL (automatically converted to raw URL)
   ./notebook_to_draft.sh https://github.com/user/repo/blob/main/notebook.ipynb
   
   # With custom output name
   ./notebook_to_draft.sh fastai/fastbook::01_intro.ipynb my-custom-slug
   ```

### Option 2: Automatic GitHub Actions

1. **Push a notebook to your repository** - the GitHub Action will automatically convert it
2. **Use manual trigger** with repository URLs or specific notebook paths
3. **Review the generated draft** in the `_drafts` directory
4. **Publish when ready** using your existing publish workflow

## Supported Notebook Sources

| Source Type | Example | Description |
|-------------|---------|-------------|
| 📁 **Local File** | `my_notebook.ipynb` | Local Jupyter notebook file |
| 🌐 **Direct URL** | `https://raw.githubusercontent.com/user/repo/main/notebook.ipynb` | Direct link to notebook file |
| 📋 **GitHub Blob URL** | `https://github.com/user/repo/blob/main/notebook.ipynb` | GitHub file view URL (auto-converted) |
| 📦 **GitHub Repository** | `username/repo-name` | Searches for notebooks in common locations |
| 📂 **Specific Notebook** | `username/repo::path/to/notebook.ipynb` | Specific notebook in repository |

## How It Works

### 1. Notebook Processing

The converter:
- **Downloads/clones** notebooks from various sources (URLs, repositories)
- **Extracts the title** from the first markdown cell or notebook metadata
- **Converts all cells** to markdown format
- **Removes input/output prompts** for cleaner output
- **Preserves code blocks** and their syntax highlighting
- **Handles mathematical equations** with LaTeX syntax

### 2. Repository Support

When you specify a repository:
- **Clones the repository** with shallow clone for efficiency
- **Searches common locations** for notebooks (`notebooks/`, `examples/`, `tutorials/`, etc.)
- **Prompts for selection** if multiple notebooks are found
- **Handles subdirectories** and complex repository structures
- **Cleans up temporary files** after conversion

### 3. Metadata Generation

Automatically generates:
- **Title**: From first markdown cell or notebook name
- **Categories**: Based on notebook content, filename, or repository context
- **Tags**: Extracted from content or defaults to "Jupyter", "Notebook"
- **Description**: Generated from the first paragraph of content
- **Date**: Current timestamp
- **Math/Mermaid**: Enabled by default for technical posts

### 4. Asset Management

- Creates a dedicated subdirectory for each notebook's images
- Copies all referenced images to `/assets/img/[slug]/`
- Updates image paths in the markdown content
- Handles both local and external images
- Searches multiple possible image locations

### 5. Front Matter Structure

The generated front matter follows the Chirpy theme format:

```yaml
---
layout: post
title: "Your Notebook Title"
date: 2024-01-15 10:30:00 -0800
categories: [Machine Learning, AI]
tags: [Jupyter, Notebook, Python]
description: "Auto-generated description from content"
image: /assets/img/your-slug/cover.png
image_alt: "Your Notebook Title"
math: true
mermaid: true
pin: false
toc: true
comments: true
---
```

## Usage Examples

### Local Development

```bash
# Convert local notebook
./notebook_to_draft.sh my_analysis.ipynb

# Convert from FastAI course
./notebook_to_draft.sh fastai/fastbook::01_intro.ipynb

# Convert from Microsoft ML course
./notebook_to_draft.sh microsoft/ML-For-Beginners

# Convert from direct URL
./notebook_to_draft.sh https://github.com/user/repo/blob/main/notebook.ipynb

# Convert with custom slug
./notebook_to_draft.sh fastai/fastbook::01_intro.ipynb fastai-introduction
```

### Python Script Direct Usage

```bash
# Show help with all options
python3 notebook_to_draft.py --help

# Convert from repository
python3 notebook_to_draft.py tensorflow/docs::site/en/tutorials/quickstart/beginner.ipynb

# Convert with custom output name
python3 notebook_to_draft.py some-repo/notebooks --output my-custom-name
```

## GitHub Actions Workflow

The included GitHub Action (`/.github/workflows/notebook-to-draft.yml`) supports:

### Automatic Triggers
- **Push events**: Automatically converts notebooks when `.ipynb` files are pushed
- **Path filtering**: Only triggers when notebooks or Python files change

### Manual Triggers
- **Repository conversion**: `username/repo-name`
- **Specific notebooks**: `username/repo::path/to/notebook.ipynb`
- **URL conversion**: Any direct notebook URL
- **Custom output names**: Override the default slug generation
- **Auto-publish option**: Automatically move drafts to `_posts/` directory

### Workflow Features
- **Automatic issue creation**: Creates GitHub issues for review
- **Smart notebook detection**: Finds notebooks in common locations
- **Batch processing**: Handles multiple notebooks in one run
- **Error handling**: Graceful failure with detailed error messages

## File Structure

After conversion, your files will be organized like this:

```
your-blog/
├── _drafts/
│   └── your-notebook-slug.md          # Generated draft post
├── _posts/                            # Published posts (if auto-publish enabled)
├── assets/
│   └── img/
│       └── your-notebook-slug/        # Images for this post
│           ├── image1.png
│           ├── image2.png
│           └── cover.png              # Add this manually
├── _notebooks/                        # Your source notebooks (optional)
│   └── your_notebook.ipynb
└── notebook_to_draft.py               # Conversion script
```

## Advanced Configuration

### Repository Search Paths

When converting from a repository without specifying a notebook path, the system searches:

1. `*.ipynb` (root level)
2. `notebooks/*.ipynb`
3. `examples/*.ipynb`
4. `demos/*.ipynb`
5. `tutorials/*.ipynb`
6. `*/*.ipynb` (any subdirectory)

### Category Detection

The system automatically assigns categories based on:

- **Notebook metadata**: Explicit categories in notebook metadata
- **Content analysis**: Keywords in markdown cells
- **Filename patterns**: ML, AI, data science keywords
- **Repository context**: Repository name and structure

Common category mappings:
- `ml`, `ai`, `machine`, `learning` → `["Machine Learning", "AI"]`
- `data`, `analysis`, `science` → `["Data Science"]`
- `deep`, `neural`, `network` → `["Deep Learning", "AI"]`
- Default → `["Technology"]`

## Customization

### Adding Metadata to Notebooks

You can add metadata to your notebooks in several ways:

1. **In notebook metadata:**
   ```json
   {
     "metadata": {
       "title": "My Custom Title",
       "categories": ["Machine Learning", "Deep Learning"],
       "tags": ["Python", "TensorFlow", "Neural Networks"]
     }
   }
   ```

2. **In markdown cells:**
   ```markdown
   # My Custom Title
   
   <!-- categories: [Machine Learning, Deep Learning] -->
   <!-- tags: [Python, TensorFlow, Neural Networks] -->
   ```

3. **In the first markdown cell:**
   ```markdown
   # My Custom Title
   
   This is my notebook about machine learning.
   
   **Categories:** Machine Learning, Deep Learning
   **Tags:** Python, TensorFlow, Neural Networks
   ```

### Customizing the Converter

You can modify `notebook_to_draft.py` to:
- Change default categories and tags
- Modify the front matter structure
- Add custom image processing
- Implement different naming conventions
- Add custom repository search paths

## Best Practices

### Notebook Structure

1. **Start with a title:** Use a markdown cell with `# Title` as the first cell
2. **Add context:** Include an introduction in the first few markdown cells
3. **Organize content:** Use markdown cells to separate sections
4. **Include images:** Reference images with descriptive alt text
5. **Add metadata:** Include categories and tags in your notebook

### Repository Organization

1. **Use clear names:** Repository and notebook names should be descriptive
2. **Organize in folders:** Use standard folders like `notebooks/`, `examples/`
3. **Include README:** Document your notebooks and their purpose
4. **Manage dependencies:** Include requirements.txt or environment.yml

### Image Management

1. **Use descriptive filenames:** `model_architecture.png` instead of `image1.png`
2. **Add alt text:** Always include descriptive alt text for accessibility
3. **Optimize images:** Compress images before adding to notebooks
4. **Add cover images:** Create a `cover.png` for each post

### Post-Conversion Steps

1. **Review the draft:** Check the generated markdown in `_drafts/`
2. **Edit front matter:** Adjust title, categories, tags, and description
3. **Add cover image:** Create a `cover.png` in the image directory
4. **Review images:** Ensure all images are properly referenced
5. **Test locally:** Run Jekyll locally to preview the post
6. **Publish:** Use your existing publish workflow

## Troubleshooting

### Common Issues

1. **Repository not found:**
   - Verify the repository exists and is public
   - Check the username/repository format
   - Ensure you have git installed for repository operations

2. **Notebook not found in repository:**
   - Use the specific path format: `username/repo::path/to/notebook.ipynb`
   - Check the notebook exists in the repository
   - Verify the path is correct (case-sensitive)

3. **Images not found:**
   - Ensure images are in the same directory as the notebook
   - Check that image paths are relative to the notebook location
   - Verify image files exist in the source repository

4. **Dependencies not installed:**
   ```bash
   pip install nbformat nbconvert PyYAML
   ```

5. **Permission errors:**
   ```bash
   chmod +x notebook_to_draft.py notebook_to_draft.sh
   ```

6. **GitHub Action fails:**
   - Check that the workflow file is in `.github/workflows/`
   - Ensure the repository has write permissions for the action
   - Verify the notebook source format is correct

### Getting Help

- Check the generated draft file for any conversion issues
- Review the console output for error messages
- Ensure your notebook follows the recommended structure
- Test with a simple notebook first
- Use the `--help` flag to see all available options

## Examples

### Converting Popular Notebooks

```bash
# FastAI course notebooks
./notebook_to_draft.sh fastai/fastbook::01_intro.ipynb

# TensorFlow tutorials
./notebook_to_draft.sh tensorflow/docs::site/en/tutorials/quickstart/beginner.ipynb

# Scikit-learn examples
./notebook_to_draft.sh scikit-learn/scikit-learn::examples/classification/plot_classifier_comparison.ipynb

# Your own repository
./notebook_to_draft.sh your-username/your-ml-project::notebooks/analysis.ipynb
```

### Batch Processing

```bash
# Convert all notebooks from a repository
./notebook_to_draft.sh microsoft/ML-For-Beginners

# Convert multiple local notebooks
for notebook in notebooks/*.ipynb; do
    ./notebook_to_draft.sh "$notebook"
done
```

## Contributing

Feel free to submit issues and enhancement requests! The converter is designed to be extensible and can be customized for different needs.

## License

This project is part of your blog repository and follows the same license terms. 
