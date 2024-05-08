import os
from datetime import date

# Default YAML metadata template
default_metadata = """---
title: "{title}"
author: "Your Name"
date: "{date}"
categories: []
tags: []
draft: true
---
"""

def create_draft(post_name):
    today = date.today().isoformat()
    metadata = default_metadata.format(title=post_name, date=today)

    # Create a folder name that's a slugified version of the post name
    folder_name = post_name.replace(' ', '-').lower()
    folder_path = os.path.join("posts", folder_name)

    # Ensure the posts directory and the new folder exist
    if not os.path.exists("posts"):
        os.makedirs("posts")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    post_filepath = os.path.join(folder_path, f"{folder_name}.qmd")

    # Write the metadata template to the new .qmd file
    with open(post_filepath, "w") as post_file:
        post_file.write(metadata)

    print(f"Created draft post: {post_filepath}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python create_draft.py 'Your Post Title'")
    else:
        create_draft(sys.argv[1])
