import os
from pathlib import Path


REPO_DIR = Path(__file__).parent.parent  # Directory for the REPO
CONTENT_IMAGE_DIR = os.path.join(REPO_DIR, 'base_images', 'content_images')
STYLE_IMAGE_DIR = os.path.join(REPO_DIR, 'base_images', 'style_images')
