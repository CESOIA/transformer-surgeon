#!/bin/bash

# ==============================================================================
# Sphinx Documentation Setup Script for transformer-surgeon
# ==============================================================================
# This script sets up a complete Sphinx documentation pipeline that:
# 1. Extracts docstrings from the transformersurgeon package
# 2. Generates HTML documentation
# 3. Sets up GitHub Pages deployment via GitHub Actions
# ==============================================================================

set -e  # Exit on any error

echo "🚀 Setting up Sphinx documentation for transformer-surgeon..."

# Step 1: Install dependencies
echo "📦 Installing Sphinx and dependencies..."
pip install sphinx sphinx-autoapi sphinx-rtd-theme

# Step 2: Initialize Sphinx in docs directory
echo "🔧 Initializing Sphinx configuration..."
cd docs

# Remove any existing content
rm -rf *

# Run sphinx-quickstart with automated answers for GitHub Pages structure
cat << 'EOF' | sphinx-quickstart --no-sep --no-makefile --no-batchfile .
transformer-surgeon
CESOIA
0.2.0
en
EOF

echo "✅ Sphinx initialized successfully!"

# Step 3: Configure Sphinx for API documentation
echo "🔧 Configuring Sphinx for automatic API documentation..."

# Backup original conf.py
cp conf.py conf.py.backup

# Create enhanced conf.py
cat << 'EOF' > conf.py
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'transformer-surgeon'
copyright = '2024, CESOIA'
author = 'CESOIA'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'autoapi.extension',
]

# AutoAPI configuration for automatic API documentation
autoapi_dirs = ['../transformersurgeon']
autoapi_root = 'api'
autoapi_add_toctree_entry = True
autoapi_generate_api_docs = True

# Napoleon settings for Google/NumPy docstring styles
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# GitHub Pages configuration
html_baseurl = 'https://CESOIA.github.io/transformer-surgeon/'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# -- AutoDoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
EOF

echo "✅ Enhanced conf.py created!"

# Step 4: Create main documentation structure
echo "📝 Creating documentation structure..."

# Create enhanced index.rst
cat << 'EOF' > index.rst
Welcome to transformer-surgeon's documentation!
=============================================

**transformer-surgeon** is a Python package for transformer model compression and optimization tools.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index

Features
--------

* Model compression utilities
* Qwen2-VL model support
* Qwen2.5-VL model support
* Optimization tools

Installation
-----------

You can install transformer-surgeon using pip:

.. code-block:: bash

   pip install git+https://github.com/CESOIA/transformer-surgeon

Quick Start
----------

WIP

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
EOF

# Create installation.rst
cat << 'EOF' > installation.rst
Installation
============

Requirements
------------

* Python >= 3.8
* PyTorch
* Transformers library
* Pillow
* qwen-vl-utils

Installing from PyPI
-------------------

The easiest way to install transformer-surgeon is using pip:

.. code-block:: bash

   pip install git+https://github.com/CESOIA/transformer-surgeon

Installing from Source
--------------------

To install from source:

.. code-block:: bash

   git clone https://github.com/CESOIA/transformer-surgeon.git
   cd transformer-surgeon
   pip install -e .

Dependencies
-----------

The package automatically installs the following dependencies:

* torch
* transformers
* Pillow
* qwen-vl-utils
EOF

# Create quickstart.rst
cat << 'EOF' > quickstart.rst
Quick Start Guide
================

This guide will help you get started with transformer-surgeon.

Basic Usage
----------

WIP

Examples
--------

For more detailed examples, check out the notebooks in the ``notebooks/`` directory of the repository.
EOF

# Create _static directory if it doesn't exist
mkdir -p _static

echo "✅ Documentation structure created!"

# Step 5: Build initial documentation to docs directory
echo "🏗️ Building documentation to docs directory for GitHub Pages..."
sphinx-build -b html . _build
# Move built files to docs root for GitHub Pages
if [ -d "_build" ]; then
    cp -r _build/* .
    # Clean up intermediate directories but keep the actual files
    rm -rf _build .doctrees .buildinfo
    echo "✅ Documentation built and moved to docs directory!"
else
    echo "❌ Build failed - _build directory not found!"
    exit 1
fi

# Step 6: Create GitHub Actions workflow
echo "🔄 Setting up GitHub Actions for automatic deployment..."

# Go back to root directory
cd ..

# Create .github/workflows directory
mkdir -p .github/workflows

# Create GitHub Actions workflow for documentation
cat << 'EOF' > .github/workflows/docs.yml
name: Build and Deploy Documentation

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install sphinx sphinx-autoapi sphinx-rtd-theme
        pip install -e .

    - name: Build documentation
      run: |
        cd docs
        sphinx-build -b html . _build
        cp -r _build/* .
        rm -rf _build .doctrees .buildinfo

    - name: Setup Pages
      if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master'
      uses: actions/configure-pages@v3

    - name: Upload artifact
      if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master'
      uses: actions/upload-pages-artifact@v2
      with:
        path: 'docs'

  deploy:
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master'
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2
EOF

echo "✅ GitHub Actions workflow created!"

# Step 7: Create requirements file for documentation
echo "📋 Creating documentation requirements..."
cat << 'EOF' > docs/requirements.txt
sphinx>=5.0
sphinx-autoapi>=2.0
sphinx-rtd-theme>=1.0
EOF

# Step 8: Update .gitignore for documentation
echo "🚫 Updating .gitignore for documentation..."
if [ -f .gitignore ]; then
    # Add documentation build directories to .gitignore if not already present
    grep -q "_build/" .gitignore || echo "_build/" >> .gitignore
    # Remove docs/_build since we're building directly to docs/
    sed -i '/docs\/_build\//d' .gitignore 2>/dev/null || true
else
    cat << 'EOF' > .gitignore
# Documentation builds
_build/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
EOF
fi

echo "✅ .gitignore updated!"

# Step 9: Create build script for easy documentation updates
echo "🔨 Creating build script for documentation updates..."
cat << 'EOF' > build_docs.sh
#!/bin/bash
# Script to rebuild and update documentation

echo "🏗️ Rebuilding documentation..."
cd docs

# Clean previous build
rm -rf _build *.html _static doctrees .doctrees 2>/dev/null || true

# Build documentation
sphinx-build -b html . _build

# Move built files to docs root for GitHub Pages
if [ -d "_build" ]; then
    cp -r _build/* .
    rm -rf _build .doctrees .buildinfo
    echo "✅ Documentation rebuilt successfully!"
    echo "📂 Files are ready in docs/ directory for GitHub Pages"
else
    echo "❌ Build failed!"
    exit 1
fi
EOF

chmod +x build_docs.sh

cd ..

echo "✅ Documentation setup complete!"
echo ""
echo "📋 SETUP SUMMARY"
echo "=================="
echo "✅ Sphinx configuration created"
echo "✅ API documentation configured with sphinx-autoapi"
echo "✅ GitHub Actions workflow created"
echo "✅ Documentation structure created"
echo "✅ Build system configured"
echo ""
echo "🚀 NEXT STEPS"
echo "============="
echo "1. Enable GitHub Pages in your repository settings:"
echo "   - Go to Settings > Pages"
echo "   - Source: GitHub Actions"
echo ""
echo "2. To build documentation locally:"
echo "   ./build_docs.sh"
echo ""
echo "3. To view documentation:"
echo "   Open docs/index.html in your browser"
echo ""
echo "4. Push changes to trigger automatic deployment:"
echo "   git add ."
echo "   git commit -m 'Add Sphinx documentation'"
echo "   git push origin main"
echo ""
echo "📚 Your documentation will be available at:"
echo "   https://CESOIA.github.io/transformer-surgeon/"
echo ""
echo "🔧 To customize further:"
echo "   - Edit docs/conf.py for configuration"
echo "   - Edit docs/*.rst files for content"
echo "   - Add docstrings to your Python code"