# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import re

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath('..'))

# Function to extract version from setup.py
def get_version():
    """Extract version from setup.py file."""
    setup_py_path = os.path.join(os.path.dirname(__file__), '..', 'setup.py')
    with open(setup_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
        version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if version_match:
            return version_match.group(1)
    return '0.0.0'  # fallback version

# -- Project information -----------------------------------------------------
project = 'transformer-surgeon'
copyright = '2024, CESOIA'
author = 'CESOIA'
release = get_version()

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
autoapi_template_dir = '_templates/autoapi'
autoapi_options = ['members', 'undoc-members', 'show-inheritance', 'show-module-summary', 'imported-members']

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
