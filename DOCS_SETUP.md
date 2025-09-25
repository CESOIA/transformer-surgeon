# Sphinx Documentation Setup Instructions

This guide will help you set up automatic documentation generation for the `transformer-surgeon` package using Sphinx and GitHub Pages.

## Overview

The setup includes:
- **Sphinx**: Python documentation generator
- **sphinx-autoapi**: Automatic API documentation from docstrings
- **sphinx-rtd-theme**: Read the Docs theme
- **GitHub Actions**: Automatic documentation building and deployment
- **GitHub Pages**: Free hosting for your documentation website

## Quick Setup

### Option 1: Run the Setup Script (Recommended)

1. Make the script executable:
```bash
chmod +x setup_docs.sh
```

2. Run the setup script:
```bash
./setup_docs.sh
```

### Option 2: Manual Setup

If you prefer to set up manually, follow these steps:

#### 1. Install Dependencies

```bash
pip install sphinx sphinx-autoapi sphinx-rtd-theme
```

#### 2. Initialize Sphinx

```bash
cd docs
sphinx-quickstart
```

Answer the prompts:
- Separate source and build directories: `y`
- Project name: `transformer-surgeon`
- Author name: `CESOIA`
- Project version: `0.2.0`
- Project language: `en`

#### 3. Configure Sphinx

Replace the generated `docs/conf.py` with the enhanced configuration that includes:
- AutoAPI for automatic documentation generation
- Napoleon for Google/NumPy docstring support
- GitHub Pages configuration
- Read the Docs theme

#### 4. Create Documentation Structure

Create the following files in the `docs/` directory:
- `index.rst` - Main documentation page
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide

#### 5. Set up GitHub Actions

Create `.github/workflows/docs.yml` for automatic deployment.

## What the Setup Creates

### File Structure
```
transformer-surgeon/
├── docs/
│   ├── conf.py                 # Sphinx configuration
│   ├── index.rst              # Main documentation page
│   ├── installation.rst       # Installation guide
│   ├── quickstart.rst         # Quick start guide
│   ├── requirements.txt       # Documentation dependencies
│   ├── Makefile               # Build commands
│   └── _build/                # Generated documentation (ignored)
├── .github/workflows/
│   └── docs.yml               # GitHub Actions workflow
└── setup_docs.sh              # This setup script
```

### Generated Documentation Sections

1. **API Reference**: Automatically generated from your docstrings
   - `transformersurgeon.qwen2_vl_c` module
   - `transformersurgeon.qwen2_5_vl_c` module
   - `transformersurgeon.utils` module

2. **Installation Guide**: How to install the package

3. **Quick Start**: Basic usage examples

4. **Search**: Full-text search functionality

## Building Documentation

### Local Development

```bash
cd docs
make html
```

View the documentation by opening `docs/_build/html/index.html` in your browser.

### Automatic Deployment

Documentation is automatically built and deployed when you:
1. Push to the `main` or `master` branch
2. The GitHub Actions workflow builds the docs
3. Deploys to GitHub Pages

## Enabling GitHub Pages

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select **GitHub Actions**
4. Your documentation will be available at: `https://CESOIA.github.io/transformer-surgeon/`

## Customization

### Adding Content

- Edit `.rst` files in the `docs/` directory
- Add docstrings to your Python code (they'll appear automatically)
- Modify `docs/conf.py` for advanced configuration

### Supported Docstring Formats

The setup supports both Google and NumPy docstring formats:

```python
def example_function(param1, param2):
    """
    Brief description of the function.

    Args:
        param1 (str): Description of param1.
        param2 (int): Description of param2.

    Returns:
        bool: Description of return value.

    Example:
        >>> example_function("hello", 42)
        True
    """
    return True
```

### Themes

To change the theme, edit `html_theme` in `docs/conf.py`:
```python
html_theme = 'alabaster'  # or 'classic', 'nature', etc.
```

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure your package is installable (`pip install -e .`)
2. **Build failures**: Check that all dependencies are listed in `docs/requirements.txt`
3. **GitHub Pages not updating**: Check the Actions tab for build errors

### Rebuilding Documentation

If you need to completely rebuild:
```bash
cd docs
make clean
make html
```

### Local Testing Before Deployment

Always test locally before pushing:
```bash
cd docs
make html
# Check docs/_build/html/index.html
```

## Advanced Configuration

### Custom CSS

1. Create `docs/_static/custom.css`
2. Add to `docs/conf.py`:
```python
html_css_files = ['custom.css']
```

### Additional Extensions

Add to `extensions` list in `docs/conf.py`:
```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'autoapi.extension',
    'sphinx.ext.todo',        # TODO lists
    'sphinx.ext.mathjax',     # Math equations
]
```

## Maintenance

### Keeping Documentation Updated

- Documentation rebuilds automatically on every push to main
- Add docstrings to new functions/classes
- Update version numbers in `conf.py` when releasing
- Review and update content in `.rst` files as needed

### Monitoring Builds

- Check the **Actions** tab on GitHub for build status
- Failed builds will prevent documentation updates
- Build logs help diagnose issues

## Support

If you encounter issues:
1. Check the GitHub Actions build logs
2. Verify that your package can be imported (`python -c "import transformersurgeon"`)
3. Test local builds before pushing
4. Ensure all dependencies are properly specified

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [sphinx-autoapi Documentation](https://sphinx-autoapi.readthedocs.io/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)