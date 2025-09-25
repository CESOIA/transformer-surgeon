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
    
    # Create .nojekyll file to prevent Jekyll processing
    touch .nojekyll
    
    echo "✅ Documentation rebuilt successfully!"
    echo "📂 Files are ready in docs/ directory for GitHub Pages"
    echo "🚫 .nojekyll file created to prevent Jekyll interference"
else
    echo "❌ Build failed!"
    exit 1
fi
