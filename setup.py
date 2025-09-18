from setuptools import setup, find_packages

setup(
    name="transformer-surgeon",
    version="0.2.0",
    description="Transformer model compression and optimization tools",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "Pillow",
        "qwen-vl-utils",
    ],
)
