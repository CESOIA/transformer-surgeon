from setuptools import setup, find_packages

setup(
    name="transformer-surgeon",
    version="0.5.0",
    description="Transformer model compression and optimization tools",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.8.0",
        "transformers>=5.2.0",
    ],
)
