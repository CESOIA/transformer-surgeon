from setuptools import setup, find_packages

setup(
    name="transformer-surgeon",
    version="0.6.0",
    description="Transformer model compression and optimization tools",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.9.1",
        "transformers>=5.2.0",
        "executorch>=1.0.0",
    ],
)
