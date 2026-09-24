from setuptools import setup, find_packages

setup(
    name="transformer-surgeon",
    version="0.10.0",
    description="Transformer model compression and optimization tools",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.9.0",
        "transformers>=5.9.0",
        "torchao",
    ],
    extras_require={
        "executorch": ["executorch>=1.0.0"],
        "tensorrt": ["torch-tensorrt"],  # deprecated torch-tensorrt backend
        # ONNX export; TensorRT engines additionally need the `tensorrt` package
        # matching the CUDA version (e.g. tensorrt-cu13), installed separately.
        "onnx": ["onnx>=1.19", "onnxscript>=0.7", "ml_dtypes"],
        # "test" and "dev" are intentionally identical — "test" is the minimal
        # extra needed to run the pytest suite; "dev" is the name contributors
        # expect. Keep both so `pip install -e ".[test]"` and `-e ".[dev]"` work.
        "test": ["pytest"],
        "dev": ["pytest"],
    },
)
