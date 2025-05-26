from setuptools import setup, find_packages

setup(
    name="database",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "Pillow",
        "faiss-gpu",  # or faiss-cpu depending on your target
        "tqdm",
        "numpy",
        "transformers",
        "einops",
        "timm",
        "accelerate",
        "bitsandbytes"
    ],
    author="Chinhcachep",
    description="A metadata construction and FAISS indexing library.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/metadataconstruct",  # optional
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
