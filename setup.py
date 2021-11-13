from setuptools import find_packages, setup

setup(
    name="varclr",
    version="1.0",
    author="Qibin Chen",
    author_email="qibinc@andrew.cmu.edu",
    license="MIT",
    python_requires=">=3.6",
    packages=find_packages(exclude=[]),
    install_requires=[
        "black>=21.10b0",
        "gdown>=4.2.0",
        "isort>=5.8.0",
        "pandas>=1.1.0",
        "pre-commit>=2.15.0",
        "pytest>=6.2.4",
        "pytorch-lightning>=1.0.8,<1.3",
        "sentencepiece>=0.1.95",
        "scipy>=1.5.2",
        "torch>=1.7.1",
        "transformers==4.5.1",
        "wandb>=0.12.6",
    ],
)
