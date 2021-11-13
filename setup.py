from setuptools import setup, find_packages

setup(
    name="varclr",
    version="1.0",
    author="Qibin Chen",
    author_email="qibinc@andrew.cmu.edu",
    license="MIT",
    python_requires=">=3.6",
    packages=find_packages(exclude=[]),
    install_requires=[
        "torch>=1.7.1",
        "transformers==4.5.1",
        "pytorch-lightning>=1.0.8",
        "black>=21.10b0",
        "pre-commit>=2.15.0",
        "wandb>=0.10.12",
    ],
)
