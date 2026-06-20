#!/usr/bin/env python
# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0

from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md")

setup(
    name="geop2vnet",
    version="1.0.0",
    author="GeoP2VNet authors",
    description="GeoP2VNet: Point-to-Voxel Projection with Geometric Feature Preservation",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/somtiannes/GeoP2VNet",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "monai>=1.3.0",
        "nibabel>=5.0.0",
        "scipy>=1.10.0",
        "einops>=0.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
