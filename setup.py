"""
Setup script for backward compatibility.
Modern installations should use pyproject.toml.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cdp-generator",
    version="1.0.0",
    author="CDP Generator Contributors",
    description="Concrete Damage Plasticity (CDP) Model Input Parameter Generator for ABAQUS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sflabbe/cdp-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "openpyxl>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cdp-generator=cdp_generator.cli:main",
        ],
    },
    keywords="concrete damage-plasticity CDP ABAQUS finite-element material-modeling",
)
