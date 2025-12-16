"""
CortexOS NLP - Setup Configuration

The World's First Mathematically Certain Natural Language Processing Engine
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Package metadata
setup(
    name="cortexos-nlp",
    version="1.0.0",
    author="CortexOS Team",
    author_email="support@cortexos.ai",
    description="The World's First Mathematically Certain Natural Language Processing Engine",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/cortexos/cortexos-nlp",
    project_urls={
        "Bug Tracker": "https://github.com/cortexos/cortexos-nlp/issues",
        "Documentation": "https://docs.cortexos.ai/nlp",
        "Source Code": "https://github.com/cortexos/cortexos-nlp",
        "Homepage": "https://cortexos.ai/nlp"
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "dataclasses>=0.6; python_version<'3.7'",
        "typing-extensions>=3.7.4; python_version<'3.8'",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "performance": [
            "psutil>=5.8.0",
            "memory-profiler>=0.60.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "psutil>=5.8.0",
            "memory-profiler>=0.60.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "cortexos-nlp=cortexos_nlp.cli:main",
            "cortex-nlp=cortexos_nlp.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cortexos_nlp": [
            "data/*.json",
            "data/*.txt",
            "models/*.bin",
            "models/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "nlp",
        "natural language processing",
        "deterministic",
        "mathematical",
        "spacy",
        "linguistics",
        "text processing",
        "machine learning",
        "artificial intelligence",
        "cortexos"
    ],
    platforms=["Linux"],
)

