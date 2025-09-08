"""
Setup script for Apertus Transparency Guide
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="apertus-transparency-guide",
    version="1.0.0",
    author="Swiss AI Community",
    author_email="community@swissai.dev",
    description="Complete guide to using Apertus Swiss AI with full transparency analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/apertus-transparency-guide",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "apertus-chat=examples.basic_chat:main",
            "apertus-multilingual=examples.multilingual_demo:main",
            "apertus-dashboard=dashboards.streamlit_transparency:main",
        ],
    },
    keywords="ai, machine learning, transparency, swiss ai, apertus, huggingface, transformers",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/apertus-transparency-guide/issues",
        "Source": "https://github.com/yourusername/apertus-transparency-guide",
        "Documentation": "https://apertus-transparency-guide.readthedocs.io/",
        "Swiss AI Community": "https://swissai.community",
    },
)
