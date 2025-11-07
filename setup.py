"""Setup script for OpenMemory Python"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openmemory-python",
    version="1.0.0",
    author="Daniel Simon Jr",
    author_email="",
    description="Python implementation of OpenMemory - Long-term memory for AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielsimonjr/openmemory-python",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.2",
        "mcp>=0.9.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "openmemory-server=openmemory.mcp.server:main",
        ],
    },
)
