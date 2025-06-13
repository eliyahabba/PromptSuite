from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multipromptify",
    version="2.0.0",
    author="MultiPromptify Team",
    description="A tool that creates multi-prompt datasets from single-prompt datasets using templates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "datasets>=2.0.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
        "together>=0.2.0",
    ],
    entry_points={
        "console_scripts": [
            "multipromptify=multipromptify.cli:main",
            "multipromptify-ui=ui.main:main",
        ],
    },
    extras_require={
        "ui": [
            "streamlit>=1.28.0",
        ],
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "streamlit>=1.28.0",
        ],
    },
) 