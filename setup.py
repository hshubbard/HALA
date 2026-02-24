from setuptools import setup, find_packages

setup(
    name="hal-analyzer",
    version="1.0.0",
    description="HAL Behavior Analyzer — static analysis of HAL/driver call sequences",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "halanalyze=hal_analyzer.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "Programming Language :: Python :: 3",
    ],
)
