"""Setup script for FOMO evaluation system."""

from setuptools import setup, find_packages

setup(
    name="fomo-evaluation",
    version="1.0.0",
    description="Medical imaging evaluation system for FOMO challenge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "nibabel>=3.2.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "fomo-evaluate=main:main",
        ],
    },
)
