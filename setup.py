import os
from setuptools import setup

# --- Paths ---
ROOT = os.path.abspath(os.path.dirname(__file__))
README_PATH = os.path.join(ROOT, "README.md")
VERSION_PATH = os.path.join(ROOT, "version.txt")

# --- Read README.md ---
if not os.path.exists(README_PATH):
    raise FileNotFoundError("README.md is missing in the project root.")

with open(README_PATH, "r", encoding="utf-8") as f:
    long_description = f.read()

# --- Version auto-increment ---
with open(VERSION_PATH, "r") as f:
    current_version = f.read().strip()

major, minor, patch = map(int, current_version.split("."))
new_version = f"{major}.{minor}.{patch + 1}"

with open(VERSION_PATH, "w") as f:
    f.write(new_version)

# --- Setup ---
setup(
    name="metacountregressor",
    version=new_version,
    description="Extensive Testing for Estimation of Data Count Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zahern/CountDataEstimation",
    author="Zeke Ahern, Alexander Paz",
    author_email="z.ahern@qut.edu.au",
    license="MIT",
    packages=["metacountregressor"],
    python_requires=">=3.10",
    install_requires=[
        "numpy<2.0.0",
        "requests>=2.20.0,<3.0.0",
        "latextable",
        "pandas>=1.4,!=2.1.0",
        "scikit-learn>=1.3.1",
        "statsmodels",
        "psutil",
        "matplotlib>=3.0.0",
        "pybind11>=2.12,<3.0.0",
    ],
    zip_safe=False,
)