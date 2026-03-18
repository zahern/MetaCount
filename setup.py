from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("version.txt") as f:
    version = f.read().strip()

setup(
    name="metacountregressor",
    version=open("version.txt").read().strip(),
    packages=find_packages(),
    include_package_data=True,
)