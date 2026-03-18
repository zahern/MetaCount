from setuptools import setup, find_packages

setup(
    name="metacountregressor",
    version=open("version.txt").read().strip(),
    packages=find_packages(),
    include_package_data=True,
)