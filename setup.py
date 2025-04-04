import os
import setuptools

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the README.md file
readme_path = os.path.join(current_dir, 'README.md')

# Check if README.md exists
if not os.path.exists(readme_path):
    raise FileNotFoundError("README.md file is missing. Please ensure it exists in the project root.")

# Read the README.md file for the long description
with open(readme_path, 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Read the current version from version.txt
with open('version.txt', 'r') as f:
    current_version = f.read().strip()

# Increment the patch version
version_parts = current_version.split('.')
major, minor, patch = map(int, version_parts)
patch += 1
new_version = f"{major}.{minor}.{patch}"

# Write the new version back to version.txt
with open('version.txt', 'w') as f:
    f.write(new_version)


# Setup configuration
setuptools.setup(
    name='metacountregressor',
    version=new_version,
    description='Extensive Testing for Estimation of Data Count Models',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Specify Markdown content
    url='https://github.com/zahern/CountDataEstimation',
    author='Zeke Ahern',
    author_email='z.ahern@qut.edu.au',
    license='MIT',
    packages=['metacountregressor'],
    zip_safe=False,
    python_requires='>=3.10',
    install_requires=[
        'numpy<2.0.0',
        'requests>=2.20.0,<3.0.0',
        'latextable',
        'pandas>=1.4,!=2.1.0',
        'scikit-learn>=1.3.1',
        'statsmodels',
        'psutil',
        'matplotlib >= 3.0.0',
        "pybind11>=2.12,<3.0.0"    
        ]
)