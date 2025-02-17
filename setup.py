import codecs
#import requests
import setuptools

'''
def get_package_version(package_name):
    """
    Fetch the latest version of a package from PyPI.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['info']['version']
    else:
        return None
'''
# Read the README.md file for the long description
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()
'''
package_name = 'metacountregressor'
current_version = get_package_version(package_name)
if current_version:
    print(f"The current version of {package_name} is {current_version}")
else:
'''
with open('version.txt', 'r') as f:
        # current_version = get_version from pupi
	current_version = f.read().strip()



# Split the current version into its components
version_parts = current_version.split('.')
major, minor, patch = map(int, version_parts)

# Increment the patch version
patch += 1

# Construct the new version string
new_version = f"{major}.{minor}.{patch}"

# Write the new version number back to the file
with open('version.txt', 'w') as f:
    f.write(new_version)

setuptools.setup(
    name='metacountregressor',
    version=new_version,
    description='Extensions for a Python package for estimation of count models.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Specify the content type as Markdown
    url='https://github.com/zahern/CountDataEstimation',
    author='Zeke Ahern',
    author_email='z.ahern@qut.edu.au',
    license='QUT',
    packages=['metacountregressor'],
    zip_safe=False,
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.13.1',
        'scipy>=1.0.0',
        'requests',
        'latextable'
    ]
)
