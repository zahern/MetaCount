import codecs
import requests
import setuptools


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



package_name = 'metacountregressor'
current_version = get_package_version(package_name)
if current_version:
    print(f"The current version of {package_name} is {current_version}")
else:

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


