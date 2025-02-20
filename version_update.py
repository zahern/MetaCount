

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


