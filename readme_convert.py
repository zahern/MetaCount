import attr
import nbformat

from nbconvert import MarkdownExporter, RSTExporter


# Load your notebook
with open('README.ipynb') as fh:
    notebook_node = nbformat.read(fh, as_version=4)

# Convert to Markdown
md_exporter = MarkdownExporter()
markdown, resources = md_exporter.from_notebook_node(notebook_node)

rst_exporter = RSTExporter()
rst, resources_rst = rst_exporter.from_notebook_node(notebook_node)


# Save the markdown text to a file
with open('README.md', 'w') as fh:
    fh.write(markdown)

# Save the reStructuredText to a file
with open('README.rst', 'w') as fh:
    fh.write(rst)


print('This finished.')