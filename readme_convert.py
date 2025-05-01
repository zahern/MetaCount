import attr
import nbformat
import os
import sys

from nbconvert import MarkdownExporter, RSTExporter
#WAS README
name = 'README'
#name - 'Tutorial_Batch'
# Load your notebook
with open(f'{name}.ipynb') as fh:
    notebook_node = nbformat.read(fh, as_version=4)

# Convert to Markdown
md_exporter = MarkdownExporter()
markdown, resources = md_exporter.from_notebook_node(notebook_node)

rst_exporter = RSTExporter()
rst, resources_rst = rst_exporter.from_notebook_node(notebook_node)


# Save the markdown text to a file
with open(f'{name}.md', 'w') as fh:
    fh.write(markdown)

# Save the reStructuredText to a file
with open(f'{name}.rst', 'w') as fh:
    fh.write(rst)

# Save as HTML
html_content = "<html><body><pre>" + rst + "</pre></body></html>"
with open(f'{name}.html', 'w') as fh:
    fh.write(html_content)

print('This finished.')