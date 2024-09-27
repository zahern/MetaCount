import nbformat
from nbconvert import MarkdownExporter, RSTExporter, HTMLExporter
from nbconvert.preprocessors import Preprocessor

class HideCodePreprocessor(Preprocessor):
    def preprocess_cell(self, cell, resources, index):
        """Hide the source of cells tagged with 'hide_code' but keep the output."""
        if 'tags' in cell.metadata and 'hide_code' in cell.metadata.tags:
            if cell.cell_type == 'code':
                cell.source = ""  # Clear the source code but keep the output
        return cell, resources

def preprocess_notebook(notebook_node):
    """Apply the HideCodePreprocessor to remove code but not outputs for tagged cells."""
    preprocessor = HideCodePreprocessor()
    modified_node, _ = preprocessor.preprocess(notebook_node, resources={})
    return modified_node

# Load your notebook
with open('README.ipynb') as fh:
    notebook_node = nbformat.read(fh, as_version=4)

# Preprocess the notebook to hide code in cells tagged with 'hide_code'
notebook_node = preprocess_notebook(notebook_node)

# Convert to Markdown
md_exporter = MarkdownExporter()
markdown, resources = md_exporter.from_notebook_node(notebook_node)

# Convert to reStructuredText
rst_exporter = RSTExporter()
rst, resources_rst = rst_exporter.from_notebook_node(notebook_node)

# Convert to HTML
html_exporter = HTMLExporter()
html, resources_html = html_exporter.from_notebook_node(notebook_node)

# Save the markdown text to a file
with open('README.md', 'w') as fh:
    fh.write(markdown)

# Save the reStructuredText to a file
with open('README.rst', 'w') as fh:
    fh.write(rst)

# Save the HTML to a file
with open('README.html', 'w') as fh:
    fh.write(html)

print('Conversion finished.')