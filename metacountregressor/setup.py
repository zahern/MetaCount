import codecs

import setuptools

with codecs.open("README.rst", encoding='utf8') as fh:
    long_description = fh.read()

setuptools.setup(name='metacountregressor',
                 version='0.1.63',
                 description='Extensions for a Python package for \
                             estimation of data count models.',
                 long_description=long_description,
                 long_description_content_type="text/x-rst",
                 url='https://github.com/zahern/CountDataEstimation',
                 author='Zeke Ahern',
                 author_email='zeke.ahern@hdr.qut.edu.au',
                 license='QUT',
                 packages=['MetaCountRegressor'],
                 zip_safe=False,
                 python_requires='>=3.10',
                 install_requires=[
                     'numpy>=1.13.1',
                     'scipy>=1.0.0',
                     'latextable'
                 ])