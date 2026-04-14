from setuptools import find_packages, setup

setup(
    name='dateset-project',
    version='0.1.0',
    description='Dataset generation pipeline converted from shujuji.ipynb',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9',
)
