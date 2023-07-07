from setuptools import setup, find_packages
from os import path

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='trumpy',
    version='0.1.2',
    description='TRUMPY: Tracing and Reverse Understanding Memory in Pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kaichao You',
    author_email='youkaichao@gmail.com',
    url='https://github.com/youkaichao/TRUMPY',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    extras_require={
        'test': ['torchvision'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
