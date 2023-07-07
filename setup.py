from setuptools import setup, find_packages

setup(
    name='trumpy',
    version='0.1',
    description='TRUMPY: Tracing and Reverse Understanding Memory in Pytorch',
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
