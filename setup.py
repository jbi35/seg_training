import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='calculon',
    version='0.1',
    author='The project B team',
    author_email='biehler@ebenbuild.com',
    description='A package for semantic segmentation of \
                    chest-ct scans"',
    long_description=read('README.md'),
    setup_requires='pytest-runner',
    tests_require='pytest',
)
