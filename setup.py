import re
from setuptools import find_packages
from typing import List
from distutils.core import setup
from pathlib import Path

import pkg_resources as pkg
# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((PARENT / 'requirements.txt').read_text())]



# def get_version():
#     file = PARENT / 'poly_graphs_lib/__init__.py'
#     version= re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]
#     print(version)
#     return version


setup(
name='poly_graphs_lib',
version="1.0.0",
author='Logan Lang',
author_email='lllang@mix.wvu.edu',
packages=find_packages(),
install_requires=REQUIREMENTS
        )