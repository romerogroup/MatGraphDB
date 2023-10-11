import re
from setuptools import find_packages, setuptools
from typing import List
from distutils.core import setup
from pathlib import Path

import pkg_resources as pkg
# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((PARENT / 'requirements.txt').read_text())]



def get_version():
    file = PARENT / 'poly_graphs_lib/__init__.py'
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]



setup(
name='poly_graphs_lib',
version=get_version(),
author='Logan Lang',
author_email='lllang@mix.wvu.edu',
packages=find_packages(),
install_requires=REQUIREMENTS
        )