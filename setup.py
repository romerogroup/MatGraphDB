from setuptools import find_packages, setuptools
from typing import List


HYPHEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of reuirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return Requirements

setup(
name='poly_graphs_lib',
version='0.0.1',
author='Logan',
author_email='lllang@mix.wvu.edu',
packages=find_packages(),
install_requires=get_requirements('requiremets.txt')
        )