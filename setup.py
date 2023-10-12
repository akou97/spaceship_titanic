from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)-> List[str]:
    '''
    this function will return the list of requirements
    '''
    HYPEN_E_dot ='-e .'
    requirements = []
    with open(file_path) as  file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        
        if HYPEN_E_dot in requirements:
            requirements.remove(HYPEN_E_dot)

    return requirements



setup(

name = "spaceship_titanic",
version = "0.0.1",
author = "akou97",
author_email = "akourim97@gmail.com",
packages = find_packages(),
install_requires = get_requirements('requirements.txt')

)