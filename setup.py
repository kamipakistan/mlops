from setuptools import setup, find_packages
from setuptools.config.expand import find_packages

def load_requirements(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
        return [line for line in lines if line and not line.startswith('-e') and not line.startswith('#')]


setup(
    name='end_to_end_ML_project',
    version = '0.0.1',
    author = 'Kamran Khan',
    author_email= 'kamrankhnkami@gmail.com',
    packages=find_packages(),
    install_requires=load_requirements('requirements.txt'),
    python_requires = '>=3.8'

)