from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

requirements = [
    'numpy',
    'librosa',
    'sklearn',
    'scipy',
]

setup(
    name='temporal-features',
    version='0.0.1',
    description='Sound analysis tools for measuring change over time',
    long_description=readme,
    author='David Kant',
    author_email='david.kant@gmail.com',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=requirements
)
