from setuptools import setup
import os

# Get the package requirements
REQSFILE = os.path.join(os.path.dirname(__file__), 'requirements.txt')
with open(REQSFILE, 'r') as f:
    REQUIREMENTS = f.readlines()
REQUIREMENTS = '\n'.join(REQUIREMENTS)

setup(name='mivit',
      version='0.1',
      description='Multi-Instrument Visualization Toolkit',
      license='GPLv3',
      packages=['mivit'],
      install_requires=REQUIREMENTS,
      zip_safe=False)
