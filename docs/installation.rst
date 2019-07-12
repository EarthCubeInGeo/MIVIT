Installation
************

Basic Installation
==================

MIVIT is pip installable from github::

	pip install git+https://github.com/EarthCubeInGeo/MIVIT.git

Dependencies
============

MIVIT depends on `cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_ for mapping spatial coordinates and `apexpy <https://apexpy.readthedocs.io/en/latest/>`_ for mapping magnetic coordinates.  Both of these packages must be installed correctly for MIVIT to function. For assistance, please refer to:

| `cartopy installation guide <https://scitools.org.uk/cartopy/docs/latest/installing.html#installing>`_
| `apexpy intallation guide <https://apexpy.readthedocs.io/en/latest/installation.html>`_

Helper Scripts
==============

Although MIVIT can be used without any additional geospace packages, it includes a set of functions to make it easier to create DataSet objects from common geospace data sets.  In general, these functions use packages developed by the data providers for fetching and reading specialized data files.  Data packages are not included in the standard requirements for MIVIT and must be installed independently.  If you intend to use MIVIT helper scripts to access on of the following types of data, please make sure you have the corresponding package installed.

- AMISR: `visuamisr <https://github.com/asreimer/visuamisr>`_
- Madrigal: `madrigalWeb <https://pypi.org/project/madrigalWeb/>`_
- MANGO: `mangopy <https://github.com/astib/MANGO>`_
- SuperDARN: `davitpy <https://github.com/vtsuperdarn/davitpy>`_
