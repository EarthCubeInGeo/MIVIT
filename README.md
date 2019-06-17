# MIVIT

The Multi-Instrument Visualization Toolkit (MIVIT) provides python tools for plotting and visualizing geospace data from different sources simultaniously.  It was developed as part of the InGeO project, currently supported by the National Science Foundation's Cyberinfrastructure for Sustained Scientific Innovation (CSSI) program (Grant #1835573).

## Installation
1. Clone the MIVIT repository from github.
```
git clone https://github.com/EarthCubeInGeo/MIVIT.git
```
2. Change directories into the MIVIT directory.
```
cd MIVIT
```
3. Use pip to install MIVIT.
```
pip install .
```

This will provide the package mivit, which can be imported in python.

## Basic Usage
MIVIT is designed around the idea that to visualize a data set, you need (1) the data and (2) a set of plotting methods or parameters.  With MIVIT, you initialize a DataSet object with data and a PlotMethod object with plotting parameters, then use both of them together to create a DataVisualization object.  You can even assign multiple PlotMethod objects, allowing you to visualize the same dataset multiple different ways.  Finally, the MIVIT visualize class take a list of DataVisualization objects and creates figures that let you view all the data sets simultaniouslly in different ways.

The mivit_tutorial.ipynb jupyter notebook provides some useful examples of work cases for MIVIT.


