# datavisualization.py

class DataVisualization(object):
    def __init__(self, dataset, plotmethods):
        '''
        Combined dataset and plot methods that specifies both a dataset and how it should be displayed.
        Parameters:
            - dataset: A DataSet object
            - plotmethods: Either a single PlotMethod object or a list of PlotMethod objects
            describing different ways the dataset should be visualized
        '''
        self.dataset = dataset

        if isinstance(plotmethods, list):
            self.plotmethods = plotmethods
        else:
            self.plotmethods = [plotmethods]


