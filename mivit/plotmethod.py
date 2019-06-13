# plotmethod.py
import matplotlib.pyplot as plt

class PlotMethod(object):
    def __init__(self,plot_type='scatter',label='',**kwargs):
        '''
        Plotting information for an arbitrary dataset.
        Parameters:
            - plot_type: A string naming a matplotlib pyplot plot method.  Current valid inputs are scatter, pcolormesh, contour, contourf, and quiver.
            - label (optional): A label for this plotting method.  This will be used for the colorbar associated with this plot, if one is created.
        Other Parameters:
            - **kwargs: Other matplotlib keyword arguments that are approprite for the plot.  For contour plots, if levels is not specified, a default value of 20 is used.
        '''

        self.plot_type = plot_type
        self.label = label

        # if 'cmap' argument provided as string, convert to a colormap
        try:
            kwargs['cmap'] = plt.get_cmap(kwargs['cmap'])
        except KeyError:
            pass

        # if plot type is contour or contourf, make sure level keyword is set
        if self.plot_type=='contour' or self.plot_type=='contourf':
            try:
                self.levels = kwargs['levels']
                del kwargs['levels']
            except KeyError:
                self.levels = 20

        self.plot_kwargs = kwargs
