# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""This module adds convenient functions to use plot widgets from the console.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "27/06/2017"


import logging
import numpy

from ..gui.plot import Plot1D, Plot2D
from ..gui.plot.Colors import COLORDICT
from ..gui.plot.Colormap import Colormap
from silx.third_party import six


_logger = logging.getLogger(__name__)


def plot(*args, **kwargs):
    """
    Plot curves in a dedicated :class:`~silx.gui.plot.PlotWindow.Plot1D` widget.

    How to use:

    >>> from silx import sx
    >>> import numpy

    Plot a single curve given some values:

    >>> values = numpy.random.random(100)
    >>> plot_1curve = sx.plot(values, title='Random data')

    Plot a single curve given the x and y values:

    >>> angles = numpy.linspace(0, numpy.pi, 100)
    >>> sin_a = numpy.sin(angles)
    >>> plot_sinus = sx.plot(angles, sin_a,
    ...                      xlabel='angle (radian)', ylabel='sin(a)')

    Plot many curves by giving a 2D array, provided xn, yn arrays:

    >>> plot_curves = sx.plot(x0, y0, x1, y1, x2, y2, ...)

    Plot curve with style giving a style string:

    >>> plot_styled = sx.plot(x0, y0, 'ro-', x1, y1, 'b.')

    Supported symbols:

        - 'o' circle
        - '.' point
        - ',' pixel
        - '+' cross
        - 'x' x-cross
        - 'd' diamond
        - 's' square

    Supported types of line:

            - ' '  no line
            - '-'  solid line
            - '--' dashed line
            - '-.' dash-dot line
            - ':'  dotted line

    If provided, the names arguments color, linestyle, linewidth and marker
    override any style provided to a curve.

    This function supports a subset of `matplotlib.pyplot.plot
    <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_
    arguments.

    :param str color: Color to use for all curves (default: None)
    :param str linestyle: Type of line to use for all curves (default: None)
    :param float linewidth: With of all the curves (default: 1)
    :param str marker: Symbol to use for all the curves (default: None)
    :param str title: The title of the Plot widget (default: None)
    :param str xlabel: The label of the X axis (default: None)
    :param str ylabel: The label of the Y axis (default: None)
    """
    plt = Plot1D()
    if 'title' in kwargs:
        plt.setGraphTitle(kwargs['title'])
    if 'xlabel' in kwargs:
        plt.getXAxis().setLabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.getYAxis().setLabel(kwargs['ylabel'])

    color = kwargs.get('color')
    linestyle = kwargs.get('linestyle')
    linewidth = kwargs.get('linewidth')
    marker = kwargs.get('marker')

    # Parse args and store curves as (x, y, style string)
    args = list(args)
    curves = []
    while args:
        first_arg = args.pop(0)  # Process an arg

        if len(args) == 0:
            # Last curve defined as (y,)
            curves.append((numpy.arange(len(first_arg)), first_arg, None))
        else:
            second_arg = args.pop(0)
            if isinstance(second_arg, six.string_types):
                # curve defined as (y, style)
                y = first_arg
                style = second_arg
                curves.append((numpy.arange(len(y)), y, style))
            else:  # second_arg must be an array-like
                x = first_arg
                y = second_arg
                if len(args) >= 1 and isinstance(args[0], six.string_types):
                    # Curve defined as (x, y, style)
                    style = args.pop(0)
                    curves.append((x, y, style))
                else:
                    # Curve defined as (x, y)
                    curves.append((x, y, None))

    for index, curve in enumerate(curves):
        x, y, style = curve

        # Default style
        curve_symbol, curve_linestyle, curve_color = None, None, None

        # Parse style
        if style:
            # Handle color first
            possible_colors = [c for c in COLORDICT if style.startswith(c)]
            if possible_colors:  # Take the longest string matching a color name
                curve_color = possible_colors[0]
                for c in possible_colors[1:]:
                    if len(c) > len(curve_color):
                        curve_color = c
                style = style[len(curve_color):]

            if style:
                # Run twice to handle inversion symbol/linestyle
                for _i in range(2):
                    # Handle linestyle
                    for line in (' ', '--', '-', '-.', ':'):
                        if style.endswith(line):
                            curve_linestyle = line
                            style = style[:-len(line)]
                            break

                    # Handle symbol
                    for curve_marker in ('o', '.', ',', '+', 'x', 'd', 's'):
                        if style.endswith(curve_marker):
                            curve_symbol = style[-1]
                            style = style[:-1]
                            break

        # As in matplotlib, marker, linestyle and color override other style
        plt.addCurve(x, y,
                     legend=('curve_%d' % index),
                     symbol=marker or curve_symbol,
                     linestyle=linestyle or curve_linestyle,
                     linewidth=linewidth,
                     color=color or curve_color)

    plt.show()
    return plt


def imshow(data=None, cmap=None, norm=Colormap.LINEAR,
           vmin=None, vmax=None,
           aspect=False,
           origin=(0., 0.), scale=(1., 1.),
           title='', xlabel='X', ylabel='Y'):
    """
    Plot an image in a dedicated :class:`~silx.gui.plot.PlotWindow.Plot2D` widget.

    How to use:

    >>> from silx import sx
    >>> import numpy

    >>> data = numpy.random.random(1024 * 1024).reshape(1024, 1024)
    >>> plt = sx.imshow(data, title='Random data')

    This function supports a subset of `matplotlib.pyplot.imshow
    <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow>`_
    arguments.

    :param data: data to plot as an image
    :type data: numpy.ndarray-like with 2 dimensions
    :param str cmap: The name of the colormap to use for the plot.
    :param str norm: The normalization of the colormap:
                     'linear' (default) or 'log'
    :param float vmin: The value to use for the min of the colormap
    :param float vmax: The value to use for the max of the colormap
    :param bool aspect: True to keep aspect ratio (Default: False)
    :param origin: (ox, oy) The coordinates of the image origin in the plot
    :type origin: 2-tuple of floats
    :param scale: (sx, sy) The scale of the image in the plot
                  (i.e., the size of the image's pixel in plot coordinates)
    :type scale: 2-tuple of floats
    :param str title: The title of the Plot widget
    :param str xlabel: The label of the X axis
    :param str ylabel: The label of the Y axis
    """
    plt = Plot2D()
    plt.setGraphTitle(title)
    plt.getXAxis().setLabel(xlabel)
    plt.getYAxis().setLabel(ylabel)

    # Update default colormap with input parameters
    colormap = plt.getDefaultColormap()
    if cmap is not None:
        colormap.setName(cmap)
    assert norm in Colormap.NORMALIZATIONS
    colormap.setNormalization(norm)
    colormap.setVMin(vmin)
    colormap.setVMax(vmax)
    plt.setDefaultColormap(colormap)

    # Handle aspect
    if aspect in (None, False, 'auto', 'normal'):
        plt.setKeepDataAspectRatio(False)
    elif aspect in (True, 'equal') or aspect == 1:
        plt.setKeepDataAspectRatio(True)
    else:
        _logger.warning(
            'imshow: Unhandled aspect argument: %s', str(aspect))

    if data is not None:
        data = numpy.array(data, copy=True)

        assert data.ndim in (2, 3)  # data or RGB(A)
        if data.ndim == 3:
            assert data.shape[-1] in (3, 4)  # RGB(A) image

        plt.addImage(data, origin=origin, scale=scale)

    plt.show()
    return plt
