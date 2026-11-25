# /*##########################################################################
#
# Copyright (c) 2016-2024 European Synchrotron Radiation Facility
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

"""This module initializes matplotlib and sets-up the backend to use.

It MUST be imported prior to any other import of matplotlib.

It provides the matplotlib :class:`FigureCanvasQTAgg` class corresponding
to the used backend.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/05/2018"


import math

import matplotlib
import numpy

from .. import qt

# This must be performed before any import from matplotlib
if qt.BINDING in ("PySide6", "PyQt6", "PyQt5"):
    matplotlib.use("QtAgg", force=False)
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa

else:
    raise ImportError("Unsupported Qt binding: %s" % qt.BINDING)


from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter as _ScalarFormatter
from matplotlib import cbook, font_manager
from matplotlib.backends.backend_agg import RendererAgg


class DefaultTickFormatter(_ScalarFormatter):
    """Tick label formatter"""

    def __init__(self):
        super().__init__(useOffset=True, useMathText=True)
        self.set_scientific(True)
        self.create_dummy_axis()


_FONT_STYLES = {
    qt.QFont.StyleNormal: "normal",
    qt.QFont.StyleItalic: "italic",
    qt.QFont.StyleOblique: "oblique",
}


def qFontToFontProperties(font: qt.QFont):
    """Convert a QFont to a matplotlib FontProperties"""
    weightFactor = 10 if qt.BINDING == "PyQt5" else 1
    families = [font.family(), font.defaultFamily()]

    # Prevent 'Font family not found' warnings
    availableNames = font_manager.get_font_names()
    families = [f for f in families if f in availableNames]
    families.append(font_manager.fontManager.defaultFamily["ttf"])

    if "Sans" in font.family():
        families.insert(0, "sans-serif")

    return FontProperties(
        family=families,
        style=_FONT_STYLES[font.style()],
        weight=weightFactor * font.weight(),
        size=font.pointSizeF(),
    )


def rasterMathText(
    text: str,
    font: qt.QFont,
    dotsPerInch: float = 96.0,
) -> tuple[numpy.ndarray, float]:
    """Raster text using matplotlib supporting latex-like math syntax.

    It supports multiple lines.

    :param text: The text to raster
    :param font: Font to use
    :param dotsPerInch: The DPI resolution of the created image
    :return: Corresponding image in gray scale and baseline offset from top
    """
    font_prop = qFontToFontProperties(font)
    # Measure with the renderer that draws, so that the image fits the drawn text
    renderer = RendererAgg(1, 1, dotsPerInch)

    def measure(line: str, ismath: bool) -> tuple[float, float, float]:
        """Returns the width, ascent and descent of a line"""
        if not line:
            return 0.0, 0.0, 0.0
        width, height, descent = renderer.get_text_width_height_descent(
            line, font_prop, ismath
        )
        return width, height - descent, descent

    lines = []
    for line in text.strip("\n").split("\n"):
        ismath = cbook.is_math_text(line)
        lines.append((line if ismath else line.replace(r"\$", "$"), ismath))
    metrics = [measure(line, ismath) for line, ismath in lines]

    # Use lp string as minimum ascent/descent
    _, ref_ascent, ref_descent = measure("lp", False)
    ascents = [max(ascent, ref_ascent) for _, ascent, _ in metrics]
    descents = [max(descent, ref_descent) for _, _, descent in metrics]
    line_height = 1.2 * (ref_ascent + ref_descent)

    # Pixel aligned baselines, with a margin larger than any glyph overhang
    margin = math.ceil(ref_ascent + ref_descent)
    baselines = [margin + math.ceil(ascents[0])]
    for descent, ascent in zip(descents[:-1], ascents[1:]):
        baselines.append(baselines[-1] + math.ceil(max(line_height, descent + ascent)))

    renderer = RendererAgg(
        math.ceil(max(width for width, _, _ in metrics)) + 2 * margin,
        baselines[-1] + math.ceil(descents[-1]) + margin,
        dotsPerInch,
    )
    gc = renderer.new_gc()
    for (line, ismath), baseline in zip(lines, baselines):
        if line:
            renderer.draw_text(gc, margin, baseline, line, font_prop, 0, ismath)
    gc.restore()
    # Black text on a transparent background: alpha is the coverage
    array = numpy.asarray(renderer.buffer_rgba())[:, :, 3]

    filled_rows = numpy.nonzero(numpy.any(array, axis=1))[0]
    filled_columns = numpy.nonzero(numpy.any(array, axis=0))[0]
    if len(filled_rows) == 0:
        return numpy.ascontiguousarray(array), float(baselines[0])

    # Crop to the text with an empty border, keeping the first line ascent at the top
    top = min(baselines[0] - math.ceil(ascents[0]), filled_rows[0] - 1)
    return (
        numpy.ascontiguousarray(
            array[
                top : filled_rows[-1] + 2,
                filled_columns[0] - 1 : filled_columns[-1] + 2,
            ]
        ),
        float(baselines[0] - top),
    )
