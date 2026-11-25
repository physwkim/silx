# /*##########################################################################
#
# Copyright (c) 2026 European Synchrotron Radiation Facility
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
"""Test of silx.gui.utils.matplotlib"""

import numpy
import pytest

from silx.gui.utils.matplotlib import rasterMathText


@pytest.mark.parametrize("dotsPerInch", (96.0, 192.0))
@pytest.mark.parametrize(
    "text",
    (
        "Custom background/foreground/grid colors",
        "Time-series X axis (3 days)",
        r"$\frac{1}{2}$ y",
        "Ag\npq\ngy",
    ),
)
def test_rasterMathText_border(qapp, text: str, dotsPerInch: float) -> None:
    """The text is not clipped: the image keeps an empty border"""
    array, _ = rasterMathText(text, qapp.font(), dotsPerInch)

    assert not numpy.any(array[[0, -1], :])
    assert not numpy.any(array[:, [0, -1]])


@pytest.mark.parametrize("dotsPerInch", (96.0, 192.0))
def test_rasterMathText_baseline(qapp, dotsPerInch: float) -> None:
    """The baseline offset is the row just below a text without descent"""
    array, baseline = rasterMathText("xxx", qapp.font(), dotsPerInch)

    assert numpy.nonzero(numpy.any(array, axis=1))[0][-1] == baseline - 1
