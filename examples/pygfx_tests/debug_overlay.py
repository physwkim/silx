"""Debug overlay shape rendering in pygfx backend.

Tests that overlay shapes (like zoom selection rectangles) are visible.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import Plot1D


def main():
    app = qt.QApplication([])

    plot = Plot1D(backend="pygfx")
    plot.setWindowTitle("Debug: Overlay Shapes")

    x = numpy.linspace(0, 10, 100)
    y = numpy.sin(x)
    plot.addCurve(x, y, legend="sin", color="blue", linewidth=2)
    plot.addCurve(x, numpy.cos(x), legend="cos", color="red", linewidth=2)

    # Dashed overlay with gap color (like zoom selection rectangle)
    rx = numpy.array([2.0, 2.0, 6.0, 6.0])
    ry = numpy.array([-0.5, 0.5, 0.5, -0.5])
    plot.addShape(
        rx, ry,
        legend="dashed_overlay",
        shape="polygon",
        color="black",
        fill=False,
        overlay=True,
        linestyle="--",
        linewidth=2.0,
        gapcolor="white",
    )

    # Solid overlay
    rx2 = numpy.array([4.0, 4.0, 8.0, 8.0])
    ry2 = numpy.array([-0.8, 0.8, 0.8, -0.8])
    plot.addShape(
        rx2, ry2,
        legend="solid_overlay",
        shape="polygon",
        color="green",
        fill=False,
        overlay=True,
        linestyle="-",
        linewidth=3.0,
    )

    plot.resetZoom()
    plot.show()
    app.exec()


if __name__ == "__main__":
    main()
