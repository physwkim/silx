# /*##########################################################################
#
# Copyright (c) 2024 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""pygfx (WGPU) Plot backend."""

from __future__ import annotations

__authors__ = ["S. Kim"]
__license__ = "MIT"

import logging
import math
import weakref

import numpy

from rendercanvas.qt import QRenderWidget
import pygfx as gfx

from .. import items
from .._utils import FLOAT32_MINPOS
from . import BackendBase
from ... import colors
from ... import qt
from ._PlotFrameCore import PlotFrame2DCore
from silx.gui.colors import RGBAColorType

_logger = logging.getLogger(__name__)

import re

_MATHDEFAULT_RE = re.compile(r'\$\\mathdefault\{([^}]*)\}\$')


def _stripMathDefault(text):
    """Strip matplotlib's $\\mathdefault{...}$ LaTeX wrapping from tick labels."""
    if text is None:
        return text
    return _MATHDEFAULT_RE.sub(r'\1', text)


# Dash pattern mapping: silx linestyle -> pygfx dash_pattern
# pygfx dash_pattern is a tuple of (dash, gap, ...) relative to line thickness
_DASH_PATTERNS = {
    "": None,
    " ": None,
    "-": None,  # solid
    "--": (3.7, 1.6, 3.7, 1.6),
    "-.": (6.4, 1.6, 1, 1.6),
    ":": (1, 1.65, 1, 1.65),
}


def _lineStyleToDashPattern(linestyle):
    """Convert silx linestyle to pygfx dash_pattern tuple."""
    if linestyle is None or linestyle in ("", " "):
        return None
    if isinstance(linestyle, tuple) and len(linestyle) == 2:
        # Custom (offset, (on, off, on, off, ...))
        return linestyle[1]
    return _DASH_PATTERNS.get(linestyle)


# silx symbol -> pygfx marker shape mapping
_SYMBOL_MAP = {
    "o": "circle",
    ".": "circle",  # smaller via size
    ",": "square",  # pixel
    "+": "plus",
    "x": "cross",
    "d": "diamond",
    "s": "square",
    "^": "triangle_up",
    "v": "triangle_down",
    "<": "triangle_left",
    ">": "triangle_right",
    "*": "asterisk6",
}


def _rgbaToGfxColor(color):
    """Convert silx RGBA color (4-tuple of 0..1 floats) to pygfx Color."""
    if color is None:
        return gfx.Color(1, 1, 1, 1)
    if isinstance(color, str):
        color = colors.rgba(color)
    if len(color) == 3:
        return gfx.Color(*color, 1.0)
    return gfx.Color(*color)


# Item classes ################################################################


class _PygfxCurveItem:
    """Manages pygfx scene objects for a single curve."""

    def __init__(
        self,
        x,
        y,
        color,
        gapcolor,
        symbol,
        linewidth,
        linestyle,
        yaxis,
        xerror,
        yerror,
        fill,
        alpha,
        symbolsize,
        baseline,
    ):
        self.yaxis = yaxis
        self.group = gfx.Group()
        self._lineObj = None
        self._gapLineObj = None
        self._pointsObj = None
        self._errorGroup = None
        self._fillObj = None

        x = numpy.asarray(x, dtype=numpy.float32)
        y = numpy.asarray(y, dtype=numpy.float32)

        # Per-vertex color handling
        if isinstance(color, numpy.ndarray) and color.ndim == 2:
            perVertexColor = True
            vertexColors = numpy.asarray(color, dtype=numpy.float32)
            if vertexColors.shape[1] == 3:
                vertexColors = numpy.column_stack(
                    [vertexColors, numpy.full(len(vertexColors), alpha, dtype=numpy.float32)]
                )
            uniformColor = gfx.Color(1, 1, 1, 1)
        else:
            perVertexColor = False
            vertexColors = None
            rgba = colors.rgba(color)
            uniformColor = gfx.Color(rgba[0], rgba[1], rgba[2], rgba[3] * alpha)

        # Line
        dashPattern = _lineStyleToDashPattern(linestyle)
        hasLine = linestyle not in (None, "", " ")
        if hasLine and len(x) > 1:
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y

            lineKwargs = {}
            if perVertexColor:
                lineKwargs["colors"] = vertexColors

            geom = gfx.Geometry(positions=positions, **lineKwargs)
            mat = gfx.LineMaterial(
                thickness=max(linewidth, 1.0),
                color=uniformColor,
                color_mode="vertex" if perVertexColor else "uniform",
                dash_pattern=dashPattern if dashPattern else (),
            )
            self._lineObj = gfx.Line(geom, mat)
            self.group.add(self._lineObj)

            # Gap color line (behind the dashed line)
            if gapcolor is not None and dashPattern:
                gapRgba = colors.rgba(gapcolor)
                gapMat = gfx.LineMaterial(
                    thickness=max(linewidth, 1.0),
                    color=gfx.Color(*gapRgba),
                )
                self._gapLineObj = gfx.Line(
                    gfx.Geometry(positions=positions.copy()), gapMat
                )
                self._gapLineObj.render_order = -1
                self.group.add(self._gapLineObj)

        # Symbol / Points
        hasSymbol = symbol not in (None, "", " ")
        if hasSymbol:
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y

            markerShape = _SYMBOL_MAP.get(symbol, "circle")
            pointSize = symbolsize if symbol != "," else 1.0
            if symbol == ".":
                pointSize = max(pointSize * 0.5, 1.0)

            pointKwargs = {}
            if perVertexColor:
                pointKwargs["colors"] = vertexColors

            geom = gfx.Geometry(positions=positions, **pointKwargs)
            mat = gfx.PointsMarkerMaterial(
                marker=markerShape,
                size=pointSize,
                color=uniformColor,
                color_mode="vertex" if perVertexColor else "uniform",
                edge_width=0.5,
                edge_color=uniformColor,
            )
            self._pointsObj = gfx.Points(geom, mat)
            self.group.add(self._pointsObj)

        # Error bars
        if xerror is not None or yerror is not None:
            self._errorGroup = gfx.Group()
            errSegments = self._buildErrorBarSegments(x, y, xerror, yerror)
            if len(errSegments) > 0:
                errGeom = gfx.Geometry(
                    positions=numpy.array(errSegments, dtype=numpy.float32)
                )
                errMat = gfx.LineSegmentMaterial(
                    thickness=1.0,
                    color=uniformColor,
                )
                errLine = gfx.Line(errGeom, errMat)
                self._errorGroup.add(errLine)
            self.group.add(self._errorGroup)

        # Fill between curve and baseline
        if fill and len(x) >= 2:
            self._fillObj = self._buildFill(x, y, baseline, uniformColor, alpha)
            if self._fillObj is not None:
                self._fillObj.render_order = -2
                self.group.add(self._fillObj)

    @staticmethod
    def _buildErrorBarSegments(x, y, xerror, yerror):
        """Build line segments for error bars."""
        segments = []
        capSize = 3.0  # in data units (will be small)

        if yerror is not None:
            yerror = numpy.asarray(yerror)
            if yerror.ndim == 1:
                yErrMinus = yerror
                yErrPlus = yerror
            else:
                yErrMinus = yerror[0]
                yErrPlus = yerror[1]
            for i in range(len(x)):
                # Vertical bar
                segments.append([x[i], y[i] - yErrMinus[i], 0])
                segments.append([x[i], y[i] + yErrPlus[i], 0])

        if xerror is not None:
            xerror = numpy.asarray(xerror)
            if xerror.ndim == 1:
                xErrMinus = xerror
                xErrPlus = xerror
            else:
                xErrMinus = xerror[0]
                xErrPlus = xerror[1]
            for i in range(len(x)):
                # Horizontal bar
                segments.append([x[i] - xErrMinus[i], y[i], 0])
                segments.append([x[i] + xErrPlus[i], y[i], 0])

        return segments

    @staticmethod
    def _buildFill(x, y, baseline, color, alpha):
        """Build a filled mesh between curve and baseline."""
        if baseline is None:
            baseY = numpy.zeros_like(y)
        elif isinstance(baseline, numpy.ndarray):
            baseY = baseline
        else:
            baseY = numpy.full_like(y, float(baseline))

        n = len(x)
        # Create triangle strip: for each segment, two triangles
        vertices = []
        indices = []
        for i in range(n):
            vertices.append([x[i], y[i], 0])
            vertices.append([x[i], baseY[i], 0])

        for i in range(n - 1):
            idx = i * 2
            indices.append([idx, idx + 1, idx + 2])
            indices.append([idx + 1, idx + 3, idx + 2])

        if not indices:
            return None

        vertices = numpy.array(vertices, dtype=numpy.float32)
        indices = numpy.array(indices, dtype=numpy.int32)

        fillColor = gfx.Color(color.r, color.g, color.b, alpha * 0.5)
        geom = gfx.Geometry(positions=vertices, indices=indices)
        mat = gfx.MeshBasicMaterial(color=fillColor, side="both")
        return gfx.Mesh(geom, mat)


class _PygfxImageItem:
    """Manages pygfx scene objects for a single image."""

    def __init__(self, data, origin, scale, colormap, alpha):
        self.group = gfx.Group()
        self.yaxis = "left"

        data = numpy.asarray(data)

        if data.ndim == 2:
            # Scalar data - apply colormap on CPU
            if colormap is not None:
                rgbaData = colormap.applyToData(data)
            else:
                # Fallback: normalize to grayscale
                vmin, vmax = numpy.nanmin(data), numpy.nanmax(data)
                if vmin == vmax:
                    normalized = numpy.zeros_like(data, dtype=numpy.uint8)
                else:
                    normalized = (
                        (data - vmin) / (vmax - vmin) * 255
                    ).astype(numpy.uint8)
                rgbaData = numpy.stack(
                    [normalized, normalized, normalized,
                     numpy.full_like(normalized, 255)],
                    axis=-1,
                )
        elif data.ndim == 3 and data.shape[2] in (3, 4):
            if data.dtype == numpy.float64:
                data = data.astype(numpy.float32)
            if data.dtype in (numpy.float32, numpy.float64):
                rgbaData = (numpy.clip(data, 0, 1) * 255).astype(numpy.uint8)
            else:
                rgbaData = numpy.asarray(data, dtype=numpy.uint8)
            if rgbaData.shape[2] == 3:
                alphaChannel = numpy.full(
                    rgbaData.shape[:2] + (1,), 255, dtype=numpy.uint8
                )
                rgbaData = numpy.concatenate([rgbaData, alphaChannel], axis=-1)
        else:
            _logger.warning("Unsupported image data shape: %s", data.shape)
            return

        # Convert to float32 [0, 1] to avoid pygfx climcorrection issues
        # (rgba8unorm triggers * 255.0 in shader which breaks with default maprange)
        rgbaFloat = rgbaData.astype(numpy.float32) / 255.0

        # Apply alpha
        if alpha < 1.0:
            rgbaFloat = rgbaFloat.copy()
            rgbaFloat[:, :, 3] *= alpha

        # pygfx Image maps texcoord (0,0) to the bottom-left of the quad,
        # which matches silx's bottom-left origin convention. No flip needed.
        rgbaFloat = numpy.ascontiguousarray(rgbaFloat)

        geom = gfx.Geometry(grid=gfx.Texture(rgbaFloat, dim=2))
        mat = gfx.ImageBasicMaterial(interpolation="nearest")
        self._imageObj = gfx.Image(geom, mat)

        # Position and scale
        ox, oy = origin
        sx, sy = scale
        h, w = data.shape[:2]

        self._imageObj.local.position = (ox, oy, 0)
        self._imageObj.local.scale = (sx, sy, 1)

        self.group.add(self._imageObj)


class _PygfxTrianglesItem:
    """Manages pygfx scene objects for triangles."""

    def __init__(self, x, y, triangles, color, alpha):
        self.group = gfx.Group()
        self.yaxis = "left"

        x = numpy.asarray(x, dtype=numpy.float32)
        y = numpy.asarray(y, dtype=numpy.float32)
        triangles = numpy.asarray(triangles, dtype=numpy.int32)

        positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
        positions[:, 0] = x
        positions[:, 1] = y

        color = numpy.asarray(color, dtype=numpy.float32)
        if color.ndim == 2:
            if color.shape[1] == 3:
                color = numpy.column_stack(
                    [color, numpy.full(len(color), alpha, dtype=numpy.float32)]
                )
            geom = gfx.Geometry(
                positions=positions, indices=triangles, colors=color
            )
            mat = gfx.MeshBasicMaterial(color_mode="vertex", side="both")
        else:
            rgba = colors.rgba(color)
            geom = gfx.Geometry(positions=positions, indices=triangles)
            mat = gfx.MeshBasicMaterial(
                color=gfx.Color(rgba[0], rgba[1], rgba[2], rgba[3] * alpha),
                side="both",
            )

        self._meshObj = gfx.Mesh(geom, mat)
        self.group.add(self._meshObj)


class _PygfxShapeItem(dict):
    """Manages pygfx scene objects for shapes."""

    def __init__(
        self, x, y, shape, color, fill, overlay, linewidth,
        linestyle, gapcolor,
    ):
        super().__init__()

        if shape not in ("polygon", "rectangle", "line", "vline", "hline", "polylines"):
            raise NotImplementedError(f"Unsupported shape {shape}")

        x = numpy.asarray(x, dtype=numpy.float32)
        y = numpy.asarray(y, dtype=numpy.float32)

        if shape == "rectangle":
            xMin, xMax = x
            x = numpy.array((xMin, xMin, xMax, xMax), dtype=numpy.float32)
            yMin, yMax = y
            y = numpy.array((yMin, yMax, yMax, yMin), dtype=numpy.float32)

        fill = fill if shape != "polylines" else False

        rgba = colors.rgba(color)
        dashPattern = _lineStyleToDashPattern(linestyle)

        self.update({
            "shape": shape,
            "color": rgba,
            "fill": fill,
            "x": x,
            "y": y,
            "linewidth": linewidth,
            "overlay": overlay,
        })

        self.group = gfx.Group()

        gfxColor = gfx.Color(*rgba)

        # Build outline
        if shape in ("polygon", "rectangle"):
            positions = numpy.zeros((len(x) + 1, 3), dtype=numpy.float32)
            positions[:-1, 0] = x
            positions[:-1, 1] = y
            positions[-1, 0] = x[0]
            positions[-1, 1] = y[0]
        elif shape == "polylines":
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y
        elif shape in ("line", "hline", "vline"):
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y
        else:
            positions = numpy.zeros((len(x), 3), dtype=numpy.float32)
            positions[:, 0] = x
            positions[:, 1] = y

        if len(positions) >= 2:
            geom = gfx.Geometry(positions=positions)
            mat = gfx.LineMaterial(
                thickness=max(linewidth, 1.0),
                color=gfxColor,
                dash_pattern=dashPattern if dashPattern else (),
            )
            lineObj = gfx.Line(geom, mat)
            self.group.add(lineObj)

            # Gap color line
            if gapcolor is not None and dashPattern:
                gapRgba = colors.rgba(gapcolor)
                gapMat = gfx.LineMaterial(
                    thickness=max(linewidth, 1.0),
                    color=gfx.Color(*gapRgba),
                )
                gapLineObj = gfx.Line(
                    gfx.Geometry(positions=positions.copy()), gapMat
                )
                gapLineObj.render_order = -1
                self.group.add(gapLineObj)

        # Build fill for closed shapes
        if fill and shape in ("polygon", "rectangle") and len(x) >= 3:
            fillMesh = self._buildFilledShape(x, y, rgba)
            if fillMesh is not None:
                fillMesh.render_order = -1
                self.group.add(fillMesh)

    @staticmethod
    def _buildFilledShape(x, y, rgba):
        """Create a filled mesh using ear-clipping triangulation."""
        n = len(x)
        if n < 3:
            return None

        positions = numpy.zeros((n, 3), dtype=numpy.float32)
        positions[:, 0] = x
        positions[:, 1] = y

        # Simple fan triangulation (works for convex shapes)
        indices = []
        for i in range(1, n - 1):
            indices.append([0, i, i + 1])

        if not indices:
            return None

        indices = numpy.array(indices, dtype=numpy.int32)
        fillColor = gfx.Color(rgba[0], rgba[1], rgba[2], rgba[3] * 0.5)
        geom = gfx.Geometry(positions=positions, indices=indices)
        mat = gfx.MeshBasicMaterial(color=fillColor, side="both")
        return gfx.Mesh(geom, mat)


class _PygfxMarkerItem(dict):
    """Manages pygfx scene objects for markers."""

    def __init__(
        self, x, y, text, color, symbol, symbolsize, linewidth,
        linestyle, constraint, yaxis, font, bgcolor,
    ):
        super().__init__()

        if symbol is None:
            symbol = "+"

        # Apply constraint
        isConstraint = constraint is not None and x is not None and y is not None
        if isConstraint:
            x, y = constraint(x, y)

        dashPattern = _lineStyleToDashPattern(linestyle)

        self.update({
            "x": x,
            "y": y,
            "text": text,
            "color": colors.rgba(color),
            "constraint": constraint if isConstraint else None,
            "symbol": symbol,
            "symbolsize": symbolsize,
            "linewidth": linewidth,
            "linestyle": linestyle,
            "dashpattern": dashPattern,
            "yaxis": yaxis,
            "font": font,
            "bgcolor": bgcolor,
        })

        self.group = gfx.Group()
        rgba = colors.rgba(color)
        gfxColor = gfx.Color(*rgba)

        if x is not None and y is not None:
            # Point marker
            positions = numpy.array([[x, y, 0]], dtype=numpy.float32)
            markerShape = _SYMBOL_MAP.get(symbol, "plus")
            geom = gfx.Geometry(positions=positions)
            mat = gfx.PointsMarkerMaterial(
                marker=markerShape,
                size=symbolsize,
                color=gfxColor,
                edge_width=1.0,
                edge_color=gfxColor,
            )
            self._pointsObj = gfx.Points(geom, mat)
            self.group.add(self._pointsObj)

        # Text label will be rendered in screen-space overlay
        # (handled during draw, not here)


# BackendPygfx ################################################################


class BackendPygfx(BackendBase.BackendBase, QRenderWidget):
    """pygfx/WGPU-based Plot backend.

    Uses pygfx for GPU-accelerated rendering via WGPU (Vulkan/Metal/DX12).
    """

    _TEXT_MARKER_PADDING = 4

    def __init__(self, plot, parent=None):
        QRenderWidget.__init__(self, parent=parent)
        BackendBase.BackendBase.__init__(self, plot, parent)

        self._defaultFont = None

        self._backgroundColor = (1.0, 1.0, 1.0, 1.0)
        self._dataBackgroundColor = (1.0, 1.0, 1.0, 1.0)

        self._keepDataAspectRatio = False
        self._crosshairCursor = None
        self._mousePosInPixels = None

        # pygfx rendering objects
        self._renderer = gfx.WgpuRenderer(self)
        self._scene = gfx.Scene()

        # Camera: orthographic for 2D plotting
        self._camera = gfx.OrthographicCamera(640, 480, maintain_aspect=False)

        # Scene hierarchy
        self._bgGroup = gfx.Group()
        self._dataGroup = gfx.Group()
        self._overlayGroup = gfx.Group()
        self._frameGroup = gfx.Group()

        self._scene.add(self._bgGroup)
        self._scene.add(self._dataGroup)
        self._scene.add(self._overlayGroup)
        self._scene.add(self._frameGroup)

        # PlotFrame for coordinate transforms
        self._plotFrame = PlotFrame2DCore(
            foregroundColor=(0.0, 0.0, 0.0, 1.0),
            gridColor=(0.7, 0.7, 0.7, 1.0),
            marginRatios=(0.15, 0.1, 0.1, 0.15),
            font=self._getDefaultFont(),
        )
        self._plotFrame.size = (
            int(self.getDevicePixelRatio() * 640),
            int(self.getDevicePixelRatio() * 480),
        )

        # Screen-space scene for frame/axes rendering (PR 9)
        self._screenScene = gfx.Scene()
        self._screenScene.add(
            gfx.Background(None, gfx.BackgroundMaterial(gfx.Color(1, 1, 1, 1)))
        )
        self._screenCamera = gfx.OrthographicCamera(maintain_aspect=False)

        # Frame rendering objects (populated by _updateFrame)
        self._frameLines = None
        self._gridLines = None
        self._frameTexts = []
        self._titleText = None

        # Crosshair cursor lines
        self._crosshairHLine = None
        self._crosshairVLine = None

        self.request_draw(self._draw)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

    def _getDefaultFont(self):
        if self._defaultFont is None:
            app = qt.QApplication.instance()
            if app is not None:
                self._defaultFont = app.font()
            else:
                self._defaultFont = qt.QFont()
        return self._defaultFont

    def getDevicePixelRatio(self):
        return self.devicePixelRatioF()

    def getDotsPerInch(self):
        screen = self.screen()
        if screen is not None:
            return screen.logicalDotsPerInch() * self.getDevicePixelRatio()
        return 92

    # Drawing ###############################################################

    def _draw(self):
        plot = self._plotRef()
        if plot is None:
            return

        with plot._paintContext():
            self._syncPlotFrame()
            self._syncCamera()
            self._updateFrame()
            self._updateCrosshair()

            # First pass: render frame (background + axes) in full widget
            self._renderer.render(self._screenScene, self._screenCamera, flush=False)

            # Second pass: render data scene in plot area viewport only
            dpr = self.getDevicePixelRatio()
            left, top = self._plotFrame.plotOrigin
            pw, ph = self._plotFrame.plotSize
            # Viewport rect is in logical pixels
            plotRect = (left / dpr, top / dpr, pw / dpr, ph / dpr)
            self._renderer.render(
                self._scene, self._camera, rect=plotRect, flush=True, clear=False
            )

    def _syncPlotFrame(self):
        """Sync plot frame size with widget size."""
        dpr = self.getDevicePixelRatio()
        w = int(self.width() * dpr)
        h = int(self.height() * dpr)
        if (w, h) != self._plotFrame.size:
            self._plotFrame.size = (w, h)
        self._plotFrame.devicePixelRatio = dpr
        self._plotFrame.dotsPerInch = self.getDotsPerInch()

    def _syncCamera(self):
        """Update camera to match the current data ranges."""
        trRanges = self._plotFrame.transformedDataRanges
        xMin, xMax = trRanges.x
        yMin, yMax = trRanges.y

        if self._plotFrame.isXAxisInverted:
            xMin, xMax = xMax, xMin
        if self._plotFrame.isYAxisInverted:
            yMin, yMax = yMax, yMin

        # Ensure non-zero extent to avoid camera errors
        if xMin == xMax:
            xMin -= 0.5
            xMax += 0.5
        if yMin == yMax:
            yMin -= 0.5
            yMax += 0.5

        # show_rect(left, right, top, bottom)
        # height = bottom - top; positive height means Y increases upward
        # top=yMin, bottom=yMax → yMax at top of viewport, yMin at bottom
        extent = max(abs(xMax - xMin), abs(yMax - yMin), 1.0)
        self._camera.show_rect(xMin, xMax, yMin, yMax, depth=extent)

    def _updateFrame(self):
        """Update axes, ticks, grid, labels in screen space."""
        # Clear previous frame objects
        for child in list(self._screenScene.children):
            self._screenScene.remove(child)

        if self._plotFrame.margins == self._plotFrame._NoDisplayMargins:
            return

        w, h = self._plotFrame.size
        if w <= 2 or h <= 2:
            return

        # Set screen camera to pixel coordinates (Y=0 at top, Y=h at bottom)
        # show_rect(left, right, top, bottom):
        # PlotFrameCore uses Y=0=top, Y=h=bottom (pixel convention)
        # In pygfx: height = bottom - top, so top=h, bottom=0 flips Y axis
        extent = max(w, h, 1.0)
        self._screenCamera.show_rect(0, w, h, 0, depth=extent)

        # Build vertices and labels from the core
        vertices, gridVertices, labelDicts = self._plotFrame._buildVerticesAndLabels()
        self._plotFrame._clearDirty()

        # Render grid lines
        if len(gridVertices) >= 2:
            gridColor = gfx.Color(*self._plotFrame.gridColor)
            geom = gfx.Geometry(
                positions=numpy.column_stack([
                    gridVertices,
                    numpy.zeros(len(gridVertices), dtype=numpy.float32)
                ])
            )
            mat = gfx.LineSegmentMaterial(thickness=1.0, color=gridColor)
            gridLine = gfx.Line(geom, mat)
            self._screenScene.add(gridLine)

        # Render frame lines (axes)
        if len(vertices) >= 2:
            fgColor = gfx.Color(*self._plotFrame.foregroundColor)
            geom = gfx.Geometry(
                positions=numpy.column_stack([
                    vertices,
                    numpy.zeros(len(vertices), dtype=numpy.float32)
                ])
            )
            mat = gfx.LineSegmentMaterial(thickness=1.0, color=fgColor)
            frameLine = gfx.Line(geom, mat)
            self._screenScene.add(frameLine)

        # Render text labels (tick labels, axis titles, main title)
        for labelDict in labelDicts:
            text = labelDict.get("text", "")
            if not text:
                continue
            # Strip matplotlib LaTeX formatting
            text = _stripMathDefault(text)

            lx = labelDict["x"]
            ly = labelDict["y"]
            labelColor = labelDict.get("color", (0, 0, 0, 1))
            rotate = labelDict.get("rotate", 0)

            # Map alignment strings to pygfx anchor
            align = labelDict.get("align", "center")
            valign = labelDict.get("valign", "center")
            anchor = self._mapAnchor(align, valign)

            fontSize = 12.0
            font = labelDict.get("font")
            if font is not None:
                ps = font.pointSizeF()
                if ps > 0:
                    fontSize = ps
                else:
                    px = font.pixelSize()
                    if px > 0:
                        fontSize = px * 72.0 / self.getDotsPerInch()

            textObj = gfx.Text(
                text=text,
                material=gfx.TextMaterial(color=gfx.Color(*labelColor)),
                font_size=fontSize,
                anchor=anchor,
                screen_space=True,
            )
            textObj.local.position = (lx, ly, 0)

            if rotate:
                import pylinalg as la
                # Negate angle because screen camera flips Y
                textObj.local.rotation = la.quat_from_axis_angle(
                    (0, 0, 1), math.radians(-rotate)
                )

            self._screenScene.add(textObj)

    def _updateCrosshair(self):
        """Update crosshair cursor lines."""
        # Remove old crosshair
        if self._crosshairHLine is not None:
            if self._crosshairHLine in self._screenScene.children:
                self._screenScene.remove(self._crosshairHLine)
            self._crosshairHLine = None
        if self._crosshairVLine is not None:
            if self._crosshairVLine in self._screenScene.children:
                self._screenScene.remove(self._crosshairVLine)
            self._crosshairVLine = None

        if self._crosshairCursor is None or self._mousePosInPixels is None:
            return

        color, linewidth = self._crosshairCursor
        gfxColor = gfx.Color(*color)
        mx, my = self._mousePosInPixels

        w, h = self._plotFrame.size
        left, top = self._plotFrame.plotOrigin
        pw, ph = self._plotFrame.plotSize

        # Horizontal line
        hPositions = numpy.array([
            [left, my, 0],
            [left + pw, my, 0],
        ], dtype=numpy.float32)
        hGeom = gfx.Geometry(positions=hPositions)
        hMat = gfx.LineMaterial(thickness=linewidth, color=gfxColor)
        self._crosshairHLine = gfx.Line(hGeom, hMat)
        self._screenScene.add(self._crosshairHLine)

        # Vertical line
        vPositions = numpy.array([
            [mx, top, 0],
            [mx, top + ph, 0],
        ], dtype=numpy.float32)
        vGeom = gfx.Geometry(positions=vPositions)
        vMat = gfx.LineMaterial(thickness=linewidth, color=gfxColor)
        self._crosshairVLine = gfx.Line(vGeom, vMat)
        self._screenScene.add(self._crosshairVLine)

    @staticmethod
    def _mapAnchor(align, valign):
        """Map silx align/valign strings to pygfx anchor string."""
        vmap = {"top": "top", "bottom": "bottom", "center": "middle"}
        hmap = {"left": "left", "right": "right", "center": "center"}
        v = vmap.get(str(valign), "middle")
        h = hmap.get(str(align), "center")
        return f"{v}-{h}"

    # QWidget events ########################################################

    _MOUSE_BTNS = {
        qt.Qt.LeftButton: "left",
        qt.Qt.RightButton: "right",
        qt.Qt.MiddleButton: "middle",
    }

    def sizeHint(self):
        return qt.QSize(8 * 80, 6 * 80)

    def mousePressEvent(self, event):
        if event.button() not in self._MOUSE_BTNS:
            return super().mousePressEvent(event)
        x, y = qt.getMouseEventPosition(event)
        self._plot.onMousePress(x, y, self._MOUSE_BTNS[event.button()])
        event.accept()

    def mouseMoveEvent(self, event):
        qtPos = qt.getMouseEventPosition(event)

        previousMousePosInPixels = self._mousePosInPixels
        if qtPos == self._mouseInPlotArea(*qtPos):
            dpr = self.getDevicePixelRatio()
            devicePos = qtPos[0] * dpr, qtPos[1] * dpr
            self._mousePosInPixels = devicePos
        else:
            self._mousePosInPixels = None

        if (
            self._crosshairCursor is not None
            and previousMousePosInPixels != self._mousePosInPixels
        ):
            self._plot._setDirtyPlot(overlayOnly=True)

        self._plot.onMouseMove(*qtPos)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() not in self._MOUSE_BTNS:
            return super().mouseReleaseEvent(event)
        x, y = qt.getMouseEventPosition(event)
        self._plot.onMouseRelease(x, y, self._MOUSE_BTNS[event.button()])
        event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        angleInDegrees = delta / 8.0
        x, y = qt.getMouseEventPosition(event)
        self._plot.onMouseWheel(x, y, angleInDegrees)
        event.accept()

    def leaveEvent(self, _):
        self._plot.onMouseLeaveWidget()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        dpr = self.getDevicePixelRatio()
        self._plotFrame.size = (
            int(self.width() * dpr),
            int(self.height() * dpr),
        )

    # Backend API: Add methods ##############################################

    def addCurve(
        self, x, y, color, gapcolor, symbol, linewidth, linestyle,
        yaxis, xerror, yerror, fill, alpha, symbolsize, baseline,
    ):
        item = _PygfxCurveItem(
            x, y, color, gapcolor, symbol, linewidth, linestyle,
            yaxis, xerror, yerror, fill, alpha, symbolsize, baseline,
        )
        self._dataGroup.add(item.group)
        return item

    def addImage(self, data, origin, scale, colormap, alpha):
        item = _PygfxImageItem(data, origin, scale, colormap, alpha)
        self._dataGroup.add(item.group)
        return item

    def addTriangles(self, x, y, triangles, color, alpha):
        item = _PygfxTrianglesItem(x, y, triangles, color, alpha)
        self._dataGroup.add(item.group)
        return item

    def addShape(
        self, x, y, shape, color, fill, overlay, linestyle, linewidth, gapcolor,
    ):
        item = _PygfxShapeItem(
            x, y, shape, color, fill, overlay, linewidth, linestyle, gapcolor,
        )
        if overlay:
            self._overlayGroup.add(item.group)
        else:
            self._dataGroup.add(item.group)
        return item

    def addMarker(
        self,
        x: float | None,
        y: float | None,
        text: str | None,
        color: str,
        symbol: str | None,
        symbolsize: float,
        linestyle: str | tuple[float, tuple[float, ...] | None],
        linewidth: float,
        constraint,
        yaxis: str,
        font: qt.QFont,
        bgcolor: RGBAColorType | None,
    ) -> object:
        item = _PygfxMarkerItem(
            x, y, text, color, symbol, symbolsize, linewidth,
            linestyle, constraint, yaxis, font, bgcolor,
        )

        # Add marker lines for hline/vline markers
        if x is None or y is None:
            rgba = colors.rgba(color)
            gfxColor = gfx.Color(*rgba)
            dashPattern = _lineStyleToDashPattern(linestyle)

            # These will be updated in _renderMarkerLines during draw
            # For now, create with placeholder positions
            if x is None:
                # Horizontal line at y
                positions = numpy.array([
                    [-1e10, y, 0],
                    [1e10, y, 0],
                ], dtype=numpy.float32)
            else:
                # Vertical line at x
                positions = numpy.array([
                    [x, -1e10, 0],
                    [x, 1e10, 0],
                ], dtype=numpy.float32)

            geom = gfx.Geometry(positions=positions)
            mat = gfx.LineMaterial(
                thickness=max(linewidth, 1.0),
                color=gfxColor,
                dash_pattern=dashPattern if dashPattern else (),
            )
            lineObj = gfx.Line(geom, mat)
            item.group.add(lineObj)

        self._overlayGroup.add(item.group)
        return item

    # Backend API: Remove ####################################################

    def remove(self, item):
        if hasattr(item, "group"):
            # Check Y2 axis visibility
            if hasattr(item, "yaxis") and item.yaxis == "right":
                y2AxisItems = (
                    i
                    for i in self._plot.getItems()
                    if isinstance(i, items.YAxisMixIn) and i.getYAxis() == "right"
                )
                self._plotFrame.isY2Axis = next(y2AxisItems, None) is not None

            group = item.group
            if group.parent is not None:
                group.parent.remove(group)

    # Backend API: Interaction ###############################################

    _QT_CURSORS = {
        BackendBase.CURSOR_DEFAULT: qt.Qt.ArrowCursor,
        BackendBase.CURSOR_POINTING: qt.Qt.PointingHandCursor,
        BackendBase.CURSOR_SIZE_HOR: qt.Qt.SizeHorCursor,
        BackendBase.CURSOR_SIZE_VER: qt.Qt.SizeVerCursor,
        BackendBase.CURSOR_SIZE_ALL: qt.Qt.SizeAllCursor,
    }

    def setGraphCursorShape(self, cursor):
        if cursor is None:
            super().unsetCursor()
        else:
            cursor = self._QT_CURSORS[cursor]
            super().setCursor(qt.QCursor(cursor))

    def setGraphCursor(self, flag, color, linewidth, linestyle):
        if flag:
            color = colors.rgba(color)
            crosshairCursor = color, linewidth
        else:
            crosshairCursor = None

        if crosshairCursor != self._crosshairCursor:
            self._crosshairCursor = crosshairCursor

    _PICK_OFFSET = 3

    def _mouseInPlotArea(self, x, y):
        """Returns closest visible position in the plot."""
        left, top, width, height = self.getPlotBoundsInPixels()
        return (
            numpy.clip(x, left, left + width - 1),
            numpy.clip(y, top, top + height - 1),
        )

    def pickItem(self, x, y, item):
        dataPos = self._plot.pixelToData(x, y, axis="left", check=True)
        if dataPos is None:
            return None

        if item is None:
            _logger.error("No item provided for picking")
            return None

        # Pick markers
        if isinstance(item, _PygfxMarkerItem):
            yaxis = item["yaxis"]
            pixelPos = self._plot.dataToPixel(
                item["x"], item["y"], axis=yaxis, check=False
            )
            if pixelPos is None:
                return None

            if item["x"] is None:  # Horizontal line
                pt1 = self._plot.pixelToData(
                    x, y - self._PICK_OFFSET, axis=yaxis, check=False
                )
                pt2 = self._plot.pixelToData(
                    x, y + self._PICK_OFFSET, axis=yaxis, check=False
                )
                isPicked = min(pt1[1], pt2[1]) <= item["y"] <= max(pt1[1], pt2[1])

            elif item["y"] is None:  # Vertical line
                pt1 = self._plot.pixelToData(
                    x - self._PICK_OFFSET, y, axis=yaxis, check=False
                )
                pt2 = self._plot.pixelToData(
                    x + self._PICK_OFFSET, y, axis=yaxis, check=False
                )
                isPicked = min(pt1[0], pt2[0]) <= item["x"] <= max(pt1[0], pt2[0])

            else:
                isPicked = (
                    numpy.fabs(x - pixelPos[0]) <= self._PICK_OFFSET
                    and numpy.fabs(y - pixelPos[1]) <= self._PICK_OFFSET
                )

            return (0,) if isPicked else None

        # Pick curves
        if isinstance(item, _PygfxCurveItem):
            return self._pickCurve(item, x, y)

        # Pick images
        if isinstance(item, _PygfxImageItem):
            return self._pickImage(item, dataPos)

        # Pick triangles
        if isinstance(item, _PygfxTrianglesItem):
            return None  # TODO

        return None

    def _pickCurve(self, item, x, y):
        """Pick a curve item."""
        offset = self._PICK_OFFSET

        inAreaPos = self._mouseInPlotArea(x - offset, y - offset)
        dataPos = self._plot.pixelToData(
            inAreaPos[0], inAreaPos[1], axis=item.yaxis, check=True
        )
        if dataPos is None:
            return None
        xPick0, yPick0 = dataPos

        inAreaPos = self._mouseInPlotArea(x + offset, y + offset)
        dataPos = self._plot.pixelToData(
            inAreaPos[0], inAreaPos[1], axis=item.yaxis, check=True
        )
        if dataPos is None:
            return None
        xPick1, yPick1 = dataPos

        xPickMin = min(xPick0, xPick1)
        xPickMax = max(xPick0, xPick1)
        yPickMin = min(yPick0, yPick1)
        yPickMax = max(yPick0, yPick1)

        # Get curve data from the line geometry
        if item._lineObj is not None:
            positions = item._lineObj.geometry.positions.data
            xData = positions[:, 0]
            yData = positions[:, 1]
        elif item._pointsObj is not None:
            positions = item._pointsObj.geometry.positions.data
            xData = positions[:, 0]
            yData = positions[:, 1]
        else:
            return None

        # Find points within the pick area
        indices = numpy.where(
            (xData >= xPickMin) & (xData <= xPickMax)
            & (yData >= yPickMin) & (yData <= yPickMax)
        )[0]

        if len(indices) > 0:
            return indices
        return None

    def _pickImage(self, item, dataPos):
        """Pick an image item."""
        # TODO: implement proper image picking
        return None

    # Backend API: Update curve ##############################################

    def setCurveColor(self, curve, color):
        pass  # TODO

    # Backend API: Widget ####################################################

    def getWidgetHandle(self):
        return self

    def postRedisplay(self):
        self.update()

    def replot(self):
        self.update()

    def saveGraph(self, fileName, fileFormat, dpi):
        if dpi is not None:
            _logger.warning("saveGraph ignores dpi parameter")

        if fileFormat not in ["png", "ppm", "svg", "tif", "tiff"]:
            raise NotImplementedError("Unsupported format: %s" % fileFormat)

        # Force a synchronous render
        self._draw()
        snapshot = self._renderer.snapshot()

        # snapshot is (H, W, 4) RGBA uint8
        from PIL import Image as PILImage

        img = PILImage.fromarray(snapshot)
        if fileFormat in ("tif", "tiff"):
            img.save(fileName, format="TIFF")
        elif fileFormat == "ppm":
            img.convert("RGB").save(fileName, format="PPM")
        elif fileFormat == "svg":
            raise NotImplementedError("SVG export not supported by pygfx backend")
        else:
            img.save(fileName, format=fileFormat.upper())

    # Backend API: Labels ####################################################

    def setGraphTitle(self, title):
        self._plotFrame.title = title

    def setGraphXLabel(self, label):
        self._plotFrame.xAxis.title = label

    def setGraphYLabel(self, label, axis):
        if axis == "left":
            self._plotFrame.yAxis.title = label
        else:
            self._plotFrame.y2Axis.title = label

    # Backend API: Limits ####################################################

    def _setDataRanges(self, xlim=None, ylim=None, y2lim=None):
        self._plotFrame.setDataRanges(xlim, ylim, y2lim)

    def _ensureAspectRatio(self, keepDim=None):
        plotWidth, plotHeight = self._plotFrame.plotSize
        if plotWidth <= 2 or plotHeight <= 2:
            return

        if keepDim is None:
            ranges = self._plot.getDataRange()
            if (
                ranges.y is not None
                and ranges.x is not None
                and (ranges.y[1] - ranges.y[0]) != 0.0
            ):
                dataRatio = (ranges.x[1] - ranges.x[0]) / float(
                    ranges.y[1] - ranges.y[0]
                )
                plotRatio = plotWidth / float(plotHeight)
                keepDim = "x" if dataRatio > plotRatio else "y"
            else:
                keepDim = "x"

        (xMin, xMax), (yMin, yMax), (y2Min, y2Max) = self._plotFrame.dataRanges
        if keepDim == "y":
            dataW = (yMax - yMin) * plotWidth / float(plotHeight)
            xCenter = 0.5 * (xMin + xMax)
            xMin = xCenter - 0.5 * dataW
            xMax = xCenter + 0.5 * dataW
        elif keepDim == "x":
            dataH = (xMax - xMin) * plotHeight / float(plotWidth)
            yCenter = 0.5 * (yMin + yMax)
            yMin = yCenter - 0.5 * dataH
            yMax = yCenter + 0.5 * dataH
            y2Center = 0.5 * (y2Min + y2Max)
            y2Min = y2Center - 0.5 * dataH
            y2Max = y2Center + 0.5 * dataH
        else:
            raise RuntimeError("Unsupported dimension to keep: %s" % keepDim)

        self._setDataRanges(
            xlim=(xMin, xMax), ylim=(yMin, yMax), y2lim=(y2Min, y2Max)
        )

    def _setPlotBounds(self, xRange=None, yRange=None, y2Range=None, keepDim=None):
        self._setDataRanges(xlim=xRange, ylim=yRange, y2lim=y2Range)
        if self.isKeepDataAspectRatio():
            self._ensureAspectRatio(keepDim)

    def setLimits(self, xmin, xmax, ymin, ymax, y2min=None, y2max=None):
        if y2min is None or y2max is None:
            y2Range = None
        else:
            y2Range = y2min, y2max
        self._setPlotBounds((xmin, xmax), (ymin, ymax), y2Range)

    def getGraphXLimits(self):
        return self._plotFrame.dataRanges.x

    def setGraphXLimits(self, xmin, xmax):
        self._setPlotBounds(xRange=(xmin, xmax), keepDim="x")

    def getGraphYLimits(self, axis):
        assert axis in ("left", "right")
        if axis == "left":
            return self._plotFrame.dataRanges.y
        else:
            return self._plotFrame.dataRanges.y2

    def setGraphYLimits(self, ymin, ymax, axis):
        assert axis in ("left", "right")
        if axis == "left":
            self._setPlotBounds(yRange=(ymin, ymax), keepDim="y")
        else:
            self._setPlotBounds(y2Range=(ymin, ymax), keepDim="y")

    # Backend API: Axes ######################################################

    def getXAxisTimeZone(self):
        return self._plotFrame.xAxis.timeZone

    def setXAxisTimeZone(self, tz):
        self._plotFrame.xAxis.timeZone = tz

    def isXAxisTimeSeries(self):
        return self._plotFrame.xAxis.isTimeSeries

    def setXAxisTimeSeries(self, isTimeSeries):
        self._plotFrame.xAxis.isTimeSeries = isTimeSeries

    def setXAxisLogarithmic(self, flag):
        if flag != self._plotFrame.xAxis.isLog:
            if flag and self._keepDataAspectRatio:
                _logger.warning("KeepDataAspectRatio is ignored with log axes")
            self._plotFrame.xAxis.isLog = flag

    def setYAxisLogarithmic(self, flag):
        if flag != self._plotFrame.yAxis.isLog or flag != self._plotFrame.y2Axis.isLog:
            if flag and self._keepDataAspectRatio:
                _logger.warning("KeepDataAspectRatio is ignored with log axes")
            self._plotFrame.yAxis.isLog = flag
            self._plotFrame.y2Axis.isLog = flag

    def setYAxisInverted(self, flag: bool):
        self._plotFrame.isYAxisInverted = flag

    def isYAxisInverted(self) -> bool:
        return self._plotFrame.isYAxisInverted

    def setXAxisInverted(self, flag: bool):
        self._plotFrame.isXAxisInverted = flag

    def isXAxisInverted(self) -> bool:
        return self._plotFrame.isXAxisInverted

    def isYRightAxisVisible(self):
        return self._plotFrame.isY2Axis

    def isKeepDataAspectRatio(self):
        if self._plotFrame.xAxis.isLog or self._plotFrame.yAxis.isLog:
            return False
        return self._keepDataAspectRatio

    def setKeepDataAspectRatio(self, flag):
        if flag and (self._plotFrame.xAxis.isLog or self._plotFrame.yAxis.isLog):
            _logger.warning("KeepDataAspectRatio is ignored with log axes")
        self._keepDataAspectRatio = flag

    def setGraphGrid(self, which):
        assert which in (None, "major", "both")
        self._plotFrame.grid = which is not None

    # Backend API: Data <-> Pixel ############################################

    def dataToPixel(self, x, y, axis):
        result = self._plotFrame.dataToPixel(x, y, axis)
        if result is None:
            return None
        dpr = self.getDevicePixelRatio()
        return tuple(value / dpr for value in result)

    def pixelToData(self, x, y, axis):
        dpr = self.getDevicePixelRatio()
        return self._plotFrame.pixelToData(x * dpr, y * dpr, axis)

    def getPlotBoundsInPixels(self):
        dpr = self.getDevicePixelRatio()
        return tuple(
            int(value / dpr)
            for value in self._plotFrame.plotOrigin + self._plotFrame.plotSize
        )

    # Backend API: Margins & Colors ##########################################

    def setAxesMargins(self, left: float, top: float, right: float, bottom: float):
        self._plotFrame.marginRatios = left, top, right, bottom

    def setForegroundColors(self, foregroundColor, gridColor):
        self._plotFrame.foregroundColor = foregroundColor
        self._plotFrame.gridColor = gridColor

    def setBackgroundColors(self, backgroundColor, dataBackgroundColor):
        self._backgroundColor = backgroundColor
        self._dataBackgroundColor = dataBackgroundColor

        # Remove old background
        if hasattr(self, "_bgObj") and self._bgObj is not None:
            if self._bgObj in self._scene.children:
                self._scene.remove(self._bgObj)

        # Update scene background
        if backgroundColor is not None:
            bgColor = gfx.Color(*backgroundColor)
            self._bgObj = gfx.Background(None, gfx.BackgroundMaterial(bgColor))
            # Insert at the beginning so it's behind everything
            self._scene.add(self._bgObj)
        else:
            self._bgObj = None
