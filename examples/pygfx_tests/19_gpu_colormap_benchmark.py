"""GPU colormap performance benchmark for pygfx backend.

Streams 2D image data at maximum rate and measures FPS,
with timing breakdown: data generation vs plot rendering.

Usage:
    python 19_gpu_colormap_benchmark.py
    python 19_gpu_colormap_benchmark.py --size 2048 --duration 10
"""

import time
import numpy
import argparse
from silx.gui import qt
from silx.gui.plot.PlotWindow import PlotWindow
from silx.gui.colors import Colormap


class StreamingBenchmark(qt.QWidget):
    def __init__(self, image_size=1024, duration=5.0):
        super().__init__()
        self.setWindowTitle("pygfx GPU Colormap Benchmark")
        self._duration = duration
        self._image_size = image_size

        layout = qt.QVBoxLayout(self)

        # Controls
        ctrl = qt.QHBoxLayout()
        ctrl.addWidget(qt.QLabel("Size:"))
        self._size_combo = qt.QComboBox()
        self._size_combo.addItems(["256", "512", "1024", "2048", "4096"])
        self._size_combo.setCurrentText(str(image_size))
        ctrl.addWidget(self._size_combo)

        ctrl.addWidget(qt.QLabel("Norm:"))
        self._norm_combo = qt.QComboBox()
        self._norm_combo.addItems(["linear", "log", "sqrt", "gamma", "arcsinh"])
        ctrl.addWidget(self._norm_combo)

        ctrl.addStretch()
        self._status = qt.QLabel("Ready")
        self._status.setMinimumWidth(400)
        font = self._status.font()
        font.setPointSize(13)
        font.setBold(True)
        self._status.setFont(font)
        ctrl.addWidget(self._status)

        self._start_btn = qt.QPushButton("Start")
        self._stop_btn = qt.QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        ctrl.addWidget(self._start_btn)
        ctrl.addWidget(self._stop_btn)
        layout.addLayout(ctrl)

        # Plot with toolbar (includes colormap dialog button)
        self._plot = PlotWindow(backend="pygfx", colormap=True, mask=False,
                                roi=False, fit=False)
        self._plot.setGraphTitle("pygfx streaming")
        self._plot.setKeepDataAspectRatio(True)
        layout.addWidget(self._plot)

        # Results table
        self._results_text = qt.QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMaximumHeight(180)
        self._results_text.setFontFamily("monospace")
        self._results_text.setText(
            "Results will appear here after each run.\n"
            "Try different sizes and normalizations to compare.\n\n"
            "gen_ms  = data generation (numpy)\n"
            "plot_ms = addImage + render (GPU colormap pipeline)\n"
            "other   = Qt event loop overhead (dialog, UI updates, etc.)\n"
            "total   = gen + plot + other (should ≈ 1000/FPS)"
        )
        layout.addWidget(self._results_text)

        # State
        self._timer = qt.QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._frame_count = 0
        self._t_start = 0.0
        self._total_gen_ms = 0.0
        self._total_plot_ms = 0.0
        self._total_other_ms = 0.0
        self._last_frame_end = 0.0
        self._results = []

        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)

    def _start(self):
        size = int(self._size_combo.currentText())
        norm = self._norm_combo.currentText()

        # Set colormap
        if norm == "log":
            cm = Colormap("viridis", normalization="log", vmin=0.01, vmax=1.5)
        elif norm == "gamma":
            cm = Colormap("viridis", normalization="gamma", vmin=0.0, vmax=1.5)
            cm.setGammaNormalizationParameter(2.2)
        elif norm == "arcsinh":
            cm = Colormap("viridis", normalization="arcsinh", vmin=-0.5, vmax=1.5)
        else:
            cm = Colormap("viridis", normalization=norm, vmin=0.0, vmax=1.5)

        self._plot.setDefaultColormap(cm)
        self._image_size = size
        self._frame_count = 0
        self._t_start = time.perf_counter()
        self._total_gen_ms = 0.0
        self._total_plot_ms = 0.0
        self._total_other_ms = 0.0
        self._last_frame_end = time.perf_counter()
        self._last_fps_time = self._t_start

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._size_combo.setEnabled(False)
        self._norm_combo.setEnabled(False)
        self._status.setText(f"Running: {size}x{size} {norm}...")
        self._timer.start(0)

    def _stop(self):
        self._timer.stop()
        elapsed = time.perf_counter() - self._t_start
        n = max(self._frame_count, 1)
        avg_fps = n / elapsed if elapsed > 0 else 0
        avg_gen = self._total_gen_ms / n
        avg_plot = self._total_plot_ms / n
        avg_other = self._total_other_ms / n
        size = self._image_size
        norm = self._norm_combo.currentText()

        self._results.append((size, norm, avg_fps, avg_gen, avg_plot, avg_other, n, elapsed))

        # Update results table
        lines = [
            f"{'Size':>6} {'Norm':>8} {'FPS':>7} "
            f"{'gen_ms':>7} {'plot_ms':>8} {'other':>7} {'total':>7} "
            f"{'Frames':>7} {'Time':>5}"
        ]
        lines.append("-" * 74)
        for s, no, fps, gm, pm, om, fr, t in self._results:
            lines.append(
                f"{s:>6} {no:>8} {fps:>7.1f} "
                f"{gm:>7.2f} {pm:>8.2f} {om:>7.2f} {gm+pm+om:>7.2f} "
                f"{fr:>7} {t:>4.1f}s"
            )
        self._results_text.setText("\n".join(lines))

        self._status.setText(
            f"Done: {avg_fps:.1f} FPS | "
            f"gen {avg_gen:.1f} + plot {avg_plot:.1f} + other {avg_other:.1f}ms"
        )
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._size_combo.setEnabled(True)
        self._norm_combo.setEnabled(True)

    def _update_frame(self):
        size = self._image_size
        t = self._frame_count * 0.05

        # --- Qt/dialog overhead (time between frames) ---
        t_entry = time.perf_counter()
        if self._frame_count > 0:
            self._total_other_ms += (t_entry - self._last_frame_end) * 1000

        # --- Data generation (timed) ---
        t0 = time.perf_counter()
        x = numpy.linspace(-3, 3, size)
        xx, yy = numpy.meshgrid(x, x)
        cx, cy = numpy.sin(t) * 1.5, numpy.cos(t) * 1.5
        data = (
            numpy.exp(-((xx - cx) ** 2 + (yy - cy) ** 2))
            + 0.1 * numpy.random.random((size, size))
        ).astype(numpy.float32)
        t1 = time.perf_counter()

        # --- Plot update (timed) ---
        self._plot.addImage(data, replace=True)
        t2 = time.perf_counter()

        self._total_gen_ms += (t1 - t0) * 1000
        self._total_plot_ms += (t2 - t1) * 1000
        self._frame_count += 1
        self._last_frame_end = time.perf_counter()

        # Update status every 0.5s
        now = time.perf_counter()
        if now - self._last_fps_time >= 0.5:
            n = self._frame_count
            fps = n / (now - self._t_start)
            avg_gen = self._total_gen_ms / n
            avg_plot = self._total_plot_ms / n
            avg_other = self._total_other_ms / max(n - 1, 1)
            self._status.setText(
                f"{size}x{size} | FPS: {fps:.1f} | "
                f"gen {avg_gen:.1f} + plot {avg_plot:.1f} + other {avg_other:.1f}ms"
            )
            self._last_fps_time = now

        # Auto-stop after duration
        if now - self._t_start >= self._duration:
            self._stop()


def main():
    parser = argparse.ArgumentParser(description="pygfx GPU colormap benchmark")
    parser.add_argument("-s", "--size", type=int, default=1024,
                        help="Initial image size (default: 1024)")
    parser.add_argument("-d", "--duration", type=float, default=5.0,
                        help="Seconds per run (default: 5)")
    args = parser.parse_args()

    app = qt.QApplication([])
    w = StreamingBenchmark(image_size=args.size, duration=args.duration)
    w.resize(900, 700)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
