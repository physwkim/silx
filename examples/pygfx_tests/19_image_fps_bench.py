"""Image streaming FPS benchmark for pygfx backend.

Measures per-frame timing breakdown for continuous 2D image updates:
  gen_ms  - numpy image generation
  plot_ms - addImage call (CPU colormap + texture upload)
  other   - Qt event loop + GPU render/present
"""

import time
import argparse
import numpy as np
from silx.gui import qt
from silx.gui.plot import Plot2D


def _generate_image(size, rng):
    """Generate a test image with moving Gaussian peaks."""
    cx, cy = rng.random(2) * size
    sigma = size / 8
    y, x = np.ogrid[:size, :size]
    img = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    img += 0.05 * rng.random((size, size))
    return img.astype(np.float32)


class ImageBenchmark(qt.QWidget):
    def __init__(self, sizes=(1024, 2048, 4096), duration=5.0):
        super().__init__()
        self.setWindowTitle("pygfx Image FPS Benchmark")
        self._sizes = list(sizes)
        self._duration = duration
        self._rng = np.random.default_rng(42)

        layout = qt.QVBoxLayout(self)
        self._label = qt.QLabel("Starting...")
        self._label.setAlignment(qt.Qt.AlignCenter)
        font = self._label.font()
        font.setPointSize(14)
        self._label.setFont(font)
        layout.addWidget(self._label)

        self._plot = Plot2D(backend="pygfx")
        self._plot.setKeepDataAspectRatio(False)
        self._plot.getDefaultColormap().setName("viridis")
        layout.addWidget(self._plot)

        self._results = []
        self._queue = list(self._sizes)
        self._frame_gen = []
        self._frame_plot = []
        self._frame_total = []

        self._timer = qt.QTimer(self)
        self._timer.timeout.connect(self._tick)

        qt.QTimer.singleShot(300, self._next_size)

    def _next_size(self):
        if not self._queue:
            self._print_results()
            return

        self._cur_size = self._queue.pop(0)
        self._label.setText(
            f"Benchmarking {self._cur_size}x{self._cur_size}..."
        )
        self._frame_gen = []
        self._frame_plot = []
        self._frame_total = []
        self._bench_start = time.perf_counter()

        # Warm-up frame
        img = _generate_image(self._cur_size, self._rng)
        self._plot.addImage(img, resetzoom=True)
        qt.QApplication.processEvents()

        self._bench_start = time.perf_counter()
        self._timer.start(0)

    def _tick(self):
        t0 = time.perf_counter()

        # Generate
        img = _generate_image(self._cur_size, self._rng)
        t1 = time.perf_counter()

        # Plot
        self._plot.addImage(img, resetzoom=False)
        t2 = time.perf_counter()

        # Process events (forces Qt + GPU flush)
        qt.QApplication.processEvents()
        t3 = time.perf_counter()

        self._frame_gen.append(t1 - t0)
        self._frame_plot.append(t2 - t1)
        self._frame_total.append(t3 - t0)

        elapsed = t3 - self._bench_start
        fps = len(self._frame_gen) / elapsed if elapsed > 0 else 0
        self._label.setText(
            f"{self._cur_size}x{self._cur_size}  |  "
            f"{fps:.1f} FPS  |  {len(self._frame_gen)} frames"
        )

        if elapsed >= self._duration:
            self._timer.stop()
            self._record_result()
            qt.QTimer.singleShot(200, self._next_size)

    def _record_result(self):
        n = len(self._frame_gen)
        elapsed = sum(self._frame_total)
        gen = np.array(self._frame_gen) * 1000
        plot = np.array(self._frame_plot) * 1000
        total = np.array(self._frame_total) * 1000
        other = total - gen - plot

        self._results.append({
            "size": self._cur_size,
            "fps": n / elapsed if elapsed > 0 else 0,
            "gen_ms": float(np.mean(gen)),
            "plot_ms": float(np.mean(plot)),
            "other": float(np.mean(other)),
            "total": float(np.mean(total)),
            "frames": n,
            "time": elapsed,
        })

    def _print_results(self):
        self._label.setText("Done!")
        hdr = (
            f"{'Size':>6}  {'Norm':>8}  {'FPS':>7}  {'gen_ms':>7}  "
            f"{'plot_ms':>7}  {'other':>7}  {'total':>7}  "
            f"{'Frames':>6}  {'Time':>5}"
        )
        sep = "-" * len(hdr)
        print(f"\n{hdr}\n{sep}")
        for r in self._results:
            print(
                f"{r['size']:>6}  {'linear':>8}  {r['fps']:>7.1f}  "
                f"{r['gen_ms']:>7.2f}  {r['plot_ms']:>7.2f}  "
                f"{r['other']:>7.2f}  {r['total']:>7.2f}  "
                f"{r['frames']:>6}  {r['time']:>4.1f}s"
            )
        print()


def main():
    parser = argparse.ArgumentParser(description="Image streaming FPS benchmark")
    parser.add_argument(
        "-s", "--sizes", nargs="+", type=int, default=[1024, 2048, 4096],
        help="Image sizes to benchmark (default: 1024 2048 4096)",
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=5.0,
        help="Seconds per size (default: 5.0)",
    )
    args = parser.parse_args()

    app = qt.QApplication([])
    w = ImageBenchmark(sizes=args.sizes, duration=args.duration)
    w.resize(800, 700)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
