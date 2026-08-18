"""
interactive.py
---------------
Live drawing demo. A 16x16 grid you paint on with the mouse stands in for
the LED grid: it starts all OFF, left-click turns a pixel ON, right-click
turns it back OFF. That's the whole state -- there is no "unknown": an
un-drawn cell is simply an unlit LED, exactly what the sketch-completion
model (model_sketch.pt) was trained on. You light up part of a shape and
the model lights up the rest.

A short idle timer (default 150ms after your last edit) fires a full
pipeline.py inference through the real FPGA and lights up the predicted
completion in the second grid. The debounce exists so a drag across many
cells doesn't flood the board with UART traffic -- only the settled
drawing gets sent.

The FPGA round trip runs on a background thread so the window never
freezes while waiting on serial I/O; results come back through a queue
and get drawn on the next GUI tick.

Requires weights already loaded onto the board (see load_weights.py).
Pass --mock to skip the board entirely and preview the GUI: predictions
come from the real trained model running on the CPU instead of a UART
round trip, so it looks/behaves the same as the real thing minus the
hardware.

Usage:
    python interactive.py --port COM6 [--model ../model/model_sketch.pt] [--debounce-ms 150]
    python interactive.py --mock   (no board required, for previewing the GUI)
"""

import argparse
import os
import queue
import sys
import threading
import tkinter as tk

import numpy as np
import serial
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "driver"))

from pipeline import PixelCompletionPipeline, load_model  # noqa: E402
from python_driver import FPGADriver  # noqa: E402


class _MockAttentionDriver:
    """
    Stands in for FPGADriver when --mock is passed. run_attention_layer()
    is the only method PixelCompletionPipeline calls on its driver, so
    implementing just that is enough to make the rest of the pipeline
    (embed, per-layer PC cleanup, output projection) run unmodified.

    Computes the SAME raw attention math (context @ W_out) the FPGA is
    supposed to compute, using the real trained submodule directly --
    so mock predictions match what the board should eventually produce,
    they just skip the UART round trip.
    """

    def __init__(self, model):
        self.model = model

    def run_attention_layer(self, layer_num, x_np):
        block = self.model.blocks[layer_num - 1]
        with torch.no_grad():
            x = torch.from_numpy(x_np).unsqueeze(0)
            out = block.attn(x)
        return out.squeeze(0).numpy()

GRID_SIZE = 16
CELL_PX = 28

COLOR_BG = "#1e1e1e"
COLOR_OFF = "#15152a"    # unlit LED / un-drawn cell
COLOR_ON = "#ffd23f"    # lit LED

# The prediction grid is a stand-in for the physical LED board: every cell is
# either lit or dark, nothing in between. We threshold the model's per-pixel
# probability at 0.5 rather than shading by confidence -- a gradient has no
# hardware meaning (an LED can't be 40% on) and just makes the output look
# noisy.
PRED_THRESHOLD = 0.5


class InteractiveApp:
    def __init__(self, root: tk.Tk, pipeline: PixelCompletionPipeline, debounce_ms: int):
        self.root = root
        self.pipeline = pipeline
        self.debounce_ms = debounce_ms

        # Board starts all off. Binary state: 1.0 = lit, 0.0 = unlit.
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        self.result_queue: "queue.Queue" = queue.Queue()
        self.inference_thread = None
        self.debounce_after_id = None

        self._build_ui()
        self.root.after(50, self._poll_result_queue)

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("TPU Pixel Completer")
        self.root.configure(bg=COLOR_BG)

        header = tk.Frame(self.root, bg=COLOR_BG)
        header.pack(padx=12, pady=(12, 0), fill="x")
        tk.Label(header, text="Your drawing", fg="white", bg=COLOR_BG,
                  font=("Segoe UI", 10, "bold")).pack(side="left", expand=True)
        tk.Label(header, text="FPGA prediction", fg="white", bg=COLOR_BG,
                  font=("Segoe UI", 10, "bold")).pack(side="right", expand=True)

        canvases = tk.Frame(self.root, bg=COLOR_BG)
        canvases.pack(padx=12, pady=6)

        size_px = GRID_SIZE * CELL_PX
        self.draw_canvas = tk.Canvas(canvases, width=size_px, height=size_px,
                                      bg=COLOR_OFF, highlightthickness=0)
        self.draw_canvas.grid(row=0, column=0, padx=(0, 12))

        self.pred_canvas = tk.Canvas(canvases, width=size_px, height=size_px,
                                      bg=COLOR_OFF, highlightthickness=0)
        self.pred_canvas.grid(row=0, column=1)

        # Pre-create one rectangle per cell in each canvas so redraws just
        # recolor them -- no create/destroy churn while drawing.
        self.draw_cells = self._make_cell_rects(self.draw_canvas)
        self.pred_cells = self._make_cell_rects(self.pred_canvas)

        self.draw_canvas.bind("<Button-1>", lambda e: self._paint(e, 1.0))
        self.draw_canvas.bind("<B1-Motion>", lambda e: self._paint(e, 1.0))
        self.draw_canvas.bind("<Button-3>", lambda e: self._paint(e, 0.0))
        self.draw_canvas.bind("<B3-Motion>", lambda e: self._paint(e, 0.0))

        footer = tk.Frame(self.root, bg=COLOR_BG)
        footer.pack(padx=12, pady=(0, 12), fill="x")

        tk.Button(footer, text="Clear", command=self._clear).pack(side="left")

        self.status_var = tk.StringVar(value="Idle")
        tk.Label(footer, textvariable=self.status_var, fg="#aaaaaa", bg=COLOR_BG).pack(side="right")

        tk.Label(self.root, text="Left-click: turn LED on   Right-click: turn LED off   (board starts all off)",
                  fg="#777777", bg=COLOR_BG, font=("Segoe UI", 8)).pack(pady=(0, 10))

        self._render_draw_grid()

    def _make_cell_rects(self, canvas: tk.Canvas):
        rects = {}
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x0, y0 = col * CELL_PX, row * CELL_PX
                x1, y1 = x0 + CELL_PX, y0 + CELL_PX
                rects[(row, col)] = canvas.create_rectangle(
                    x0, y0, x1, y1, fill=COLOR_OFF, outline=COLOR_BG
                )
        return rects

    # ── Drawing ──────────────────────────────────────────────────────

    def _paint(self, event, value: float):
        col = event.x // CELL_PX
        row = event.y // CELL_PX
        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            return
        if self.grid[row, col] == value:
            return
        self.grid[row, col] = value
        self._render_draw_grid()
        self._schedule_inference()

    def _clear(self):
        self.grid[:, :] = 0.0
        self._render_draw_grid()
        self._schedule_inference()

    def _render_draw_grid(self):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                color = COLOR_ON if self.grid[row, col] == 1.0 else COLOR_OFF
                self.draw_canvas.itemconfig(self.draw_cells[(row, col)], fill=color)

    def _render_prediction(self, prediction: np.ndarray):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                lit = prediction[row, col] >= PRED_THRESHOLD
                color = COLOR_ON if lit else COLOR_OFF
                self.pred_canvas.itemconfig(self.pred_cells[(row, col)], fill=color)

    # ── Debounced FPGA inference ────────────────────────────────────

    def _schedule_inference(self):
        if self.debounce_after_id is not None:
            self.root.after_cancel(self.debounce_after_id)
        self.debounce_after_id = self.root.after(self.debounce_ms, self._fire_inference)

    def _fire_inference(self):
        self.debounce_after_id = None

        if self.inference_thread is not None and self.inference_thread.is_alive():
            # A round trip is still in flight -- try again once it's done
            # instead of overlapping requests on the same serial port.
            self._schedule_inference()
            return

        snapshot = self.grid.copy()
        self.status_var.set("Running inference...")
        self.inference_thread = threading.Thread(
            target=self._run_inference_worker, args=(snapshot,), daemon=True
        )
        self.inference_thread.start()

    def _run_inference_worker(self, snapshot: np.ndarray):
        prediction = self.pipeline.complete(snapshot)
        self.result_queue.put(prediction)

    def _poll_result_queue(self):
        try:
            prediction = self.result_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            if prediction is None:
                self.status_var.set("Inference failed -- check FPGA connection")
            else:
                self._render_prediction(prediction)
                self.status_var.set("Idle")
        self.root.after(50, self._poll_result_queue)


def main():
    parser = argparse.ArgumentParser(description="Interactive live pixel-completion demo")
    parser.add_argument("--port", default=None, help="Serial port, e.g. COM6 -- find yours with: python -m serial.tools.list_ports -v (required unless --mock)")
    parser.add_argument("--baud", type=int, default=None, help="Override baud rate (default 4000000)")
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "..", "model", "model_sketch.pt"),
        help="Path to trained checkpoint (default: ../model/model_sketch.pt)",
    )
    parser.add_argument("--debounce-ms", type=int, default=150,
                         help="Idle time after your last edit before an inference runs (default 500)")
    parser.add_argument("--mock", action="store_true",
                         help="Skip the FPGA -- run predictions on the CPU to preview the GUI without a board")
    args = parser.parse_args()

    if not args.mock and args.port is None:
        parser.error("--port is required unless --mock is passed")

    model = load_model(args.model)

    driver = None
    if args.mock:
        print("[interactive] --mock: running predictions on the CPU, no board required")
        pipeline = PixelCompletionPipeline(model, _MockAttentionDriver(model))
    else:
        driver_kwargs = {"port": args.port}
        if args.baud is not None:
            driver_kwargs["baud_rate"] = args.baud

        driver = FPGADriver(**driver_kwargs)
        try:
            driver.connect()
        except serial.SerialException as e:
            print(f"[interactive] Could not open {args.port}: {e}")
            sys.exit(1)

        pipeline = PixelCompletionPipeline(model, driver)

    root = tk.Tk()
    app = InteractiveApp(root, pipeline, args.debounce_ms)  # noqa: F841
    try:
        root.mainloop()
    finally:
        if driver is not None:
            driver.disconnect()


if __name__ == "__main__":
    main()
