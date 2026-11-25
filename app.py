"""Application entry point with a startup splash screen."""

from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parent
SPLASH_IMAGE = PROJECT_ROOT / "assets" / "cctv_startup.png"
IMAGE_SIZE = (1553, 768)
BAR_REGION = (565, 690, 988, 725)
BAR_CENTER_Y = 707


def _remove_embedded_bar(image: Image.Image) -> Image.Image:
    """Clear the static progress bar baked into the supplied artwork."""
    image = image.convert("RGB")
    pixels = image.load()
    width, height = image.size
    x0 = round(BAR_REGION[0] * width / IMAGE_SIZE[0])
    x1 = round(BAR_REGION[2] * width / IMAGE_SIZE[0])
    y0 = round(BAR_REGION[1] * height / IMAGE_SIZE[1])
    y1 = round(BAR_REGION[3] * height / IMAGE_SIZE[1])
    top = max(0, y0 - 1)
    bottom = min(height - 1, y1 + 1)

    for x in range(x0, x1 + 1):
        top_color = pixels[x, top]
        bottom_color = pixels[x, bottom]
        for y in range(y0, y1 + 1):
            ratio = (y - top + 1) / (bottom - top + 1)
            pixels[x, y] = tuple(
                round(top_color[channel] * (1.0 - ratio) + bottom_color[channel] * ratio)
                for channel in range(3)
            )
    return image


def _bar_geometry(
    source_size: tuple[int, int], window_size: tuple[int, int]
) -> tuple[int, int, int]:
    source_width, source_height = source_size
    window_width, window_height = window_size
    scale = max(window_width / source_width, window_height / source_height)
    offset_x = (source_width * scale - window_width) / 2.0
    offset_y = (source_height * scale - window_height) / 2.0
    left = round(BAR_REGION[0] * scale - offset_x)
    right = round(BAR_REGION[2] * scale - offset_x)
    y = round(BAR_CENTER_Y * scale - offset_y)
    return left, right, y


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    try:
        screen_size = (root.winfo_screenwidth(), root.winfo_screenheight())
        with Image.open(SPLASH_IMAGE) as source:
            source_size = source.size
            background = _remove_embedded_bar(source)
        fitted_background = ImageOps.fit(
            background,
            screen_size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        background_photo = ImageTk.PhotoImage(fitted_background)
    except Exception as error:
        messagebox.showerror(
            "Startup Error",
            f"Could not load the CCTV startup screen:\n\n{type(error).__name__}: {error}",
            parent=root,
        )
        root.destroy()
        return

    splash = tk.Toplevel(root)
    splash.overrideredirect(True)
    splash.attributes("-topmost", True)
    splash.configure(bg="#061321")

    canvas = tk.Canvas(
        splash,
        width=screen_size[0],
        height=screen_size[1],
        bg="#061321",
        borderwidth=0,
        highlightthickness=0,
    )
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=background_photo, anchor="nw")
    canvas.image = background_photo

    bar_left, bar_right, bar_y = _bar_geometry(source_size, screen_size)
    track_width = max(2, round(screen_size[1] / source_size[1] * 4))
    line_width = max(2, round(screen_size[1] / source_size[1] * 3))
    canvas.create_line(
        bar_left,
        bar_y,
        bar_right,
        bar_y,
        fill="#17344D",
        width=track_width,
        capstyle=tk.ROUND,
    )
    travel = max(1, bar_right - bar_left)
    segment_length = min(travel, max(50, round(travel * 0.30)))
    progress_line = canvas.create_line(
        bar_left,
        bar_y,
        bar_left + segment_length,
        bar_y,
        fill="#159BFF",
        width=line_width,
        capstyle=tk.ROUND,
    )

    phase = 0.0

    def animate_loading_line() -> None:
        nonlocal phase
        if not splash.winfo_exists():
            return
        phase = (phase + 0.012) % 1.0
        eased_position = (1.0 - math.cos(2.0 * math.pi * phase)) / 2.0
        start = bar_left + round((travel - segment_length) * eased_position)
        canvas.coords(progress_line, start, bar_y, start + segment_length, bar_y)
        splash.after(16, animate_loading_line)

    startup_result: queue.Queue[Callable[[tk.Tk], object] | Exception] = queue.Queue(
        maxsize=1
    )

    def initialize_application() -> None:
        try:
            from cctv_app import SmartCCTVApp
        except Exception as error:
            startup_result.put(error)
        else:
            startup_result.put(SmartCCTVApp)

    dashboard_app = None

    def show_startup_error(error: Exception) -> None:
        try:
            messagebox.showerror(
                "Startup Error",
                f"Smart CCTV could not finish initializing:\n\n"
                f"{type(error).__name__}: {error}",
                parent=splash,
            )
        finally:
            root.destroy()

    def finish_initialization() -> None:
        nonlocal dashboard_app
        try:
            dashboard_factory = startup_result.get_nowait()
        except queue.Empty:
            if root.winfo_exists():
                root.after(25, finish_initialization)
            return

        if isinstance(dashboard_factory, Exception):
            show_startup_error(dashboard_factory)
            return

        try:
            dashboard_app = dashboard_factory(root)
            root.update_idletasks()
        except Exception as error:
            show_startup_error(error)
            return

        splash.destroy()
        root.deiconify()
        root.lift()

    splash.geometry(f"{screen_size[0]}x{screen_size[1]}+0+0")
    splash.deiconify()
    splash.lift()
    splash.update_idletasks()
    animate_loading_line()
    root.after(25, finish_initialization)

    try:
        threading.Thread(
            target=initialize_application,
            name="cctv-startup-initializer",
            daemon=True,
        ).start()
    except Exception as error:
        show_startup_error(error)
        return

    root.mainloop()


if __name__ == "__main__":
    main()
