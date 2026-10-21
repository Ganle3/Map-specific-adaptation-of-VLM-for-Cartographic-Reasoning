"""Small Tk canvas used to draw original-image ROI boxes during live inference."""
from __future__ import annotations

from PIL import Image, ImageTk


def draw_rois(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Return one or more original-pixel rectangles after the user confirms.

    Drag with the left mouse button to add a rectangle.  The displayed image
    may be downscaled, but returned coordinates are always in the original
    image frame.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        raise RuntimeError("The canvas ROI picker requires the standard Windows Tk installation.") from exc

    root = tk.Tk()
    root.title("Live ROI-only intervention — draw visual evidence")
    root.update_idletasks()
    max_width = max(640, root.winfo_screenwidth() - 120)
    max_height = max(480, root.winfo_screenheight() - 220)
    scale = min(1.0, max_width / image.width, max_height / image.height)
    display_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    shown = image.resize(display_size, Image.Resampling.LANCZOS) if scale < 1 else image.copy()
    photo = ImageTk.PhotoImage(shown)
    canvas = tk.Canvas(root, width=shown.width, height=shown.height, cursor="crosshair", highlightthickness=0)
    canvas.pack(padx=10, pady=(10, 4))
    canvas.create_image(0, 0, image=photo, anchor="nw")
    controls = tk.Frame(root); controls.pack(fill="x", padx=10, pady=(0, 10))
    message = tk.Label(controls, text="Drag to draw ROI(s).  Undo removes the latest box.  Click Use ROIs to continue live inference.")
    message.pack(side="left")
    boxes: list[tuple[int, int, int, int]] = []
    canvas_ids: list[int] = []
    start: tuple[int, int] | None = None
    preview: int | None = None
    result: list[tuple[int, int, int, int]] | None = None

    def clamp(event) -> tuple[int, int]:
        return min(max(event.x, 0), shown.width), min(max(event.y, 0), shown.height)

    def press(event) -> None:
        nonlocal start, preview
        start = clamp(event)
        preview = canvas.create_rectangle(*start, *start, outline="#f59e0b", width=3, dash=(4, 2))

    def move(event) -> None:
        if start is not None and preview is not None:
            end = clamp(event)
            canvas.coords(preview, start[0], start[1], end[0], end[1])

    def release(event) -> None:
        nonlocal start, preview
        if start is None:
            return
        end = clamp(event)
        if preview is not None:
            canvas.delete(preview)
        left, right = sorted((start[0], end[0])); top, bottom = sorted((start[1], end[1]))
        start, preview = None, None
        if right - left < 3 or bottom - top < 3:
            return
        box = (round(left / scale), round(top / scale), round(right / scale), round(bottom / scale))
        # Ensure displayed edge rounding never exceeds original bounds.
        box = (max(0, min(image.width - 1, box[0])), max(0, min(image.height - 1, box[1])),
               max(1, min(image.width, box[2])), max(1, min(image.height, box[3])))
        if box[0] >= box[2] or box[1] >= box[3]:
            return
        boxes.append(box)
        index = len(boxes)
        canvas_ids.extend([
            canvas.create_rectangle(left, top, right, bottom, outline="#f59e0b", width=3),
            canvas.create_text(left + 6, top + 10, text=str(index), fill="black", anchor="w",
                               font=("Segoe UI", 12, "bold")),
        ])
        message.config(text=f"{len(boxes)} ROI(s): " + "; ".join(str(value) for value in boxes))

    def undo() -> None:
        if not boxes:
            return
        boxes.pop()
        for _ in range(2):
            canvas.delete(canvas_ids.pop())
        message.config(text=f"{len(boxes)} ROI(s) retained.")

    def clear() -> None:
        boxes.clear()
        while canvas_ids:
            canvas.delete(canvas_ids.pop())
        message.config(text="Drag to draw ROI(s).")

    def accept() -> None:
        nonlocal result
        if not boxes:
            messagebox.showwarning("No ROI", "Draw at least one ROI before continuing.")
            return
        result = list(boxes)
        root.destroy()

    def cancel() -> None:
        root.destroy()

    tk.Button(controls, text="Undo", command=undo).pack(side="right", padx=(5, 0))
    tk.Button(controls, text="Clear", command=clear).pack(side="right", padx=(5, 0))
    tk.Button(controls, text="Use ROIs and continue", command=accept).pack(side="right", padx=(5, 0))
    root.protocol("WM_DELETE_WINDOW", cancel)
    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", move)
    canvas.bind("<ButtonRelease-1>", release)
    root.mainloop()
    if result is None:
        raise RuntimeError("ROI drawing was cancelled; live inference cache was not continued.")
    return result
