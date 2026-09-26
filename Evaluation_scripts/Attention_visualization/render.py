"""Standalone, offline HTML viewer and a PNG for the first prediction step."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def render_comparison(directory: Path) -> Path:
    records, arrays = [], []
    for variant in ("baseline", "adapted"):
        records.append(json.loads((directory / variant / "result.json").read_text(encoding="utf-8")))
        with np.load(directory / variant / "attention.npz") as archive:
            arrays.append(archive["attention"])
    for key in ("layers", "heads", "merged_grid_hw", "prompt", "image"):
        if records[0][key] != records[1][key]:
            raise ValueError(f"Cannot compare incompatible {key} values.")
    # The browser receives the head mean only. Full per-head float32 weights
    # remain in NPZ; embedding all heads makes a single HTML hundreds of MB.
    means = [a.mean(axis=2) for a in arrays]
    maximum = np.maximum(*(a.max(axis=(0, 2, 3)) for a in means))
    maximum = np.maximum(maximum, np.finfo(np.float32).tiny)
    for record, array in zip(records, means):
        record["display_f32"] = base64.b64encode(array.astype("<f4").tobytes()).decode("ascii")
        record["display_shape"] = list(array.shape)
    with Image.open(records[0]["image"]) as source:
        image = source.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = {"records": records, "vmax": maximum.tolist(),
               "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}
    # Escape '<' to prevent user question/response text from closing the script element.
    data = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    output = directory / "comparison.html"
    output.write_text(template.replace("__PAYLOAD__", data), encoding="utf-8")
    render_first_frame(directory, records, arrays, maximum, image)
    return output


def render_first_frame(directory, records, arrays, maximum, image):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    for ax, record, array in zip(axes, records, arrays):
        ax.imshow(image)
        heat = array[0, 0].mean(axis=0)
        ax.imshow(heat, extent=(-0.5, image.width - 0.5, image.height - 0.5, -0.5),
                  interpolation="nearest", cmap="inferno", vmin=0, vmax=maximum[0],
                  alpha=0.65 * heat / maximum[0])
        ax.set_title(f"{record['variant']} | predicting token 0: {record['tokens'][0]!r}")
        ax.axis("off")
    fig.suptitle(f"{records[0]['question']}\nLayer {records[0]['layers'][0]}, mean of {len(records[0]['heads'])} heads | shared scale")
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig.colorbar(ScalarMappable(norm=Normalize(0, maximum[0]), cmap="inferno"), ax=axes,
                 label="Raw attention probability per image token", shrink=0.6)
    fig.savefig(directory / "first_step.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(render_comparison(parser.parse_args().directory))
