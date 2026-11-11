"""Validate captured Relay1 attention and all standalone viewer assets."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3] / "Training_outputs/Prototype/relay1_visualization_step4000"


def main():
    expected = {"mapverse_qa273": (2, 36, 26, 37), "mapwise_qa294": (2, 36, 19, 31)}
    for folder, shape in expected.items():
        directory = ROOT / folder
        with np.load(directory / "attention.npz") as archive:
            maps = archive["relay_attention"]
        assert maps.shape == shape and np.isfinite(maps).all()
        assert not np.allclose(maps[0], maps[1])
        record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        assert record["map_axes"] == ["relay_token", "decoder_layer", "patch_y", "patch_x"]
        html = (directory / "relay1_attention.html").read_text(encoding="utf-8")
        references = set(re.findall(r"heatmaps/[^\" ]+\.png", html))
        assert len(references) == 78
        for reference in references:
            path = directory / reference
            assert path.is_file()
            with Image.open(path) as image:
                assert max(image.size) <= 1800
        print(folder, maps.shape, len(references))
    assert (ROOT / "index.html").is_file()
    print("PASS: Relay1 visualization artifacts")


if __name__ == "__main__":
    main()
