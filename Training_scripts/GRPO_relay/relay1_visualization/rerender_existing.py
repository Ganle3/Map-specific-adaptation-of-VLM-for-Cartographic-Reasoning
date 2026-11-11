"""Rebuild HTML/PNG viewers from already captured Relay1 attention arrays."""
from pathlib import Path

from render import render_question


ROOT = Path(__file__).resolve().parents[3] / "Training_outputs/Prototype/relay1_visualization_step4000"


def main():
    pages = [render_question(path.parent) for path in sorted(ROOT.glob("*/result.json"))]
    links = "\n".join(
        f'<li><a href="{page.relative_to(ROOT).as_posix()}">{page.parent.name}</a></li>'
        for page in pages
    )
    (ROOT / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Relay1 attention</title>'
        '<style>body{font:18px system-ui;max-width:1000px;margin:40px auto;line-height:1.6}'
        'a{color:#0759b8}</style><h1>Relay1 step-4000 prefill attention</h1><ul>'
        + links + '</ul>', encoding="utf-8"
    )
    for page in pages:
        print(page)


if __name__ == "__main__":
    main()
