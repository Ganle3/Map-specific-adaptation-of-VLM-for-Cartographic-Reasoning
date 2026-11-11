"""Redraw cluster montages from an existing dataset analysis."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from analyze_test_maps import enrichment_chart, montage


ROOT = Path(__file__).resolve().parents[3] / "Training_outputs/Prototype/relay1_test_map_analysis_step4000"


def main():
    with (ROOT / "patch_assignments.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["cluster"] = int(row["cluster"]); row["relay"] = int(row["relay"])
        row["score"] = float(row["score"]); row["cluster_cosine"] = float(row["cluster_cosine"])
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    enrichment_chart(summary["clusters"], ROOT / "relay_cluster_enrichment.png")
    for stat in summary["clusters"]:
        cluster_rows = [row for row in rows if row["cluster"] == stat["cluster"]]
        cluster_rows.sort(key=lambda row: row["cluster_cosine"], reverse=True)
        montage(cluster_rows, ROOT / "clusters" / f"cluster_{stat['cluster']:02d}.jpg",
                f"Cluster {stat['cluster']}: red=R1, blue=R2")
    report = ROOT / "report.html"
    html = report.read_text(encoding="utf-8")
    if 'relay_cluster_enrichment.png' not in html:
        html = html.replace('<div class="bias">',
                            '<img src="relay_cluster_enrichment.png"><div class="bias">', 1)
        report.write_text(html, encoding="utf-8")
    print(ROOT / "report.html")


if __name__ == "__main__":
    main()
