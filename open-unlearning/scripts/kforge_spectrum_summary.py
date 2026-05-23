#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path


PATTERN = re.compile(
    r"kforge_(?P<model>.+?)_(?P<forget>forget\d+)_B(?P<batches>\d+)_"
    r"layer(?P<layer>\d+)_(?P<module>.+?)_(?P<timestamp>\d{8}T\d{6}Z)\.json"
)


def main() -> None:
    rows = []
    for path in sorted(Path("saves/spectrum").glob("kforge_*.json")):
        match = PATTERN.fullmatch(path.name)
        if not match:
            continue
        payload = json.loads(path.read_text())
        if not payload:
            continue
        item = payload[0]
        top = item.get("top_singular_values", [])
        q = item.get("quantiles", {})
        rows.append(
            {
                **match.groupdict(),
                "module": match.group("module"),
                "top1": top[0] if top else "",
                "top2": top[1] if len(top) > 1 else "",
                "top4_sum": sum(top[:4]) if top else "",
                "top8_sum": sum(top[:8]) if top else "",
                "q50": q.get("q50", ""),
                "q90": q.get("q90", ""),
                "q95": q.get("q95", ""),
                "q99": q.get("q99", ""),
                "q100": q.get("q100", ""),
                "path": str(path),
            }
        )

    out = Path("saves/spectrum/kforge_spectrum_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "forget",
        "batches",
        "layer",
        "module",
        "timestamp",
        "top1",
        "top2",
        "top4_sum",
        "top8_sum",
        "q50",
        "q90",
        "q95",
        "q99",
        "q100",
        "path",
    ]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(
        rows,
        key=lambda r: float(r["top1"]) if r["top1"] != "" else float("-inf"),
        reverse=True,
    )
    top_out = Path("saves/spectrum/kforge_spectrum_top20.csv")
    with top_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked[:20])

    print(f"Wrote {len(rows)} rows to {out}")
    print(f"Wrote top 20 rows to {top_out}")
    for row in ranked[:10]:
        print(
            f"layer={row['layer']} module={row['module']} "
            f"top1={float(row['top1']):.6g} q99={float(row['q99']):.6g}"
        )


if __name__ == "__main__":
    main()
