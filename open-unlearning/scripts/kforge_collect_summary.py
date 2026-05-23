#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path


PATTERN = re.compile(
    r"KFORGE_TOFU_(?P<forget>forget\d+)_R(?P<rank>\d+)_M(?P<modules>[^_]+)"
    r"_B(?P<batches>\d+)_S(?P<strength>[^_]+)_(?P<mode>[^_]+)"
    r"_(?P<retain>retain|forgetonly)(?P<suffix>.*?)_EVAL_FP32"
)


def main() -> None:
    eval_root = Path("saves/eval")
    rows = []
    for summary_path in sorted(eval_root.glob("KFORGE_TOFU_*_EVAL_FP32/TOFU_SUMMARY.json")):
        match = PATTERN.fullmatch(summary_path.parent.name)
        if not match:
            continue
        data = json.loads(summary_path.read_text())
        row = match.groupdict()
        row["strength_float"] = float(row["strength"].replace("p", "."))
        for key in [
            "forget_quality",
            "model_utility",
            "forget_Q_A_Prob",
            "forget_Q_A_ROUGE",
            "extraction_strength",
            "privleak",
        ]:
            row[key] = data.get(key)
        row["path"] = str(summary_path)
        rows.append(row)

    out = eval_root / "kforge_all_summary.csv"
    fieldnames = [
        "forget",
        "rank",
        "modules",
        "batches",
        "strength",
        "strength_float",
        "mode",
        "retain",
        "suffix",
        "forget_quality",
        "model_utility",
        "forget_Q_A_Prob",
        "forget_Q_A_ROUGE",
        "extraction_strength",
        "privleak",
        "path",
    ]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
