#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    protocol = [
        "# Shadow Pilot Protocol",
        "",
        "1. Shadow-mode risk routing (no decision override).",
        "2. Reviewer packet A/B: DINO-only vs DINO+VLM packet.",
        "3. Safety gate logging with human confirmation.",
        "",
        "Primary KPIs: reviewer_yield, p50/p90 decision time, false-alarm prevented, human override rate.",
    ]
    (out / "shadow_pilot_protocol.md").write_text("\n".join(protocol), encoding="utf-8")

    pd.DataFrame(
        [
            {"metric": "reviewer_yield", "type": "ratio"},
            {"metric": "decision_time_p50", "type": "time_sec"},
            {"metric": "decision_time_p90", "type": "time_sec"},
            {"metric": "override_rate", "type": "ratio"},
            {"metric": "false_alarm_prevented_rate", "type": "ratio"},
            {"metric": "audit_disagreement_rate", "type": "ratio"},
        ]
    ).to_csv(out / "field_metrics_schema.csv", index=False)

    pd.DataFrame(
        [
            {"record_id": "", "arm": "dino_only|dino_vlm", "decision": "", "time_sec": "", "usefulness_1_5": "", "notes": ""}
        ]
    ).to_csv(out / "reviewer_audit_form.csv", index=False)

    (out / "sample_size_notes.md").write_text(
        "\n".join(
            [
                "# Sample Size Notes",
                "",
                "- risk routing: >=1000 items/arm",
                "- false alarm checker: >=300 candidates",
                "- bad crop gate: >=500 bad + >=1000 good",
                "- reviewer packet: >=300 reviewed cards",
            ]
        ),
        encoding="utf-8",
    )

    (out / "E08_report.md").write_text("# E08 Shadow Field Pilot Design\n\nProtocol and KPI schema prepared.", encoding="utf-8")


if __name__ == "__main__":
    main()
