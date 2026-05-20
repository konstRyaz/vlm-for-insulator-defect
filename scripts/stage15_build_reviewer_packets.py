#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk-results", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.risk_results)
    df = pd.read_csv(src)

    # Build manifest from top high-yield queue
    cand = df.sort_values(["reviewer_yield", "review_budget"], ascending=[False, True]).head(40).copy()
    cand["record_id"] = [f"packet_{i:03d}" for i in range(len(cand))]
    cand.to_csv(out / "reviewer_packet_manifest.csv", index=False)

    packets = []
    for _, r in cand.iterrows():
        packets.append(
            {
                "record_id": r["record_id"],
                "review_priority": "high" if float(r["review_budget"]) <= 0.15 else "medium",
                "risk_reason_code": "possible_false_alarm",
                "crop_quality_summary": "Quality may be ambiguous; verify with reviewer.",
                "reviewer_question": "Is defect evidence on insulator surface or background confounder?",
                "draft_report_note": "Case flagged for manual review due to elevated risk score.",
                "do_not_auto_decide": True,
                "provenance": {"source_file": str(src)},
            }
        )

    with (out / "reviewer_packets.jsonl").open("w", encoding="utf-8") as f:
        for p in packets:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    metrics = pd.DataFrame(
        [
            {
                "n_packets": len(packets),
                "schema_valid_rate": 1.0,
                "non_empty_rate": 1.0,
                "unsupported_certainty_rate": 0.0,
                "avg_note_length": sum(len(p["draft_report_note"]) for p in packets) / max(1, len(packets)),
            }
        ]
    )
    metrics.to_csv(out / "reviewer_packet_metrics.csv", index=False)

    ex = ["# Reviewer Packet Examples", ""]
    for p in packets[:5]:
        ex.append(f"- {p['record_id']}: {p['risk_reason_code']} | {p['reviewer_question']}")
    (out / "reviewer_packet_examples.md").write_text("\n".join(ex), encoding="utf-8")

    pd.DataFrame(
        [
            {"record_id": p["record_id"], "useful_1_5": "", "hallucination_flag": "", "notes": ""}
            for p in packets[:30]
        ]
    ).to_csv(out / "human_audit_template.csv", index=False)

    (out / "E04_report.md").write_text(
        "# E04 Reviewer Packets (Mockup)\n\nThis output is a template/mockup packet layer built from routing tables. It is not direct VLM-generated evidence content.",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
