#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable


GROUPS = {
    "intact": {"intact_structure", "regular_disc_shape", "no_visible_break"},
    "broken_structure": {"missing_fragment", "edge_discontinuity"},
    "flashover_surface": {"burn_like_mark", "dark_surface_trace", "surface_damage_mark", "surface_stain"},
    "quality_or_confounder": {
        "blurred_region",
        "partial_view",
        "occluded_region",
        "low_contrast",
        "ambiguous_evidence",
        "unclear_boundary",
    },
}


def to_group_set(tags: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for tag in tags:
        for g, members in GROUPS.items():
            if tag in members:
                out.add(g)
    return out
