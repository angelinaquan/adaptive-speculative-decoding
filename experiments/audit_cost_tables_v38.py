"""Independently reconstruct saved v38 target tables from profiling samples.

This is a post-run implementation audit, not a change to the frozen statistical
analysis. Only the captured graph range can be checked from these samples.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(root):
    checked = []
    for role in ("original", "measured"):
        for round_index in range(4):
            path = root / f"cost_v38_{role}-round{round_index}.json"
            result = json.loads(path.read_text())
            assert result["status"] == "completed", path
            costs = result["costs"]
            assert costs["mode"] == role, path
            samples = costs["raw_target_samples"]
            sizes = sorted({int(n) for n, _ in samples})
            medians = np.array([
                np.median([value for n, value in samples if n == size])
                for size in sizes
            ])
            assert np.isfinite(medians).all() and (medians > 0).all()
            if role == "original":
                medians = np.maximum.accumulate(medians)
            graph_limit = max(result["metadata"]["engine"]["compilation_config"]["cudagraph_capture_sizes"])
            positions = np.arange(graph_limit + 1)
            expected = medians[np.minimum(np.searchsorted(sizes, positions), len(sizes) - 1)]
            actual = np.asarray(costs["cost_tables"][1])[:graph_limit + 1]
            np.testing.assert_array_equal(actual, expected, err_msg=str(path))
            checked.append({"session": path.stem, "checked_positions": len(positions),
                            "profile_sizes": sizes, "sample_count": len(samples)})
    return {"all_saved_target_tables_match_raw_profiles": True, "sessions": checked,
            "scope": "Target table entries at or below graph limit; no independent draft-table or above-limit reconstruction."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.root)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
