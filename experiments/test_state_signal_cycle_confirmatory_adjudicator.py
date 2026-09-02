import csv
import tempfile
import unittest
from pathlib import Path

import state_signal_cycle_confirmatory_adjudicator as adj


FIELDS = [
    "family", "seed", "source_idx", "target_idx", "mode", "effect",
    "tv_A_source", "tv_A_target", "entropy_gap_source", "entropy_gap_target",
    "common_A_range", "b_head_change", "max_delta_norm_relerr", "degenerate_fraction",
]


def make_rows(cycle_by_family=None):
    cycle_by_family = cycle_by_family or {f: 0.5 for f in range(5)}
    rows = []
    for family in range(5):
        for source in range(6):
            for target in range(6):
                for mode in adj.MODES:
                    edge = (source == 0 and target > 0) or (target == 0 and source > 0)
                    if mode == "Bsub_aligned":
                        effect = 2.0 if edge else 0.0
                    else:
                        effect = cycle_by_family[family] if edge else 0.0
                    rows.append({
                        "family": str(family),
                        "seed": str(adj.DEFAULT_SEED),
                        "source_idx": str(source),
                        "target_idx": str(target),
                        "mode": mode,
                        "effect": str(effect),
                        "tv_A_source": "0.5",
                        "tv_A_target": "0.5",
                        "entropy_gap_source": "0",
                        "entropy_gap_target": "0",
                        "common_A_range": "0.001",
                        "b_head_change": "0",
                        "max_delta_norm_relerr": "1e-7",
                        "degenerate_fraction": "0",
                        "_source": "synthetic",
                    })
    return rows


class AdjudicatorTests(unittest.TestCase):
    def test_locked_pass(self):
        result = adj.adjudicate_rows(make_rows())
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["n_D_negative"], 5)
        self.assertAlmostEqual(result["pooled_ratio"], 0.25)
        self.assertEqual(result["n_family_ratio_le_0_60"], 5)
        self.assertEqual(result["n_M_aligned_ge_1"], 5)

    def test_complete_scientific_fail(self):
        rows = make_rows({0: 3.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5})
        result = adj.adjudicate_rows(rows)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["criteria"]["1_D_negative_5of5"])

    def test_incomplete_is_uncertain(self):
        rows = make_rows()[:-1]
        result = adj.adjudicate_rows(rows)
        self.assertEqual(result["verdict"], "UNCERTAIN")
        self.assertFalse(result["complete"])

    def test_audit_failure_is_fail(self):
        rows = make_rows()
        rows[0]["common_A_range"] = "0.02"
        result = adj.adjudicate_rows(rows)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["criteria"]["5_audits_pass"])

    def test_directory_loader_ignores_other_seed(self):
        rows = make_rows()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "all.csv"
            with p.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row[k] for k in FIELDS})
                other = {k: rows[0][k] for k in FIELDS}
                other["seed"] = "18400"
                writer.writerow(other)
            result = adj.adjudicate_inputs([td])
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["n_rows_used"], 360)


if __name__ == "__main__":
    unittest.main()
