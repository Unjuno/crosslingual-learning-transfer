import json
import unittest

import state_signal_cycle_confirmatory_adjudicator as adj
from test_state_signal_cycle_confirmatory_adjudicator import make_rows


class AdjudicatorEdgeCaseTests(unittest.TestCase):
    def test_nonpositive_aligned_penalty_is_json_safe_fail(self):
        rows = make_rows()
        for row in rows:
            if row["mode"] == "Bsub_aligned":
                row["effect"] = "0"
        result = adj.adjudicate_rows(rows)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIsNone(result["pooled_ratio"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
