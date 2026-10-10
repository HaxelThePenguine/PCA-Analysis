"""Primary and sensitivity results must stay distinct in reports."""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reporting.network import print_summary  # noqa: E402


class ReportingScopeTest(unittest.TestCase):
    def test_summary_does_not_report_sensitivity_edges_as_primary(self):
        forecasts = pd.DataFrame([
            {"spec_name": "factor_adjusted_120_expanding", "model": "own_har", "qlike": .2},
            {"spec_name": "factor_adjusted_120_expanding", "model": "network_har_l1_1se", "qlike": .19},
        ])
        hac = pd.DataFrame(columns=["level", "spec_name"])
        edges = pd.DataFrame([
            {"spec_name": "factor_adjusted_120_expanding", "stable_edge": False, "selection_probability": .2, "edge_id": "JPM->BAC"},
            {"spec_name": "factor_adjusted_60_expanding", "stable_edge": True, "selection_probability": 1., "edge_id": "C->WFC"},
        ])
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            summary = print_summary(forecasts, hac, edges, out_dir=Path(temporary))
        self.assertIn("No directed edge", summary)
        self.assertNotIn("C->WFC", summary)


if __name__ == "__main__":
    unittest.main()
