"""Local Alpaca credential loading and environment precedence."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.acquisition import read_credentials  # noqa: E402


class CredentialTests(unittest.TestCase):
    def test_loads_root_env_file_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text(
                "ALPACA_API_KEY=test-key\nALPACA_SECRET_KEY=test-secret\n",
                encoding="utf-8",
            )
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            with patch.dict(os.environ, {}, clear=True), patch("utils.acquisition.PROJECT_ROOT", root):
                previous = Path.cwd()
                try:
                    os.chdir(elsewhere)
                    self.assertEqual(read_credentials(), ("test-key", "test-secret"))
                finally:
                    os.chdir(previous)

    def test_environment_values_take_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text(
                "ALPACA_API_KEY=file-key\nALPACA_SECRET_KEY=file-secret\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {
                "ALPACA_API_KEY": "environment-key",
                "ALPACA_SECRET_KEY": "environment-secret",
            }, clear=True), patch("utils.acquisition.PROJECT_ROOT", root):
                self.assertEqual(read_credentials(), ("environment-key", "environment-secret"))

    def test_missing_or_incomplete_credentials_raise(self):
        for content in ("", "ALPACA_API_KEY=test-key\n", "ALPACA_API_KEY=\nALPACA_SECRET_KEY=\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / ".env").write_text(content, encoding="utf-8")
                with patch.dict(os.environ, {}, clear=True), patch("utils.acquisition.PROJECT_ROOT", root):
                    with self.assertRaisesRegex(RuntimeError, "ALPACA_API_KEY and ALPACA_SECRET_KEY"):
                        read_credentials()


if __name__ == "__main__":
    unittest.main()
