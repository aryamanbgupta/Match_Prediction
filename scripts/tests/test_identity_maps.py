from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from identity_maps import (  # noqa: E402
    assert_venue_alias_contract,
    canonicalize_match_id,
    canonicalize_venue,
    clear_identity_map_caches,
    load_venue_aliases,
    venue_alias_contract,
)


class VenueIdentityMapTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_identity_map_caches()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "aliases.csv"

    def tearDown(self) -> None:
        clear_identity_map_caches()
        self.temporary_directory.cleanup()

    def _write(self, rows: list[dict[str, str]]) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["version", "alias", "canonical", "status"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _row(
        alias: str,
        canonical: str,
        status: str = "active",
        version: str = "venue_aliases_v1",
    ) -> dict[str, str]:
        return {
            "version": version,
            "alias": alias,
            "canonical": canonical,
            "status": status,
        }

    def test_only_active_exact_aliases_are_consumed(self) -> None:
        self._write([
            self._row("Bay Oval", "Bay Oval, Mount Maunganui"),
            self._row(
                "Old Trafford",
                "Old Trafford, Manchester",
                status="proposed",
            ),
        ])
        aliases = load_venue_aliases(self.path)

        self.assertEqual(
            canonicalize_venue("Bay Oval", aliases),
            "Bay Oval, Mount Maunganui",
        )
        self.assertEqual(
            canonicalize_venue("Old Trafford", aliases),
            "Old Trafford",
        )
        self.assertEqual(
            canonicalize_venue("Unlisted Ground", aliases),
            "Unlisted Ground",
        )
        self.assertEqual(canonicalize_venue(None, aliases), "unknown")

    def test_match_id_venue_suffix_is_canonicalized(self) -> None:
        aliases = {"Bay Oval": "Bay Oval, Mount Maunganui"}
        self.assertEqual(
            canonicalize_match_id(
                "2026-01-02_Team_A_Team_B_Bay_Oval",
                aliases,
            ),
            "2026-01-02_Team_A_Team_B_Bay_Oval,_Mount_Maunganui",
        )

    def test_contract_records_version_hash_and_active_count(self) -> None:
        self._write([
            self._row("A", "A, City"),
            self._row("B", "B, City", status="rejected"),
        ])
        expected_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()

        self.assertEqual(
            venue_alias_contract(self.path),
            {
                "venue_alias_version": "venue_aliases_v1",
                "venue_alias_sha256": expected_hash,
                "venue_alias_active_count": 1,
            },
        )
        assert_venue_alias_contract(
            {
                "venue_alias_version": "venue_aliases_v1",
                "venue_alias_sha256": expected_hash,
                "venue_alias_active_count": "1",
            },
            context="test artifact",
            path=self.path,
        )

    def test_contract_mismatch_requires_rebuild(self) -> None:
        self._write([self._row("A", "A, City")])
        with self.assertRaisesRegex(RuntimeError, "Rebuild the artifact"):
            assert_venue_alias_contract(
                {},
                context="old cache",
                path=self.path,
            )

    def test_self_alias_is_rejected(self) -> None:
        self._write([self._row("A", "A")])
        with self.assertRaisesRegex(ValueError, "self-alias"):
            load_venue_aliases(self.path)

    def test_duplicate_alias_is_rejected(self) -> None:
        self._write([
            self._row("A", "A, City"),
            self._row("A", "A, Other City"),
        ])
        with self.assertRaisesRegex(ValueError, "repeats alias"):
            load_venue_aliases(self.path)

    def test_active_chain_is_rejected(self) -> None:
        self._write([
            self._row("A", "B"),
            self._row("B", "C"),
        ])
        with self.assertRaisesRegex(ValueError, "must not also be active aliases"):
            load_venue_aliases(self.path)

    def test_wrong_version_and_status_are_rejected(self) -> None:
        self._write([self._row("A", "B", version="venue_aliases_v2")])
        with self.assertRaisesRegex(ValueError, "expected 'venue_aliases_v1'"):
            load_venue_aliases(self.path)

        clear_identity_map_caches()
        self._write([self._row("A", "B", status="maybe")])
        with self.assertRaisesRegex(ValueError, "unsupported status"):
            load_venue_aliases(self.path)


if __name__ == "__main__":
    unittest.main()
