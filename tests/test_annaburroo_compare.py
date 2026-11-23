from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pandion.annaburroo_compare import compare


class AnnaburrooCompareTests(unittest.TestCase):
    def test_audited_table_and_no_fit_record_count(self):
        table = Path("data/raw/annaburroo/Plot_Average_Data.csv")
        if not table.exists():
            self.skipTest("external source table unavailable; fetch with pandion.annaburroo")
        result = compare(table)
        self.assertEqual(result["source_rows"], 121)
        self.assertEqual(result["included"], 120)
        self.assertEqual(result["excluded"][0]["fire_id"], "F011")
        self.assertGreater(result["mae_m_s"], 0)

    def test_source_hash_must_match(self):
        with TemporaryDirectory() as directory:
            table = Path(directory) / "Plot_Average_Data.csv"
            table.write_text("FireID,MidFlwind.ms,DFMC.%,RSS.ms\nA001,1,5,1\n")
            with self.assertRaisesRegex(ValueError, "differs from the audited version"):
                compare(table)


if __name__ == "__main__":
    unittest.main()
