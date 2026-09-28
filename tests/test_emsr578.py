from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from nightjar import emsr578


class CEMSProductAuditTests(unittest.TestCase):
    def test_reads_versioned_polygon_and_detects_changed_package(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            target = directory / "del.zip"
            with ZipFile(target, "w") as archive:
                archive.writestr("EMSR578_AOI01_DEL_PRODUCT_observedEventA_r1_v1.json", json.dumps({
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "properties": {"notation": "Burnt area", "area": 12.5},
                                  "geometry": {"type": "Polygon", "coordinates": []}}],
                }))
            digest = sha256(target.read_bytes()).hexdigest()
            with patch.object(emsr578, "PRODUCTS", (("del", "PRODUCT", "2022-06-17T10:06:00Z", digest),)):
                output = emsr578.report(directory)
                self.assertEqual(output["observations"][0]["area_ha"], 12.5)
                self.assertEqual(output["observations"][0]["source_zip_sha256"], digest)
                target.write_bytes(target.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    emsr578.report(directory)


if __name__ == "__main__":
    unittest.main()
