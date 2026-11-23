import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pandion.terrain_acquire import acquire_terrain


class TerrainAcquireTests(unittest.TestCase):
    def scenario(self):
        return {'activation_id':'EMSR578','crs':{'code':'EPSG:25831'},
                'grid':{'rows':2,'cols':2,'cell_size_m':100,'origin_utm_easting_m':331136.44,'origin_utm_northing_m':4642171.57}}

    def test_acquisition_records_source_request_hash_crs_and_timestamp(self):
        def fetch(command,check):
            self.assertTrue(check)
            self.assertIn('crs=EPSG%3A25831',command[-3])
            Path(command[-1]).write_text('ncols 2\nnrows 2\nxllcorner 331136.4\nyllcorner 4642171.6\ncellsize 100\n10 20\n0 30\n')
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'dem'
            with patch('pandion.terrain_acquire.load_geographic_split',side_effect=lambda split,*args:[self.scenario()] if split=='development' else []), patch('pandion.terrain_acquire.subprocess.run',side_effect=fetch):
                sources=acquire_terrain(Path('unused'),out)
            manifest=json.loads((out/'sources.json').read_text())
            self.assertEqual(manifest['schema_version'],'nightjar-terrain-sources-0.1')
            self.assertEqual(manifest['units']['elevation'],'m')
            self.assertIsNone(manifest['seed'])
            self.assertTrue(manifest['created_at_utc'].endswith('Z'))
            self.assertEqual(sources['EMSR578']['crs'],'EPSG:25831')
            self.assertEqual(len(sources['EMSR578']['raw_sha256']),64)
            with self.assertRaises(ValueError):acquire_terrain(Path('unused'),out)

    def test_missing_height_does_not_produce_a_source_manifest(self):
        def fetch(command,check):
            Path(command[-1]).write_text('ncols 2\nnrows 2\nxllcorner 331136.4\nyllcorner 4642171.6\ncellsize 100\nNODATA_value -9999\n10 20\n-9999 30\n')
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'dem'
            with patch('pandion.terrain_acquire.load_geographic_split',side_effect=lambda split,*args:[self.scenario()] if split=='development' else []), patch('pandion.terrain_acquire.subprocess.run',side_effect=fetch):
                with self.assertRaises(ValueError):acquire_terrain(Path('unused'),out)
            self.assertFalse((out/'sources.json').exists())

    def test_activation_filter_acquires_only_named_spanish_incidents(self):
        extra={'activation_id':'EMSR602_AOI02','crs':{'code':'EPSG:25829'},
               'grid':{'rows':2,'cols':2,'cell_size_m':100,'origin_utm_easting_m':700000.0,'origin_utm_northing_m':4700000.0}}
        def fetch(command,check):
            self.assertIn('crs=EPSG%3A25829',command[-3])
            Path(command[-1]).write_text('ncols 2\nnrows 2\nxllcorner 700000\nyllcorner 4700000\ncellsize 100\n10 20\n0 30\n')
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'dem'
            loader=lambda split,*args:[self.scenario(),extra] if split=='development' else []
            with patch('pandion.terrain_acquire.load_geographic_split',side_effect=loader), patch('pandion.terrain_acquire.subprocess.run',side_effect=fetch):
                sources=acquire_terrain(Path('unused'),out,['EMSR602_AOI02'])
                self.assertEqual(set(sources),{'EMSR602_AOI02'})
                self.assertTrue((out/'emsr602-aoi02.asc').exists())
                with self.assertRaises(ValueError):acquire_terrain(Path('unused'),Path(temporary)/'other',['EMSR999'])

if __name__=='__main__':unittest.main()
