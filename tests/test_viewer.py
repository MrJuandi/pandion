import html
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from nightjar.viewer import build_viewer_html, write_viewer_html

ROOT = Path(__file__).parents[1]
JAVASCRIPT = ROOT / "nightjar" / "viewer_assets" / "workspace.js"


class ViewerTests(unittest.TestCase):
    def test_requires_nonempty_objects_and_finite_json(self):
        for replays in ([], [None], [[]], [{"horizon_s": float("nan")}], [{"horizon_s": float("inf")}], [{"data": object()}]):
            with self.subTest(replays=replays), self.assertRaises(ValueError):
                build_viewer_html(replays)

    def test_payload_round_trips_and_cannot_close_script(self):
        title = '__PAYLOAD__ __TITLE__ "quoted" </title> 火'
        replay = {"scenario_id": '__JS__ __TITLE__ </script><script>alert(1)</script>',
                  "nested": ["__CSS__", "<tag>", "雪", "\u2028\u2029", "A&B"]}
        page = build_viewer_html([replay], title)
        document_title = re.search(r"<title>(.*?)</title>", page, re.S).group(1)
        payload = re.search(r"const INITIAL_REPLAYS = (.*);", page).group(1)
        self.assertEqual(html.unescape(document_title), title)
        self.assertEqual(json.loads(payload), [replay])
        self.assertEqual(page.count("</script>"), 1)
        self.assertIn(r"\u003c/script\u003e", page)
        self.assertNotIn("__PAYLOAD__", payload)
        self.assertIn("__CSS__", json.loads(payload)[0]["nested"])

    def test_portable_assets_are_bundled(self):
        page = build_viewer_html([{"scenario_id": "offline-case"}])
        self.assertIn("<!DOCTYPE html>", page)
        self.assertIn("Nightjar Research Workspace", page)
        self.assertNotRegex(page, r'<(?:script|link)[^>]+(?:src|href)=')
        self.assertNotIn("@import", page)
        self.assertNotIn("fetch(", page)
        self.assertIn("Final outcomes", page)
        self.assertIn('aria-label="Replay time"', page)

    def test_model_descriptors_are_embedded_as_inert_finite_json(self):
        models = [{"schema_version": "nightjar-viewer-model-0.1", "id": "checkpoint",
                   "name": "</script><script>unsafe()</script> __MODELS__", "role": "controller"}]
        page = build_viewer_html([{"scenario_id": "offline-case"}], models=models)
        payload = re.search(r"const INITIAL_MODELS = (.*);", page).group(1)
        self.assertEqual(json.loads(payload), models)
        self.assertEqual(page.count("</script>"), 1)
        for invalid in ([None], [{"parameter_count": float("nan")}], [{"data": object()}]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_viewer_html([{}], models=invalid)

    def test_write_creates_parent_and_preserves_custom_title(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "viewer.html"
            write_viewer_html([{"scenario_id": "offline-case"}], target, title="Comparison")
            self.assertIn("<title>Comparison</title>", target.read_text())

    def test_cli_demo_uses_only_reproducible_development_case(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "demo.html"
            completed = subprocess.run(
                [shutil.which("python3"), "-m", "nightjar.viewer", "--demo", "--output", str(target)],
                cwd=ROOT, check=True, text=True, capture_output=True,
            )
            self.assertIn(str(target), completed.stdout)
            page = target.read_text()
            payload = re.search(r"const INITIAL_REPLAYS = (.*);", page).group(1)
            replays = json.loads(payload)
            self.assertEqual({r["scenario_id"] for r in replays}, {"discrimination-development-00"})
            self.assertEqual({r["policy"] for r in replays}, {"heuristic", "containment_heuristic", "no_intervention"})
            self.assertEqual(len({r["scenario_sha256"] for r in replays}), 1)
            self.assertEqual({r["seed"] for r in replays}, {500000})
            self.assertTrue(all(r["provenance"] == "synthetic" for r in replays))

    def test_cli_reports_missing_malformed_and_conflicting_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.json"
            bad.write_text("not json")
            for args in ([], [str(bad)], [str(bad.with_name("missing.json"))], ["--demo", str(bad)]):
                with self.subTest(args=args):
                    completed = subprocess.run([shutil.which("python3"), "-m", "nightjar.viewer", *args],
                                               cwd=ROOT, text=True, capture_output=True)
                    self.assertEqual(completed.returncode, 2)
                    self.assertIn("error:", completed.stderr)
                    self.assertNotIn("Traceback", completed.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable; JavaScript runtime tests skipped")
    def test_pair_deltas_and_recorded_area_steps(self):
        script = r'''
const assert = require('node:assert/strict');
const ui = require(process.argv[1]);
const candidate = {
  horizon_s: 20, scenario_sha256: 'paired-scenario',
  grid: {rows: 2, cols: 3, cell_size_m: 50},
  arrival_time_s: [[0, 5, 5], [20, null, 25]],
  affected_area_ha: 1,
  aircraft_metrics: {water_dropped_l: 100, flight_s: 15, drops_count: 2, rejected_orders: 0}
};
const reference = {...candidate, affected_area_ha: 1.5,
  aircraft_metrics: {water_dropped_l: 200, flight_s: 10, drops_count: 3, rejected_orders: 1}};
const series = ui.areaSeries(candidate);
assert.equal(series.available, true);
assert.deepEqual(series.points, [{time_s:0, area_ha:0.25}, {time_s:5, area_ha:0.75}, {time_s:20, area_ha:1}]);
assert.equal(series.unit, 'ha');
for (const missing of [{}, {...candidate, arrival_time_s: undefined},
  {...candidate, grid: {rows:2, cols:3}}, {...candidate, horizon_s: undefined}]) {
  assert.equal(ui.areaSeries(missing).available, false);
  assert.deepEqual(ui.areaSeries(missing).points, []);
}
const rows = ui.comparisonRows(candidate, reference);
assert.equal(rows.compatible, true);
const areas = rows.rows.find(row=>row.unit==='ha');
assert.equal(areas.delta, -0.5);
assert.equal(rows.rows.find(row=>row.unit==='L').delta, -100);
assert.equal(rows.rows.find(row=>row.unit==='s').delta, 5);
for (const unrelated of [candidate, {...reference, scenario_sha256: undefined},
  {...reference, horizon_s: 10}, {...reference, grid:{rows:2,cols:3,cell_size_m:100}}]) {
  assert.equal(ui.comparisonRows(candidate, unrelated).rows.every(row=>row.delta===null), true);
}
const missing = ui.comparisonRows(candidate, {...reference, affected_area_ha:undefined,
  aircraft_metrics: {water_dropped_l: '200', rejected_orders:0}});
assert.equal(missing.rows.find(row=>row.unit==='ha').delta, null);
assert.equal(missing.rows.find(row=>row.unit==='L').delta, null);
const drops = ui.comparisonRows({...candidate, drops: [{time_s:2}], aircraft_metrics:{}},
  {...reference, drops:[], aircraft_metrics:{}}).rows.find(row=>row.key==='drops');
assert.equal(drops.candidate, 1);
assert.equal(drops.reference, 0);
assert.equal(drops.delta, 1);
const model={schema_version:'nightjar-viewer-model-0.1',id:'controller-fixture',role:'controller',
  config:{grid_rows:2,grid_cols:3,recurrent:true},source_files:[{file:'nightjar/model.py',sha256:'fixture'}]};
assert.equal(ui.validateModelRecord(model),model);
for(const invalid of [null,[],{...model,role:'unknown'},{...model,config:[]},
  {...model,config:{grid_rows:0}}, {...model,config:{recurrent:'true'}},
  {...model,source_files:[null]},{...model,details:[[]]},
  {...model,checkpoint:{bytes:-1}}, {...model,parameter_count:Infinity}])
  assert.throws(()=>ui.validateModelRecord(invalid));
process.stdout.write('Comparison invariants passed');
'''
        completed = subprocess.run([shutil.which("node"), "-e", script, str(JAVASCRIPT)],
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "Comparison invariants passed")

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable; JavaScript runtime tests skipped")
    def test_canvas_fits_frame_without_stretching_cells(self):
        script = r'''
const assert = require('node:assert/strict');
const ui = require(process.argv[1]);
for (const geometry of [{rows:11,cols:11}, {rows:4,cols:12}, {rows:12,cols:4}]) {
  for (const [width,height] of [[178.5,360], [960,400], [380,240], [1,1]]) {
    const size = ui.fitGridToFrame(geometry,width,height);
    assert.ok(size.width <= width + 1e-9 && size.height <= height + 1e-9);
    assert.ok(Math.abs(size.width / geometry.cols - size.height / geometry.rows) < 1e-9);
    assert.ok(Math.abs(size.width-width)<1e-9 || Math.abs(size.height-height)<1e-9);
  }
}
for (const [geometry,width,height] of [[null,100,100], [{rows:0,cols:2},100,100],
  [{rows:2,cols:2},0,100], [{rows:2,cols:2},100,NaN]]) {
  assert.equal(ui.fitGridToFrame(geometry,width,height),null);
}
'''
        completed = subprocess.run([shutil.which("node"), "-e", script, str(JAVASCRIPT)],
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable; JavaScript runtime tests skipped")
    def test_javascript_syntax(self):
        subprocess.run([shutil.which("node"), "--check", str(JAVASCRIPT)], check=True, capture_output=True)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable; JavaScript runtime tests skipped")
    def test_runtime_time_geometry_comparison_and_import_invariants(self):
        script = r'''
const assert = require('node:assert/strict');
const ui = require(process.argv[1]);
const oldObservation = {aircraft: {water_l: 100}, cell_size_m: 20};
const currentObservation = {aircraft: {water_l: 0}, cell_size_m: 20};
const replay = {
  horizon_s: 30, scenario_sha256: 'fixed-input-hash',
  arrival_time_s: [[0, null, 10, null], [5, null, null, 15]],
  decisions: [{time_s: 12, observation: currentObservation},
              {time_s: 2, observation: oldObservation},
              {time_s: 7, observation: currentObservation}],
  pending_observation: {aircraft: {water_l: 9999}}
};
assert.equal(ui.latestDecisionAt(replay, 1), null);
assert.equal(ui.observationAt(replay, 1), null);
assert.equal(ui.latestDecisionAt(replay, 7).time_s, 7);
assert.equal(ui.observationAt(replay, 3), oldObservation);
assert.equal(ui.observationAt(replay, 30), currentObservation);
const geometry = ui.gridGeometry(replay);
assert.deepEqual(geometry, {rows: 2, cols: 4, cellSize: 20});
assert.deepEqual(ui.positionOnCanvas([40, 20], geometry, 400, 200), {x: 250, y: 50});
assert.deepEqual(ui.cellAtPoint(250, 50, geometry, 400, 200), {row: 1, col: 2});
assert.deepEqual(ui.cellAtPoint(0, 199, geometry, 400, 200), {row: 0, col: 0});
assert.equal(ui.cellAtPoint(400, 100, geometry, 400, 200), null);
assert.equal(ui.positionOnCanvas([-100, 20], geometry, 400, 200), null);
assert.equal(ui.gridGeometry({}), null);
assert.equal(ui.positionOnCanvas([0, 0], {rows: 2, cols: 4, cellSize: null}, 400, 200), null);
assert.equal(ui.reachedAreaAt(replay, 0), 0.04);
assert.equal(ui.reachedAreaAt(replay, 10), 0.12);
assert.equal(ui.replayTime({...replay, horizon_s: 8}, 100), 8);
assert.equal(ui.reachedAreaAt({...replay, horizon_s: 8}, 100), 0.08);
assert.equal(ui.latestDecisionAt({...replay, horizon_s: 8}, 100).time_s, 7);
assert.equal(ui.reachedAreaAt({grid: {rows: 2, cols: 4, cell_size_m: 20}}, 10), null);
assert.equal(ui.compatibleComparison([replay, {...replay, policy: 'other'}]).paired, true);
assert.equal(ui.compatibleComparison([replay, {...replay, scenario_sha256: undefined}]).paired, false);
assert.equal(ui.compatibleComparison([replay, {...replay, horizon_s: 20}]).paired, false);
assert.equal(ui.compatibleComparison([replay, {...replay, grid: {rows: 2, cols: 4, cell_size_m: 10}}]).paired, false);
assert.equal(ui.validateReplay(replay), replay);
for (const invalid of [null, [], {horizon_s: -1}, {horizon_s: '30'},
  {arrival_time_s: [[1], [1, 2]]}, {arrival_time_s: [[-1]]},
  {decisions: [{time_s: -1}]}, {decisions: [{time_s: 1, observation: {fuel_model: [[1], [1, 2]]}}]},
  {drops: [{time_s: '10'}]}, {arrival_time_s: [[0, 1]], grid: {rows: 5, cols: 5}}
]) assert.throws(() => ui.validateReplay(invalid));
const geographic = {
  schema_version: 'nightjar-geographic-replay-0.1', horizon_s: 60,
  grid: {rows: 2, cols: 4, cell_size_m: 100},
  decisions: [{time_s: 10, observation: {cell_size_m: 100,
    fuel_model: [['101', '0', '141', '0'], ['101', '101', '0', '0']]}}]
};
assert.deepEqual(ui.gridGeometry(geographic), {rows: 2, cols: 4, cellSize: 100});
assert.equal(ui.reachedAreaAt(geographic, 60), null);
assert.equal(ui.observationAt(geographic, 9), null);
assert.equal(ui.validateReplay(geographic), geographic);
process.stdout.write('Replay invariants passed');
'''
        completed = subprocess.run([shutil.which("node"), "-e", script, str(JAVASCRIPT)],
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "Replay invariants passed")


if __name__ == "__main__":
    unittest.main()
