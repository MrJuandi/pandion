// Optional browser regression check: requires Node, Playwright and Chromium.
const {chromium} = require('playwright');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const assert = require('node:assert/strict');
const Fleet = require('../pandion/viewer_assets/fleet.js');

// A refill service waypoint must keep positionAt fixed at the water source.
assert.deepEqual(Fleet.positionAt([
  {t: 0, p: [0, 0]}, {t: 60, p: [600, 0]}, {t: 240, p: [600, 0]},
], 150), [600, 0]);
const refillReplay = {aircraft_initial_state: [{index: 0, position_m: [0, 0], water_l: 0, capacity_l: 6137}],
  aircraft_track: [{time_s: 0, position_m: [0, 0]}, {time_s: 60, position_m: [600, 0]}, {time_s: 100, position_m: [600, 0]}],
  decisions: [{time_s: 240, decision_time_s: 0, completed_at_s: 240, aircraft_water_l: 6137,
    order: {type: 'RETURN_TO_REFILL'}}]};
const refillAtHorizon = Fleet.at(Fleet.derive(refillReplay)[0], 100);
assert.equal(refillAtHorizon.water, 0);
assert.equal(refillAtHorizon.phase, 'Heading to refill / refilling');

// Advance the render loop deterministically (camera transitions, damping) and return a snapshot.
const settle = (page) => page.evaluate(() => {
  for (const v of PandionViewer.views) for (let i = 0; i < 90; i++) v.renderer.frame(performance.now() + i * 16);
  return PandionViewer.views.map(v => v.renderer.canvas.toDataURL());
});
const camera = (page, index = 0) => page.evaluate(i => {
  const s = PandionViewer.views[i].renderer.cameraState();
  return {ortho: s.ortho, position: s.position ? s.position.toArray().map(n => Math.round(n)) : null, target: s.target ? s.target.toArray().map(n => Math.round(n)) : null};
}, index);

(async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'pandion-viewer-'));
  const target = path.join(directory, 'index.html');
  const fixture = `
import sys
from pathlib import Path
from pandion.viewer import write_viewer_html
base = dict(scenario_id='rectangular', policy='PPO de prueba', horizon_s=180,
    grid=dict(rows=2, cols=3, cell_size_m=100), origin_m=[300000,4600000], crs='EPSG:25831',
    arrival_time_s=[[0,None,120],[None,60,None]], fuel_grid=[[101,145,186],[0,101,145]],
    water_mask=[[False,False,False],[True,False,False]], elevation_m=[[300,310,320],[320,330,340]],
    environment=dict(schema_version='nightjar-environment-0.1',weather_mode='station_intervals',time_origin_utc='2025-08-15T12:00:00Z',
        terrain=dict(source=dict(provider='Copernicus',dataset='DEM 30 m')),
        station=dict(code='WX-1',name='Estación de prueba'),sources=['AEMET'],assumptions=['Humedad estimada'],
        forcing=[
            dict(start_s=0,end_s=60,available_at_s=60,midflame_wind_to_m_s=[3,4],moisture=dict(dead_1h=0.08),
                station_weather=dict(air_temperature_c=24,relative_humidity_pct=42,precipitation_mm=0.2,wind_speed_2m_m_s=5,wind_from_deg=180),
                interval_start_utc='2025-08-15T12:00:00Z',interval_end_utc='2025-08-15T12:01:00Z'),
            dict(start_s=60,end_s=180,available_at_s=180,midflame_wind_to_m_s=[0,8],moisture=dict(dead_1h=0.12),
                station_weather=dict(air_temperature_c=27,relative_humidity_pct=35,precipitation_mm=1.5,wind_speed_2m_m_s=9,wind_from_deg=270),
                interval_start_utc='2025-08-15T12:01:00Z',interval_end_utc='2025-08-15T12:03:00Z')]),
    affected_area_ha=3, aircraft_metrics=dict(total_water_dropped_l=6137,total_flight_distance_km=1.5,rejected_orders=0),
    aircraft_track=[dict(time_s=0,position_m=[50,50]),dict(time_s=90,position_m=[200,100]),dict(time_s=150,position_m=[250,150])],
    decisions=[
        dict(time_s=0, aircraft_water_l=6137, observation=dict(aircraft=dict(position_m=[50,50]),fire_reports=[[0,0]],environment_weather=None),order=dict(type='HOLD')),
        dict(time_s=60, aircraft_water_l=6137, observation=dict(aircraft=dict(position_m=[50,50]),fire_reports=[[0,0]],
            environment_weather=dict(available_at_s=60,midflame_wind_to_m_s=[3,4],moisture=dict(dead_1h=0.08),
                station_weather=dict(air_temperature_c=24,relative_humidity_pct=42,precipitation_mm=0.2))),order=dict(type='HOLD')),
        dict(time_s=150, aircraft_water_l=0, observation=dict(aircraft=dict(position_m=[200,100]),fire_reports=[[0,0],[1,1]]), order=dict(type='DROP',target_m=[200,100]))],
    drops=[dict(time_s=90,position_m=[200,100],newly_blocked_cells=[[1,2]])])
long = {**base,'policy':'Heurística de prueba','horizon_s':600,'environment':dict(schema_version='nightjar-environment-0.1',weather_mode='constant_synthetic',time_origin_utc='2025-08-15T12:00:00Z',terrain=dict(source=''),forcing=[]),'metadata':dict(wind_to_m_s=[2,0],terrain=dict(label='Relieve plano',source='',display_only=False))}
static = {**base,'scenario_id':'static','policy':'Cartografía · sin episodio','horizon_s':0,'affected_area_ha':None,'decisions':[],'drops':[],'aircraft_track':[],'environment':None}
high = {**base,'scenario_id':'high','elevation_m':[[1300,1310,1320],[1320,1330,1340]]}
third = {**base,'policy':'Tercera política','horizon_s':900,'affected_area_ha':9}
fleet = {**base,'scenario_id':'fleet','aircraft_metrics':dict(total_water_dropped_l=12274),
    'aircraft_initial_state':[dict(index=i,position_m=[50 if i == 0 else 250,50],water_l=6137,capacity_l=6137) for i in range(2)],
    'aircraft_track':[dict(time_s=0,position_m=[50,50],aircraft_index=0),dict(time_s=90,position_m=[200,100],aircraft_index=0),
                      dict(time_s=0,position_m=[250,50],aircraft_index=1),dict(time_s=120,position_m=[50,100],aircraft_index=1)],
    'decisions':[dict(time_s=90,decision_time_s=0,aircraft_index=0,aircraft_water_l=0,order=dict(type='DROP',target_m=[200,100])),
                 dict(time_s=120,decision_time_s=0,aircraft_index=1,aircraft_water_l=0,order=dict(type='DROP',target_m=[50,100]))],
    'drops':[dict(time_s=90,aircraft_index=0,water_l=6137,position_m=[200,100],treated_cells=[[1,2]]),
             dict(time_s=120,aircraft_index=1,water_l=6137,position_m=[50,100],treated_cells=[[1,0]])]}
write_viewer_html([base,long,third,static,high,fleet],Path(sys.argv[1]))
`;
  execFileSync(process.env.PANDION_PYTHON || 'python3', ['-c', fixture, target], {cwd: path.resolve(__dirname, '..')});
  const launch = (args = []) => chromium.launch({headless: true, args, ...(process.env.PANDION_CHROMIUM_EXECUTABLE ? {executablePath: process.env.PANDION_CHROMIUM_EXECUTABLE} : {})});
  let browser;
  try {
    browser = await launch(['--use-angle=swiftshader', '--enable-unsafe-swiftshader']);
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(pathToFileURL(target).href);
    const slider = page.locator('#time-slider');
    const canvas = page.locator('#map-canvas');

    // Default view: interactive WebGL scene in 3D.
    assert.equal(await page.locator('#view-3d').getAttribute('aria-pressed'), 'true');
    assert.equal(await canvas.getAttribute('data-renderer'), 'webgl');
    assert.equal(await page.locator('#condiciones').isVisible(), false);
    assert.equal(await page.locator('#datos').isVisible(), false);
    assert.equal(await page.locator('#inspector-action').textContent(), 'Hold');
    assert.ok(await page.evaluate(() => PandionViewer.views[0].renderer.terrain.mesh.geometry.index.count > 0));

    // Separate views keep the selected time; the timeline follows the conditions page.
    await page.locator('.nav-link[href="#condiciones"]').click();
    assert.equal(await page.locator('#condiciones').isVisible(), true);
    assert.equal(await canvas.isVisible(), false);
    assert.equal(await slider.isVisible(), true);
    await slider.fill('70');
    assert.match(await page.locator('#environment-temperature').textContent(), /27/);
    await page.locator('.nav-link[href="#datos"]').click();
    assert.equal(await page.locator('#datos').isVisible(), true);
    assert.equal(await page.locator('.nav-link[href="#datos"]').getAttribute('aria-current'), 'page');
    await page.locator('.nav-link[href="#mapa"]').click();
    assert.equal(await slider.inputValue(), '70');
    const fit = await canvas.evaluate(el => ({height: el.getBoundingClientRect().height, container: el.parentElement.getBoundingClientRect().height}));
    assert.ok(fit.height <= fit.container + 1, 'Canvas must fit its viewport after returning from a hidden view');

    // Scenario/policy boundaries, final metrics and causal environment values.
    await slider.fill('0');
    assert.equal(await page.locator('#scenario-select option').count(), 4);
    assert.equal(await slider.getAttribute('max'), '180');
    assert.match(await page.locator('#m-water').textContent(), /6137|6.137/);
    assert.match(await page.locator('#m-flight').textContent(), /1,5/);
    assert.equal(await page.locator('#environment-mode').textContent(), 'Station · intervals');
    assert.match(await page.locator('#environment-time').textContent(), /2025-08-15T12:00:00Z/);
    assert.match(await page.locator('#environment-wind').textContent(), /5(?:\.0)? m\/s · NE/);
    assert.match(await page.locator('#environment-temperature').textContent(), /24(?:\.0)? °C/);
    assert.equal(await page.locator('#environment-humidity').textContent(), '42 %');
    assert.equal(await page.locator('#environment-rain').textContent(), '0,2 mm');
    assert.match(await page.locator('#environment-moisture').textContent(), /8(?:,0)? %/);
    assert.equal(await page.locator('#environment-elevation').textContent(), '300–340 m');
    assert.match(await page.locator('#environment-elevation-use').textContent(), /used by the physics/);
    assert.equal(await page.locator('#environment-observation').textContent(), 'Not available at this time.');
    assert.match(await page.locator('#hud-clock').textContent(), /12:00:00 UTC/);
    assert.match(await page.locator('#hud-conditions').textContent(), /Terrain exaggerated ×1\.5/);
    await page.selectOption('#policy-select', '1');
    assert.equal(await page.locator('#environment-mode').textContent(), 'Constant synthetic wind');
    assert.match(await page.locator('#environment-wind').textContent(), /2 m\/s · E/);
    assert.equal(await page.locator('#environment-temperature').textContent(), '—');
    assert.match(await page.locator('#environment-elevation-use').textContent(), /DEM source and physics use not recorded/);
    await page.selectOption('#policy-select', '0');

    // Live state derived from the arrival raster and recorded aircraft data.
    await slider.fill('90');
    assert.equal(await page.locator('#live-area').textContent(), '2');
    assert.match(await page.locator('#live-drops').textContent(), /1 descargas/);
    assert.match(await page.locator('#live-phase').textContent(), /En espera/);
    await slider.fill('150');
    assert.equal(await page.locator('#live-water').textContent(), '6137');
    assert.match(await page.locator('#live-tank').textContent(), /0 \/ 6137 L/);
    assert.equal(await page.evaluate(() => PandionViewer.views[0].renderer.aircraft.visible), true);

    // Independent aircraft tracks, water accounting and selected chase target.
    await page.selectOption('#scenario-select', 'fleet');
    await slider.fill('45');
    assert.equal(await page.locator('#fleet-count').textContent(), '2 unidades');
    assert.equal(await page.locator('#fleet-body tr').count(), 2);
    assert.equal(await page.locator('#live-water').textContent(), '0');
    const planes = await page.evaluate(() => {
      const renderer = PandionViewer.views[0].renderer;
      return [...renderer.aircraftVisuals.values()].map(v => ({visible: v.aircraft.visible, position: v.aircraft.position.toArray()}));
    });
    assert.equal(planes.filter(p => p.visible).length, 2);
    assert.notDeepEqual(planes[0].position, planes[1].position);
    await slider.fill('90');
    assert.equal(await page.locator('#live-water').textContent(), '6137');
    assert.match(await page.locator('#fleet-body tr[data-aircraft-index="1"]').textContent(), /6137/);
    await slider.fill('120');
    assert.equal((await page.locator('#live-water').textContent()).replace(/[., ]/g, ''), '12274');
    await page.selectOption('#fleet-select', '1');
    assert.equal(await page.evaluate(() => {
      const r = PandionViewer.views[0].renderer; return r.aircraft === r.aircraftVisuals.get(1).aircraft;
    }), true);
    assert.match(await page.locator('.decision-item').first().textContent(), /NJ-01/);
    assert.match(await page.locator('.decision-item').nth(1).textContent(), /NJ-02/);
    await page.selectOption('#scenario-select', 'rectangular');
    await slider.fill('150');

    // Terrain geometry is relative to the minimum height: an absolute datum shift changes nothing.
    const heights = () => page.evaluate(() => Array.from(PandionViewer.views[0].renderer.terrain.heights).map(v => Math.round(v * 100)).join(','));
    const relief = await heights();
    await page.selectOption('#scenario-select', 'high');
    assert.equal(await heights(), relief);
    await page.selectOption('#scenario-select', 'rectangular');

    // Camera controls: zoom, reset, keyboard pan, 2D orthographic, chase.
    await settle(page);
    const home = await camera(page);
    await page.locator('#zoom-in').click();
    await settle(page);
    assert.notDeepEqual(await camera(page), home);
    await page.locator('#camera-reset').click();
    await settle(page);
    assert.deepEqual(await camera(page), home);
    await canvas.focus();
    await page.keyboard.press('ArrowRight');
    assert.notDeepEqual((await camera(page)).target, home.target);
    await page.keyboard.press('Home');
    await settle(page);
    assert.deepEqual(await camera(page), home);
    assert.ok((await page.locator('#map-scale i').boundingBox()).width > 0);

    // Layers and colour styles change the rendered texture.
    const before = (await settle(page))[0];
    await page.locator('#layer-contours').uncheck();
    assert.notEqual((await settle(page))[0], before);
    await page.locator('#layer-contours').check();
    await page.selectOption('#terrain-style', 'elevation');
    assert.match(await page.locator('#terrain-color-key').textContent(), /300–340 m/);
    await page.selectOption('#terrain-style', 'slope');
    assert.match(await page.locator('#terrain-color-key').textContent(), /Pendiente/);
    await page.selectOption('#terrain-style', 'fuel');
    await page.locator('#view-2d').click();
    await settle(page);
    assert.equal((await camera(page)).ortho, true);
    assert.equal(await page.locator('#view-2d').getAttribute('aria-pressed'), 'true');
    await page.locator('#view-3d').click();
    await page.locator('#cam-chase').click();
    assert.equal(await page.evaluate(() => PandionViewer.cameraMode), 'chase');
    await page.locator('#view-3d').click();

    // Causal decision inspector and delivered observations.
    await slider.fill('120');
    assert.match(await page.locator('#environment-time').textContent(), /2025-08-15T12:02:00Z/);
    assert.match(await page.locator('#environment-wind').textContent(), /8(?:\.0)? m\/s · N/);
    assert.equal(await page.locator('#environment-rain').textContent(), '1,5 mm');
    assert.match(await page.locator('#environment-source').textContent(), /disponible al cierre t = 180 s/);
    assert.match(await page.locator('#environment-observation').textContent(), /24(?:\.0)? °C/);
    assert.equal(await page.locator('#inspector-action').textContent(), 'Hold');
    assert.match(await page.locator('#inspector-time').textContent(), /01:00/);
    assert.equal(await page.locator('.decision-item.current').count(), 1);
    await slider.fill('150');
    assert.match(await page.locator('#inspector-action').textContent(), /Drop/);
    assert.match(await page.locator('#inspector-observation').textContent(), /2 celdas/);
    await slider.fill('120');
    const texture = () => page.evaluate(() => PandionViewer.views[0].renderer.painter.color.toDataURL());
    const simulator = await texture();
    await page.selectOption('#state-select', 'observation');
    assert.notEqual(await texture(), simulator);
    await page.selectOption('#state-select', 'simulator');

    // Strict two-policy comparison with pair-specific horizons and a shared camera.
    await page.locator('#compare-toggle').click();
    assert.equal(await page.locator('#map-panels canvas').count(), 2);
    assert.equal(await page.locator('.nav-link[href="#comparacion"]').getAttribute('aria-current'), 'page');
    assert.equal(await slider.getAttribute('max'), '600');
    assert.equal(await page.locator('#compare-policy-select option').count(), 2);
    assert.equal(await page.locator('.compare-table tbody tr').count(), 2);
    await page.selectOption('#compare-policy-select', '2');
    assert.equal(await page.locator('#map-panels canvas').count(), 2);
    assert.equal(await slider.getAttribute('max'), '900');
    assert.match(await page.locator('.map-panel-title').nth(1).textContent(), /Tercera/);
    await page.selectOption('#compare-policy-select', '1');
    assert.equal(await slider.getAttribute('max'), '600');
    await page.locator('#zoom-in').click();
    await settle(page);
    assert.deepEqual(await camera(page, 0), await camera(page, 1));
    await page.locator('.map-panel-title').nth(1).click();
    assert.equal(await page.locator('#policy-select').inputValue(), '1');
    assert.equal(await page.locator('#map-panels canvas').count(), 2);
    assert.equal(await page.locator('.selected-policy .map-panel-title').textContent(), 'Heurística de prueba');
    assert.equal(await page.locator('.map-panel-scale').count(), 2);
    await page.selectOption('#policy-select', '0');
    await slider.fill('500');
    await page.locator('#compare-toggle').click();
    assert.equal(await slider.getAttribute('max'), '180');
    assert.equal(await slider.inputValue(), '180');
    assert.equal(await page.locator('#map-panels canvas').count(), 1);

    // Historical cartography without an episode.
    await page.selectOption('#scenario-select', 'static');
    assert.equal(await page.locator('#map-panels canvas').count(), 1);
    assert.equal(await page.locator('#compare-toggle').getAttribute('aria-pressed'), 'false');
    assert.equal(await slider.getAttribute('max'), '0');
    assert.equal(await page.locator('#btn-play').isDisabled(), true);
    assert.equal(await page.locator('#m-area').textContent(), '—');
    assert.equal(await page.locator('#environment-mode').textContent(), 'Initial conditions · no episode');
    assert.equal(await page.evaluate(() => PandionViewer.views[0].renderer.aircraft.visible), false);
    await page.locator('.nav-link[href="#comparacion"]').click();
    assert.match(await page.locator('#comparison-results').textContent(), /no dispone de dos episodios/);
    await page.locator('.nav-link[href="#mapa"]').click();

    // Orbit drag, playback and reset.
    await page.selectOption('#scenario-select', 'rectangular');
    const box = await canvas.boundingBox();
    const beforeOrbit = await camera(page);
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 60, box.y + box.height / 2 + 20, {steps: 6});
    await page.mouse.up();
    await settle(page);
    assert.notDeepEqual(await camera(page), beforeOrbit);
    await page.locator('#camera-reset').click();
    await page.locator('#btn-play').click();
    await page.waitForFunction(() => Number(document.querySelector('#time-slider').value) > 0);
    await page.locator('#btn-play').click();
    await page.locator('#btn-reset').click();
    assert.equal(await slider.inputValue(), '0');

    // Narrow layout without horizontal overflow.
    await page.locator('#compare-toggle').click();
    await page.setViewportSize({width: 390, height: 844});
    const narrow = await page.evaluate(() => ({width: innerWidth, scroll: document.documentElement.scrollWidth}));
    assert.equal(narrow.scroll, narrow.width);
    assert.equal(await page.locator('#map-panels canvas').count(), 2);
    for (const view of ['condiciones', 'datos', 'mapa']) {
      await page.locator(`.nav-link[href="#${view}"]`).click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 390);
    }
    assert.equal((await page.locator('body').textContent()).includes('[object Object]'), false);
    await page.goto(pathToFileURL(target).href + '#datos');
    await page.reload();
    assert.equal(await page.locator('#datos').isVisible(), true);
    assert.equal(await page.locator('.workspace').isVisible(), false);
    assert.deepEqual(errors, []);

    // Explicit compatibility path when WebGL is unavailable.
    const fallbackBrowser = await launch(['--disable-webgl', '--disable-3d-apis']);
    try {
      const fallbackPage = await fallbackBrowser.newPage();
      const fallbackErrors = [];
      fallbackPage.on('pageerror', error => fallbackErrors.push(String(error)));
      await fallbackPage.goto(pathToFileURL(target).href);
      assert.equal(await fallbackPage.locator('#map-canvas').getAttribute('data-renderer'), 'canvas-fallback');
      assert.match(await fallbackPage.locator('#renderer-status').textContent(), /WebGL unavailable/);
      await fallbackPage.locator('#view-2d').click();
      assert.equal(await fallbackPage.locator('#map-canvas').getAttribute('data-renderer'), '2d');
      await fallbackPage.locator('#time-slider').fill('120');
      assert.equal(await fallbackPage.locator('#time-display').textContent(), '02:00');
      assert.deepEqual(fallbackErrors, []);
    } finally { await fallbackBrowser.close(); }
    console.log(`Runtime: Node ${process.version}; Chromium ${browser.version()}`);
    console.log('Browser checks: WebGL scene, scenario/policy boundaries, live fire/aircraft state, datum-invariant terrain, camera zoom/reset/keyboard/2D/chase/orbit, layers and colour styles, causal decision inspector, observation state, strict two-policy comparison with shared camera, static cartography, playback, view navigation, 390px layout, WebGL fallback, no runtime errors: OK');
  } finally {
    if (browser) await browser.close();
    fs.rmSync(directory, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
