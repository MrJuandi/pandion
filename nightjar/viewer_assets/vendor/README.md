# Viewer dependency

The offline viewer embeds Three.js r128 (0.128.0) and its `OrbitControls` example, matching the rendering API used by the local FireLab reference viewer. The scene, terrain, aircraft and overlays are implemented separately in `../scene3d.js`; the interface is in `../viewer.js` (module `nightjar.viewer`).

- Upstream: https://github.com/mrdoob/three.js/tree/r128
- Source file: https://raw.githubusercontent.com/mrdoob/three.js/r128/build/three.min.js
- Local file: `three-r128.min.js`
- SHA-256: `9274bbcec8d96168626c732b5d31c775aa8cfb7eaa0599bec0c175908a2c1ce2`
- OrbitControls source: https://raw.githubusercontent.com/mrdoob/three.js/r128/examples/js/controls/OrbitControls.js
- OrbitControls local file: `OrbitControls-r128.js`
- OrbitControls SHA-256: `02bb4ade710f3e607329e37a21f098bc3ac70eb6e33daf8a65e79f4db785e7b2` (byte-identical to the FireLab-RL vendor copy)
- License: MIT (both files); the full upstream license is preserved in `THREE-LICENSE.txt` and embedded in generated HTML.
- Both files verified against upstream on 2026-10-01. No CDN or runtime network request is required.

The version is deliberately pinned for reproducibility; this is not a claim that r128 is the newest Three.js release. Future upgrades should preserve offline generation and verify terrain orientation, projection, picking, contour geometry and WebGL fallback.
