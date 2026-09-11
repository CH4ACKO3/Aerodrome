# Validation — 2026-09-11

- Native Windows: Node 24.19.0, npm 11.8.0. Dependency install completed with zero reported vulnerabilities.
- `npm run check`: zero errors, warnings or hints.
- Astro production build: 27 static pages, Pagefind search index and sitemap.
- Output validation: all `/Aerodrome/` internal paths exist; no rendered KaTeX errors.
- WSL Ubuntu, Python 3.14.7t: full simulation suite **218 passed** (144 seconds), including seven new teaching tests. Existing python-control emits 36 NumPy deprecation warnings.
- Fresh `uv sync --locked --all-extras` succeeded; installed `aerodrome-teach --help` succeeded.
- Actual browser: local cookie/session handshake succeeded; changed target velocity from 2 to 3 m/s, submitted a job, received the JAX World result (final error about 0.0483 m/s).
- Static preview without Python API: reference data remained available and run button was disabled.
- Desktop 1440px and mobile 390px viewport inspected. Mobile document width 375px excludes the native 15px scrollbar; no horizontal overflow. Math rendered without errors.
- HTTP tests cover missing session, invalid Host, cross-origin POST, version mismatch, path traversal and successful asynchronous job retrieval. Numerical test compares the World trajectory to the independent sampled feedback recurrence.

The static build reports Starlight warnings for its optional empty i18n collection and default 404 entry; pages and search build successfully. This turn did not rerun Node builds on macOS/Linux, configure GitHub Pages, push branches or deploy. Earlier core platform results remain in `ngc/docs/platform-builds.md`.

Current limits: only the rigid-velocity lesson runs through the bridge; no job cancellation, persistent history or streamed 3D renderer. Each job constructs its JAX experiment, so compilation is not cached between jobs. The server is intended for one user's loopback teaching session.

## Scene integration follow-up

- Added Three.js 0.186.0, 3d-tiles-renderer 0.5.2, CesiumJS 1.145.0 and MapLibre GL JS 6.9.0 using npm lockfile.
- Production site now contains 28 pages; internal paths and math validation pass.
- Five Node tests verify imported Python Scene/Frame data, NED/asset axes, WGS84 ECEF conversion, Cesium glTF axis correction, interpolation and invalid input rejection.
- Seven Python teaching tests pass with added Frame export and independent position integration validation.
- Focused Python teaching/rendering/backend regression suite: **24 passed**. No simulation core algorithm was changed.
- Actual browser verified all four Three.js environment presets, local 3D Tiles fixture, GLB model rendering, playback, articulation control, JSON result import, and Cesium/MapLibre initialization without console errors.
- No external city or terrain service was selected; offline fixtures and empty/default globe/map modes were exercised. CORS, quota and credentials for a future provider remain provider-specific validation.
- Cesium (~4.9 MB minified JavaScript before transfer compression) and MapLibre (~1.1 MB) remain lazy chunks. Vite reports the large optional chunks; this is not evidence of those engines loading on ordinary lesson pages.
- 3D Tiles Renderer warns that 1.1 feature support is partial; the bundled simple glTF-content fixture was exercised, not all 3D Tiles 1.1 extensions.
- Mobile 390px viewport has no horizontal document overflow; ocean/aircraft visible. Deliberately missing GLB returns a readable error, corrected URL recovers, release removes all canvases, and reload succeeds.
