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

## Textbook visual redesign

- Added self-hosted Inter Variable and a theme-aware editorial stylesheet; preserved chapter titles and lesson URLs.
- Astro check: 0 errors, warnings or hints. Production build and internal-path/math validation pass for 39 pages. Existing optional i18n/404 and lazy engine chunk warnings remain.
- Direct browser review: homepage at desktop and 390px mobile; chapter navigation page; light and dark scene layouts; Three.js loading and playback; numerical experiment controls and chart.
- Mobile scene document width is 375px within a 390px viewport (15px native scrollbar), with no horizontal document overflow.
- Local experiment returned a fresh CPU result with final velocity error 0.0176 m/s; inspected browser error log was empty.
- Local review screenshots: .impeccable/review/redesign-home-desktop.png, redesign-home-mobile.png and redesign-lab-desktop.png. No subagents were used.
- Style detector reported only undocumented radius values; DESIGN.md now records the intentional 4/6/8/12px scale.

## Navigation corrections

- Unified computed sidebar link/group-label font family and size (14px).
- Removed the unconditional right-sidebar padding that displaced the desktop TOC behind the header; removed the extra main-frame inline padding.
- Verified 1280px desktop, 1000px compact navigation after an anchor jump, and 390px mobile with expanded TOC. Header/desktop TOC boundary is 68px; compact navigation stays directly below the header and its current-section text stays inside its line box. Mobile document width remains 375px within a 390px viewport.
- Added bilingual site title: 概率飞行工程 · Probabilistic Aviation.
- Astro check reports 0 errors/warnings/hints; production build and link/math validation pass for 39 pages.

## Click/navigation stability

- Removed network-font dependence from logo/sidebar; kept selected sidebar weight unchanged and reserved the document scrollbar gutter. Body font now uses optional display rather than late swap.
- At 1280px, measured the logo and chapter 1/6 rectangles before and after scene loading, navigating to chapter 6, and opening search: coordinates and dimensions were identical in all four snapshots. This verifies settled layout, not every intermediate browser paint.
- Checked the 390px scene layout visually. Astro check: 0 errors/warnings/hints; 39-page production build and internal path/math validation pass.

## Compact reading layout — 2026-09-12

- Verified 1280px desktop with both panels visible/hidden and preference restoration after navigation. At 390px, the native menu still opens despite saved desktop collapse preferences; chapter 1.1 is correctly highlighted.
- Astro check has 0 errors/warnings/hints; static build and path/math validation pass for 45 pages.
