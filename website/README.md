# Probabilistic Aviation

## Course authoring

The main navigation follows seven chapters and appendices A–D, as confirmed by
the author. Chapter landing pages are in `src/content/docs/chapters`, appendix
landing pages in `src/content/docs/appendices`. Their titles are fixed; sections
and prose will be agreed and written incrementally. Do not automatically fill
chapter outlines or promote existing engineering notes into course sections.
Existing framework documentation and program examples remain under appendix C
in the sidebar, retaining their original URLs.

One static Astro + Starlight build, two execution modes. Node 24 is used to build;
end users of the local textbook only need the built files and the Python runtime.

```sh
npm ci
npm run check
npm run build
npm run preview
```

The site lives under `/Aerodrome/` both on GitHub Pages and locally. Source lessons
are in `src/content/docs`; `npm run prepare-content` imports `../ngc/docs` into the
ignored `reference` directory and generates the version manifest. Edit original
reference Markdown in `ngc/docs`, never the generated copy.

To run local experiments, from `ngc`:

```sh
uv sync --locked --all-extras
uv run --locked --all-extras aerodrome-teach --site ../website/dist
```

Open `http://127.0.0.1:8765/Aerodrome/`. The server binds only IPv4 loopback,
serves the same static build, and exposes a cookie-authenticated same-origin API.
It rejects mismatched manifests and only executes registered bounded experiments.
There is one worker, at most two outstanding jobs, and 32 retained jobs. Closing
the process loses results; download JSON to retain them. Jobs cannot yet be
cancelled and compilation is not cached across submissions. This is a local
teaching service, not a multi-user deployment server.

API v1: `GET /api/v1/hello`, `POST /api/v1/jobs`, `GET /api/v1/jobs/{id}`.
POST accepts `{manifest, experiment, config}`; download parameters from the lesson
for a complete example. Adding an experiment requires an explicitly registered
server runner, a capability ID, validation and a corresponding lesson component.
The current version exposes `rigid-velocity-v1` only. Its result now includes
Scene/Frame v1 recording data for the browser's 3D viewer. Continuous streaming
of running World instances is a later transport extension.

Regenerate the committed static reference with
`uv run --locked --all-extras python examples/export_teaching_baseline.py` from
`ngc`, then rebuild the website. The result comes from JAX and the existing rigid
body World, not a separate JavaScript approximation.

The GitHub workflow builds on `docs` and uploads a Pages artifact. Deployment
requires the repository's Pages source to be set to GitHub Actions. No deployment
or remote branch change happens merely by building this directory locally.

## Scene backends

`src/rendering` contains engine-neutral protocol validation/interpolation and
lazy Three.js, CesiumJS and MapLibre adapters. Three.js includes Sky, Water,
GLTFLoader, OrbitControls and 3d-tiles-renderer. Open the scene-viewer lesson to
play the committed real World recording and load optional data URLs.

Regenerate original GLB fixtures using `npm run generate-models`; regenerate the
turn recording using `python examples/export_scene_demo.py` from `ngc`.
`npm test` verifies axes, ECEF transforms, quaternion interpolation and the Python
serialization fixture. New local velocity results can be downloaded and imported.

Cesium's worker/assets directories are copied from the locked npm package by
prepare-content; do not edit or commit generated `public/vendor`. All engines are
loaded on demand. Cesium is a substantial optional download; no imagery, global
terrain subscription or private API key is bundled. MapLibre defaults to a blank
offline map. Real terrain/city content must be supplied with attribution and CORS.

Renderers expose `update`, `draw`, `resize`, `dispose`. ThreeBackend also accepts
RenderRequest v1 camera/joints and produces receipts. API callers supply an
explicit logical-URI-to-GLB URL map. The teaching UI can override the default
asset URL; arbitrary imported assets still require compatible asset axes.
