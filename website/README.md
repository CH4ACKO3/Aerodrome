# Probabilistic Aviation

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
The current version exposes `rigid-velocity-v1` only. Rendering/streaming of full
World scenes is a later extension of this protocol.

Regenerate the committed static reference with
`uv run --locked --all-extras python examples/export_teaching_baseline.py` from
`ngc`, then rebuild the website. The result comes from JAX and the existing rigid
body World, not a separate JavaScript approximation.

The GitHub workflow builds on `docs` and uploads a Pages artifact. Deployment
requires the repository's Pages source to be set to GitHub Actions. No deployment
or remote branch change happens merely by building this directory locally.
