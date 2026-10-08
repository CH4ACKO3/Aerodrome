# Probabilistic Aviation

## Course authoring

The main navigation follows eight chapters and appendices A–D, as confirmed by
the author. Chapter landing pages are in `src/content/docs/chapters`, appendix
landing pages in `src/content/docs/appendices`. Their titles are fixed; sections
and prose will be agreed and written incrementally. Do not automatically fill
chapter outlines or promote existing engineering notes into course sections.
Chapter 1.6 introduces the mathematical tools of dynamic programming; chapter 2.5
introduces reinforcement learning through parametric DP and optimization.
Chapter 3 develops control/DP/RL connections, Bellman and LQR, parametric
optimization, rollout and MPC, value learning, policy optimization, and common
control experiments in seven sections. The former chapter 2.6 has been removed.
Existing framework documentation and program examples remain under appendix C
in the sidebar, retaining their original URLs.

Chapter examples link to the relevant Project Zero component section instead
of copying its runnable code. The canonical anchor list is maintained in
`../ngc/docs/project-zero.md`; for example use
`/Aerodrome/reference/project-zero/#coding-styles` for the script/function/object
comparison, or `#control` for control examples. Preserve these explicit anchors
when renaming section titles. Add links when writing actual examples; do not
fill placeholder chapters with an invented outline.

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

实战工程的详细说明以 `ngc/projects/*/README.md` 为唯一来源，由 `prepare-content.mjs` 导入 `/reference/project-<name>/`；组件和项目之间的本地 Markdown 链接在构建时转为站内链接。新增工程时把目录加入该脚本的项目列表，并在 `ngc/docs/teaching-projects.md` 登记。当前这些工程仍通过 Python CLI 运行，文档上线不代表已接网页计算服务。

后续教学内容使用 `$research-paper-writing` 的中文教学适配版（“中文教学文档写作”），安装位置见仓库根目录 `AGENTS.md`。技能保留必要推导，要求中文术语一致、公式对应实际代码、实验结论对应真实记录，并约束重复套话、翻译腔和过度列表化。章节示例仍链接到零号工程的固定小节；技能参考文件 `references/aerodrome.md` 说明源码位置和页面导入方式。

## 全站样式与科学插图

编写页面和新增配图前阅读 [全站渲染与插图规范](DESIGN.md)。共享页面样式在
`src/styles/editorial.css`，Matplotlib 配置在 `scripts/teaching.mplstyle`；规范包含
图号、图注、坐标单位、线型、深色主题和窄屏行为。已有数学图使用
`scripts/plot-math-intuitions.py --font /path/to/chinese-font.ttf` 生成，Node 构建直接使用
已生成的 SVG。保持章节中的图和零号工程的数值例子一致。

正文图片已自动接入悬浮查看器（点击或 Enter 打开，滚轮缩放、拖动、Esc 关闭）。
新教学图片使用 `src/components/TeachingFigure.astro`：通过 `src`、`alt`、`number`、`title`、`caption` 和可选 `source` 声明图片。组件支持富文本图注插槽，悬浮查看器自动同步图注。章节使用 MDX 导入，见 [组件用法](DESIGN.md#6-图注原图与窄屏)。源包的格式和接线见
[规范第 9 节](DESIGN.md#9-图片查看器与源码)。`prepare-content` 直接读取实际源文件生成数学图源码包。
