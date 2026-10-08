import { cp, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
const root = path.resolve(import.meta.dirname, '../..');
for(const name of ['Assets','ThirdParty','Widgets','Workers']) {
  await cp(path.join(root,'website/node_modules/cesium/Build/Cesium',name),path.join(root,'website/public/vendor/cesium',name),{recursive:true});
}
const source = path.join(root, 'ngc/docs');
const target = path.join(root, 'website/src/content/docs/reference');
await mkdir(target, {recursive:true});
await mkdir(path.join(root, 'website/public'), {recursive:true});
// Offer the same short examples used by the chapter figures as local downloads.
await mkdir(path.join(root, 'website/public/downloads'), {recursive:true});
for (const example of ['math_tools.py', 'finite_horizon_dp.py', 'statistical_learning.py', 'learning_control.py', 'mlp_backprop.py', 'dp_control.py', 'policy_learning.py', 'bayesian_update.py', 'symbolic_state_conversion.py']) {
  await cp(path.join(root, 'ngc/examples', example), path.join(root, 'website/public/downloads', example));
}
// The viewer fetches real generator files only when readers request source.
// Publish the shared script and its dependencies together; never hand-copy snippets.
await mkdir(path.join(root, 'website/public/figures/sources'), { recursive: true });
for (const [bundle, script, examples] of [
  ['math-tools', 'plot-math-intuitions.py', ['math_tools.py']],
  ['math-foundations', 'plot-math-foundations.py', ['math_tools.py', 'finite_horizon_dp.py']],
  ['statistical-learning', 'plot-statistical-learning.py', ['statistical_learning.py']],
  ['learning-control', 'plot-learning-control.py', ['learning_control.py']],
  ['neural-computation', 'plot-neural-computation.py', ['mlp_backprop.py']],
  ['reinforcement-learning', 'plot-reinforcement-learning.py', ['dp_control.py', 'policy_learning.py']],
  ['bayesian-update', 'plot-bayesian-update.py', ['bayesian_update.py']],
]) {
  // Put the current figure's generator first, followed by the files it needs.
  const names = [`website/scripts/${script}`, 'website/scripts/teaching.mplstyle',
    ...examples.map(name => `ngc/examples/${name}`)];
  await writeFile(path.join(root, `website/public/figures/sources/${bundle}.json`), JSON.stringify({
    note: `保留文件所示目录结构；在 website 目录运行 python scripts/${script} --font /path/to/chinese-font.ttf。需要 NumPy、Matplotlib 和中文字体；脚本输出 SVG 与 PNG。`,
    files: await Promise.all(names.map(async name => ({ name, content: await readFile(path.join(root, name), 'utf8') }))),
  }));
}
// Framework pages and project READMEs are the canonical sources. Import both
// so chapter examples never need a second, drifting copy of their instructions.
async function importReference(sourcePath, name) {
  let content = await readFile(sourcePath, 'utf8');
  const title = content.match(/^#\s+(.+)/m)?.[1] ?? name.slice(0,-3);
  content = content.replace(/^#\s+.+\r?\n/, '');
  content = content.replace(/\]\(([^)\s]+)\)/g, (whole, href) => {
    if (/^(https?:|\/|#|mailto:)/.test(href)) return whole;
    const [file, anchor] = href.split('#');
    const relative = path.relative(root, path.resolve(path.dirname(sourcePath), file)).split(path.sep).join('/');
    if (/^ngc\/docs\/[\w-]+\.md$/.test(relative)) {
      return `](/Aerodrome/reference/${path.basename(file, '.md')}/${anchor ? '#' + anchor : ''})`;
    }
    if (/^ngc\/projects\/[\w-]+\/README\.md$/.test(relative)) {
      const project = path.basename(path.dirname(relative)).replaceAll('_', '-');
      return `](/Aerodrome/reference/project-${project}/${anchor ? '#' + anchor : ''})`;
    }
    if (href.startsWith('../') || href.startsWith('src/') || href.startsWith('examples/')) {
      const repoPath = (href.startsWith('src/') || href.startsWith('examples/') ? 'ngc/' + href : relative + (anchor ? '#' + anchor : ''));
      return `](https://github.com/CH4ACKO3/Aerodrome/blob/docs/${repoPath})`;
    }
    return whole;
  });
  await writeFile(path.join(target,name), `---\ntitle: ${JSON.stringify(title)}\n---\n\n${content}`);
}
for (const name of await readdir(source)) {
  if (name.endsWith('.md')) await importReference(path.join(source, name), name);
}
for (const project of ['obstacle_navigation', 'dynamic_decision', 'variable_team', 'task_scheduling', 'adaptive_control']) {
  await importReference(path.join(root, 'ngc/projects', project, 'README.md'), `project-${project.replaceAll('_', '-')}.md`);
}
const pyproject = await readFile(path.join(root,'ngc/pyproject.toml'),'utf8');
const version = pyproject.match(/^version\s*=\s*"([^"]+)"/m)[1];
await writeFile(path.join(root,'website/public/teaching-manifest.json'),JSON.stringify({api_version:1,core_version:version,curriculum_version:'2026.09',experiments:['rigid-velocity-v1']},null,2));
