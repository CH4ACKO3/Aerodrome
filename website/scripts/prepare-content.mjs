import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
const root = path.resolve(import.meta.dirname, '../..');
const source = path.join(root, 'ngc/docs');
const target = path.join(root, 'website/src/content/docs/reference');
await mkdir(target, {recursive:true});
await mkdir(path.join(root, 'website/public'), {recursive:true});
for (const name of await readdir(source)) {
  if (!name.endsWith('.md')) continue;
  let content = await readFile(path.join(source,name), 'utf8');
  const title = content.match(/^#\s+(.+)/m)?.[1] ?? name.slice(0,-3);
  content = content.replace(/^#\s+.+\r?\n/, '');
  content = content.replace(/\]\(([^)\s]+)\)/g, (whole, href) => {
    if (/^(https?:|#|mailto:)/.test(href)) return whole;
    if (/^[\w-]+\.md(?:#.*)?$/.test(href)) return `](/Aerodrome/reference/${href.replace('.md','/').replace('/#','#')})`;
    if (href.startsWith('../') || href.startsWith('src/') || href.startsWith('examples/')) {
      const repoPath = path.posix.normalize((href.startsWith('../') ? 'ngc/docs/' : 'ngc/')+href);
      return `](https://github.com/CH4ACKO3/Aerodrome/blob/docs/${repoPath})`;
    }
    return whole;
  });
  await writeFile(path.join(target,name), `---\ntitle: ${JSON.stringify(title)}\n---\n\n${content}`);
}
const pyproject = await readFile(path.join(root,'ngc/pyproject.toml'),'utf8');
const version = pyproject.match(/^version\s*=\s*"([^"]+)"/m)[1];
await writeFile(path.join(root,'website/public/teaching-manifest.json'),JSON.stringify({api_version:1,core_version:version,curriculum_version:'2026.09',experiments:['rigid-velocity-v1']},null,2));
