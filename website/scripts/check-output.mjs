import { readFile, readdir, stat } from 'node:fs/promises';
import path from 'node:path';
const root = path.resolve(import.meta.dirname, '../dist');
let pages = 0;
const errors = [];
async function walk(directory) {
  for (const entry of await readdir(directory, {withFileTypes:true})) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) { await walk(file); continue; }
    if (!entry.name.endsWith('.html')) continue;
    pages++;
    const html = await readFile(file,'utf8');
    if (html.includes('class="katex-error"')) errors.push(`${file}: invalid formula`);
    for (const [,href] of html.matchAll(/href="([^"]+)"/g)) {
      if (!href.startsWith('/Aerodrome/')) continue;
      let target = path.join(root,decodeURIComponent(href.split(/[?#]/)[0].slice('/Aerodrome/'.length)));
      try {
        if ((await stat(target)).isDirectory()) target = path.join(target,'index.html');
        if (!(await stat(target)).isFile()) throw new Error();
      } catch { errors.push(`${path.relative(root,file)}: missing ${href}`); }
    }
  }
}
await walk(root);
if (errors.length) { console.error([...new Set(errors)].join('\n')); process.exitCode=1; }
else console.log(`Verified internal paths and rendered math in ${pages} pages.`);
