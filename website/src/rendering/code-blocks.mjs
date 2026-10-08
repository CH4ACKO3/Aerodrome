/** Give existing examples a useful name; authors can supply a filename or title
 * in the fence. Keep the code itself untouched so copying preserves the example. */
export function remarkCodeTitles() {
  return (tree) => {
    let heading = '代码示例';
    let example = 0;
    const text = (node) => node.value ?? (node.children ?? []).map(text).join('');
    function walk(node) {
      if (node.type === 'heading') {
        heading = text(node).replace(/^\d+(?:\.\d+)*[.、]?\s*/, '');
        example = 0;
      }
      if (node.type === 'code') {
        example += 1;
        node.lang ||= 'text';
        if (!/\btitle\s*=/.test(node.meta ?? '')) {
          const title = `${heading}${example > 1 ? ` · 示例 ${example}` : ''}`.replaceAll('"', '”');
          node.meta = `${node.meta ?? ''} title="${title}"`.trim();
        }
      }
      node.children?.forEach(walk);
    }
    walk(tree);
  };
}

/** A real text label remains visible and accessible in both site themes. */
export const codeLanguageLabels = {
  name: 'Code language labels',
  hooks: {
    postprocessRenderedBlock: ({ codeBlock, renderData }) => {
      const header = renderData.blockAst.children.find((node) => node.tagName === 'figcaption');
      if (!header) return;
      const names = {
        python: 'Python', py: 'Python', sh: 'Shell', bash: 'Bash', zsh: 'Zsh',
        js: 'JavaScript', javascript: 'JavaScript', ts: 'TypeScript', typescript: 'TypeScript',
        json: 'JSON', yaml: 'YAML', yml: 'YAML', toml: 'TOML', html: 'HTML', css: 'CSS',
        cpp: 'C++', c: 'C', text: '纯文本', plaintext: '纯文本', console: '终端输出',
      };
      header.children.push({
        type: 'element', tagName: 'span', properties: { className: ['code-language'] },
        children: [{ type: 'text', value: names[codeBlock.language] ?? codeBlock.language }],
      });
    },
  },
};
