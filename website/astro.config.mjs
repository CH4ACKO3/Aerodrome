import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://ch4acko3.github.io', base: '/Aerodrome', output: 'static',
  markdown: { processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }) },
  integrations: [starlight({
    title: 'Probabilistic Aviation', defaultLocale: 'root',
    locales: {root: {label: '简体中文', lang: 'zh-CN'}},
    description: '从航空力学与统计方法，到可以运行的控制实验。',
    customCss: ['katex/dist/katex.min.css', './src/styles/teaching.css'],
    social: [{icon: 'github', label: 'GitHub', href: 'https://github.com/CH4ACKO3/Aerodrome'}],
    sidebar: [
      {label: '开始学习', items: [{label: '课程导读', slug: ''}, {label: '在线与本地', slug: 'getting-started'}, {label: '刚体速度控制实验', slug: 'experiments/velocity-control'}]},
      {label: '模型与算法', items: ['reference/equations', 'reference/rigid-body', 'reference/simulation-tools', 'reference/geography-atmosphere', 'reference/linear-control', 'reference/f16-level-flight']},
      {label: 'World 与实验管线', items: ['reference/world', 'reference/dataflow', 'reference/module-compiler', 'reference/batch', 'reference/configuration', 'reference/gymnasium']},
      {label: '全部工程文档', collapsed: true, items: [{autogenerate: {directory: 'reference'}}]}
    ]
  })]
});
