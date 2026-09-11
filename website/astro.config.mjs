import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://ch4acko3.github.io', base: '/Aerodrome', output: 'static',
  markdown: { processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }) },
  integrations: [starlight({
    title: '概率飞行工程 · Probabilistic Aviation', components: { SiteTitle: './src/components/SiteTitle.astro' }, defaultLocale: 'root',
    locales: {root: {label: '简体中文', lang: 'zh-CN'}},
    description: '从航空力学与统计方法，到可以运行的控制实验。',
    customCss: ['./src/styles/fonts.css', 'katex/dist/katex.min.css', './src/styles/teaching.css', './src/styles/editorial.css'],
    social: [{icon: 'github', label: 'GitHub', href: 'https://github.com/CH4ACKO3/Aerodrome'}],
    sidebar: [
      {label: '手册目录', slug: ''},
      {label: '1.数学工具基础', items: [
        {label: '本章目录', slug: 'chapters/01-mathematical-tools'},
        'chapters/01-mathematical-tools/01-linear-algebra',
        'chapters/01-mathematical-tools/02-probability',
        'chapters/01-mathematical-tools/03-statistics',
        'chapters/01-mathematical-tools/04-decision-theory',
        'chapters/01-mathematical-tools/05-optimization',
        'chapters/01-mathematical-tools/06-dynamic-programming'
      ]},
      'chapters/02-statistical-learning',
      'chapters/03-aircraft-control-models',
      'chapters/04-navigation',
      'chapters/05-control',
      'chapters/06-guidance-and-policy',
      'chapters/07-integrated-experiments',
      'appendices/a-notation',
      'appendices/b-engineering-programming',
      {label: 'C.Aerodrome 框架', collapsed: true, items: [
        {label: '附录说明', slug: 'appendices/c-aerodrome'},
        {label: '在线与本地使用', slug: 'getting-started'},
        {label: '现有程序示例', collapsed: true, items: ['experiments/velocity-control', 'experiments/scene-viewer']},
        {label: '工程文档', collapsed: true, items: [{autogenerate: {directory: 'reference'}}]}
      ]},
      'appendices/d-references'
    ]
  })]
});
