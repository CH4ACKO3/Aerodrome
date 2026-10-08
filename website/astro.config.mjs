import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkCodeTitles, codeLanguageLabels } from './src/rendering/code-blocks.mjs';

export default defineConfig({
  site: 'https://ch4acko3.github.io', base: '/Aerodrome', output: 'static',
  redirects: {
    "/chapters/03-aircraft-control-models/": "/Aerodrome/chapters/04-aircraft-control-models/",
    "/chapters/04-navigation/": "/Aerodrome/chapters/05-navigation/",
    "/chapters/05-control/": "/Aerodrome/chapters/06-control/",
    "/chapters/06-guidance-and-policy/": "/Aerodrome/chapters/07-guidance-and-policy/",
    "/chapters/07-integrated-experiments/": "/Aerodrome/chapters/08-integrated-experiments/",
    "/chapters/02-statistical-learning/06-parametric-control/": "/Aerodrome/chapters/02-statistical-learning/05-reinforcement-learning/#policy-optimization",
    "/chapters/02-statistical-learning/03-linear-regression/": "/Aerodrome/chapters/02-statistical-learning/02-linear-models/#linear-regression",
    "/chapters/02-statistical-learning/01-discriminant-analysis/": "/Aerodrome/chapters/02-statistical-learning/02-linear-models/#discriminant-analysis",
    "/chapters/02-statistical-learning/02-logistic-regression/": "/Aerodrome/chapters/02-statistical-learning/02-linear-models/#logistic-regression",
    "/chapters/02-statistical-learning/04-generalized-linear-models/": "/Aerodrome/chapters/02-statistical-learning/02-linear-models/#generalized-linear-models",
    "/chapters/02-statistical-learning/05-neural-networks/": "/Aerodrome/chapters/02-statistical-learning/03-deep-neural-networks/#neural-networks",
    "/chapters/02-statistical-learning/06-image-networks/": "/Aerodrome/chapters/02-statistical-learning/03-deep-neural-networks/#image-networks",
    "/chapters/02-statistical-learning/07-sequence-networks/": "/Aerodrome/chapters/02-statistical-learning/03-deep-neural-networks/#sequence-networks",
    "/chapters/02-statistical-learning/11-fewer-labels/": "/Aerodrome/chapters/02-statistical-learning/03-deep-neural-networks/#fewer-labels",
    "/chapters/02-statistical-learning/08-neighbor-methods/": "/Aerodrome/chapters/02-statistical-learning/04-other-methods/#neighbor-methods",
    "/chapters/02-statistical-learning/09-kernel-methods/": "/Aerodrome/chapters/02-statistical-learning/04-other-methods/#kernel-methods",
    "/chapters/02-statistical-learning/10-tree-ensembles/": "/Aerodrome/chapters/02-statistical-learning/04-other-methods/#tree-ensembles",
    "/chapters/02-statistical-learning/12-dimensionality-reduction/": "/Aerodrome/chapters/02-statistical-learning/04-other-methods/#dimensionality-reduction",
    "/chapters/02-statistical-learning/13-clustering/": "/Aerodrome/chapters/02-statistical-learning/04-other-methods/#clustering",
    "/chapters/02-statistical-learning/14-recommender-systems/": "/Aerodrome/chapters/02-statistical-learning/04-other-methods/#recommender-systems",
    "/chapters/02-statistical-learning/15-graph-embeddings/": "/Aerodrome/chapters/02-statistical-learning/04-other-methods/#graph-embeddings"
},
  markdown: { processor: unified({ remarkPlugins: [remarkMath, remarkCodeTitles], rehypePlugins: [rehypeKatex] }) },
  integrations: [starlight({
    title: '概率飞行工程 · Probabilistic Aviation', components: { SiteTitle: './src/components/SiteTitle.astro', Footer: './src/components/Footer.astro' }, defaultLocale: 'root',
    locales: {root: {label: '简体中文', lang: 'zh-CN'}},
    description: '从航空力学与统计方法，到可以运行的控制实验。',
    expressiveCode: {
      themes: ['github-light', 'github-dark'],
      useStarlightUiThemeColors: true,
      plugins: [codeLanguageLabels],
      defaultProps: { frame: 'code' },
      frames: { extractFileNameFromCode: false, removeCommentsWhenCopyingTerminalFrames: false },
      styleOverrides: {
        borderRadius: '6px', codeFontSize: '0.875rem', codeLineHeight: '1.75',
        codePaddingBlock: '1rem', codePaddingInline: '1rem',
        frames: { frameBoxShadowCssValue: 'none' },
      },
    },
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
      {label: '2.统计学习方法', collapsed: true, items: [
        {label: '本章目录', slug: 'chapters/02-statistical-learning'},
        'chapters/02-statistical-learning/01-introduction',
        'chapters/02-statistical-learning/02-linear-models',
        'chapters/02-statistical-learning/03-deep-neural-networks',
        'chapters/02-statistical-learning/04-other-methods',
        'chapters/02-statistical-learning/05-reinforcement-learning'
      ]},
      {label: '3.强化学习', collapsed: true, items: [
        {label: '本章目录', slug: 'chapters/03-reinforcement-learning'},
        'chapters/03-reinforcement-learning/01-control-and-learning',
        'chapters/03-reinforcement-learning/02-bellman-and-lqr',
        'chapters/03-reinforcement-learning/03-parametric-optimization',
        'chapters/03-reinforcement-learning/04-rollout-and-mpc',
        'chapters/03-reinforcement-learning/05-value-learning',
        'chapters/03-reinforcement-learning/06-policy-optimization',
        'chapters/03-reinforcement-learning/07-control-experiments'
      ]},
      {label: '4.飞行器建模与数值仿真', collapsed: true, items: [
        {label: '本章目录', slug: 'chapters/04-aircraft-control-models'},
        'chapters/04-aircraft-control-models/01-simulation-models',
        'chapters/04-aircraft-control-models/02-frames-and-attitude',
        'chapters/04-aircraft-control-models/03-rigid-body-dynamics',
        'chapters/04-aircraft-control-models/04-numerical-simulation',
        'chapters/04-aircraft-control-models/05-environment-and-components',
        'chapters/04-aircraft-control-models/06-fixed-wing-models',
        'chapters/04-aircraft-control-models/07-rotorcraft-models',
        'chapters/04-aircraft-control-models/08-trim-and-linearization',
        'chapters/04-aircraft-control-models/09-validation-and-experiments'
      ]},
      'chapters/05-navigation',
      'chapters/06-control',
      'chapters/07-guidance-and-policy',
      'chapters/08-integrated-experiments',
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
