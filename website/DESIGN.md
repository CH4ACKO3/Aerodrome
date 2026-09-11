---
name: Probabilistic Aviation
description: A restrained textbook interface with integrated engineering experiments.
rounded:
  code: "4px"
  control: "6px"
  surface: "8px"
  laboratory: "12px"
spacing:
  label-gap: "0.5rem"
  field-gap: "1rem"
  section-padding: "1.5rem"
---

# Design System: Probabilistic Aviation

## Direction

The user requested a visual redesign with a GitBook-like reading experience. Keep the Astro/Starlight navigation and accessibility behavior, with a quiet textbook layout, clear chapter hierarchy and an integrated laboratory surface. Course content remains the seven approved chapter placeholders and appendices A–D. Do not invent subsections during visual work.

## Typography and layout

Self-hosted Inter Variable covers Latin text, with Segoe UI and Microsoft YaHei fallbacks for Chinese. Body text is 1rem with 1.9 line height; headings use weight 650. The reading column has a 48rem maximum, the desktop sidebar is 18rem, and the header is 4.25rem. Below 50rem, the header is 3.75rem and experiment controls stack. Preserve Starlight's mobile navigation and search.

The homepage chapter list uses numbered flat rows separated by fine rules. Appendices form a quieter two-column list, becoming one column on mobile. The sidebar separates chapters from appendices; framework details remain under C.

## Colors and surfaces

`src/styles/editorial.css` owns the overrides after `teaching.css`. Light mode uses white paper, #192b35 ink, #536b76 secondary text, #dce3e7 borders, and #17656d primary controls with white text. Selection uses pale teal. Dark mode uses theme-native backgrounds, #adbdc5 secondary text, #35454d borders, and #9bdddf primary controls with #142d33 text. Use the semantic `--pa-*` and `--sl-*` variables rather than independent component palettes.

Rounded corners use 4px for inline code, 6px for controls, 8px for disclosures and canvases, and 12px for desktop laboratory panels. Panels use a restrained shadow and tinted surface; chapter rows remain flat.

## Experiments

Labels retain units and native validation. Inputs are at least 42px high and action buttons 40px high. The run/load action is teal; secondary actions use bordered theme surfaces. Disabled states, visible focus outlines, status text and reduced-motion handling are required.

Scene order is environment selection, actions, status, canvas and metrics, then import, playback details and advanced resource configuration. The loaded canvas is 440px high on desktop and 320px on mobile; the empty state is 160px with a visible loading invitation. Engines remain lazy and physics remains independent of rendering.

Connection and result messages retain role=status. Reference and local results remain explicitly distinguished. Parameter edits must not imply a recomputed trajectory. Keep quantitative chart axes, solid/dashed line distinction, accessible chart descriptions and downloadable data.

## Review

The primary agent performs visual and functional review directly; the user's no-subagent instruction applies. Review desktop/mobile layouts and both themes when changing shared styling. This design is recorded in PRODUCT.md and website/.impeccable/design.json; screenshots are local review evidence, not source assets.

## Navigation correction and bilingual title

Use 概率飞行工程 as the primary Chinese title, with Probabilistic Aviation retained as the secondary English name. The shared SiteTitle component stacks both names within the existing header height. Sidebar links and group labels use the same 14px font; nested group-label spans must inherit it instead of Starlight's large style. Preserve Starlight's header-dependent fixed TOC offset: never add unconditional padding to .right-sidebar. Compact TOC labels have an explicit 1.6 line height for Chinese glyphs.
