---
name: Probabilistic Aeronautics
description: Chinese aerospace teaching documentation in the native Starlight reading interface.
rounded:
  control: "0.25rem"
spacing:
  label-gap: "0.35rem"
  input-padding: "0.5rem"
  action-gap: "0.75rem"
  field-gap: "1rem"
  section-padding: "1.5rem"
components:
  experiment-button:
    rounded: "{rounded.control}"
    padding: "0.5rem 0.8rem"
  experiment-input:
    rounded: "{rounded.control}"
    padding: "{spacing.input-padding}"
    width: "100%"
---

# Design System: Probabilistic Aeronautics

## Overview

Preserve the native Astro/Starlight documentation interface. Chinese aerospace teaching content, English technical terms, visible equations and reproducible experiment data determine the reading experience. No separate creative metaphor or custom brand has been approved.

**Key Characteristics:**

- Familiar documentation navigation and reading hierarchy.
- Theme-aware controls integrated into the lesson.
- Explicit distinction between precomputed references and local simulation results.

This records the implemented interface in `astro.config.mjs`, `src/styles/teaching.css` and `src/components/Experiment.astro`. Review evidence is stored in the repository's `.impeccable/review/desktop.png` and `mobile.png`; final reviewer disposition is ship after the quantitative chart-axis correction.

## Colors

Use Starlight's active theme as the color source of truth; this site does not define a separate palette. The experiment uses `--sl-color-text-accent` for the trajectory and keyboard focus, `--sl-color-text` for text and chart axes, `--sl-color-bg` for input backgrounds, `--sl-color-gray-6` for buttons, `--sl-color-gray-3` for control borders and `--sl-color-gray-5` for section dividers. Preserve these semantic bindings across light and dark themes.

## Typography

The reading interface inherits Starlight typography and its system font fallbacks. No custom display font or font-size scale is introduced by the teaching stylesheet. Equations use the configured KaTeX stylesheet; structured result data appears in a native preformatted block. Keep units visible in field labels and chart axes.

## Layout

Keep Starlight's reading column, sidebar and responsive navigation. The experiment is an inline lesson section separated by block-axis borders. Fields use an auto-fitting grid with a minimum column width of 180px and the field gap above; labels stack over controls. Action buttons wrap with the action gap above. The SVG scales to its container width while preserving its 640 × 260 view box. Result JSON scrolls within a maximum height of 24rem. No custom viewport breakpoint is defined by the teaching stylesheet.

## Elevation & Depth

The experiment is flat: borders and theme surfaces provide separation, with no custom shadows or motion. Keep the incumbent Starlight chrome and its own treatments.

## Shapes

Inputs and buttons use the shared control radius and thin solid borders. The experiment container uses horizontal dividers rather than a raised card treatment.

## Components

The experiment has labelled numeric inputs grouped by a fieldset and legend, with required values and native min/max/step validation. Buttons use the shared neutral treatment; disabled buttons have half opacity and a not-allowed cursor. Keyboard focus has a 2px accent outline with a 3px offset. There is no custom hover animation.

Connection and result messages use `role="status"`. Local execution stays disabled until the same-origin service confirms matching API, core and curriculum versions and the required experiment capability. Loading, offline, version mismatch, computation and error states are stated in text. Errors retain the previous result and do not automatically resubmit. Parameter edits explicitly warn that the graph still shows the previous result.

The chart has an accessible title and description. Solid velocity and dashed target lines distinguish meaning independently of color; quantitative axis labels include velocity bounds and elapsed time. A native details/summary exposes reproducible parameters and sampled data, and a JSON download exposes the full result. Reference trajectories are visibly labelled as precomputed. Keep native Starlight navigation and mobile behavior.

## Do's and Don'ts

- **Do** inherit Starlight theme and typography for new teaching content.
- **Do** preserve visible units, keyboard focus, status text and downloadable numerical results.
- **Do** label precomputed and local results accurately and retain readable content when disconnected.
- **Don't** imply that parameter edits recompute a reference chart.
- **Don't** introduce a separate brand palette or speculative component system without an approved direction.
