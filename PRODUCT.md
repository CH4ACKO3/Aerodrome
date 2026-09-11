# Product
<!-- impeccable:product-schema 1 -->

## Platform
web

## Stack
User confirmed Astro + Starlight, built with Node.js. The user approved a visual redesign inspired by GitBook reading patterns: a restrained teal palette, self-hosted Inter typography, chapter-first navigation and integrated experiment panels. Starlight supplies responsive navigation and search.

## Users
Aerospace undergraduates and beginning graduate students learning statistical control methods.

## Product Purpose
Probabilistic Aviation connects explanations, equations and reproducible Aerodrome experiments. The same version has an online static textbook and a local edition connected to a local Python/JAX instance.

## Capabilities and Constraints
Online pages offer full reading and clearly identified precomputed experiments. Local pages can submit bounded experiments, view results and download configurations. A same-origin local server checks API/core versions and reports capabilities. GitHub Pages hosts only static content. No account service or arbitrary Python execution. Core checkpoint b84d0bf is on local main; documentation work is on docs.

## Evidence on Hand
ngc/docs, ngc/examples, ngc/src and cross-platform validation. The F-16 model is a longitudinal teaching subset, not a complete aircraft. Aerodynamics and materials courses are future topics, not completed lessons.

## Brand Commitments
概率飞行工程 · Probabilistic Aviation; Chinese teaching content with English technical terminology. The existing code and legacy documentation remain available.

## Product Principles
Keep equations visible. Separate teaching tasks from physics. Label static and computed results accurately. Preserve readable content when disconnected. Keep Node.js builds independent of Python.

## Editorial terminology

User-facing self-reference is 手册 (manual), including 手册目录 and 在线手册. Do not call this project a 课程 or 教材 in its own interface. Chapter 1 now has six approved section headings: 1.1 线性代数与矩阵计算, 1.2 概率论, 1.3 统计, 1.4 决策论, 1.5 优化, 1.6 动态规划. Their bodies remain placeholders until discussed with the user.
