# Product
<!-- impeccable:product-schema 1 -->

## Platform
web

## Stack
User confirmed Astro + Starlight, built with Node.js. The standard Starlight reading interface is retained; this work adds teaching content and an experiment panel rather than a new visual identity.

## Users
Aerospace undergraduates and beginning graduate students learning statistical control methods.

## Product Purpose
Probabilistic Aeronautics connects explanations, equations and reproducible Aerodrome experiments. The same version has an online static textbook and a local edition connected to a local Python/JAX instance.

## Capabilities and Constraints
Online pages offer full reading and clearly identified precomputed experiments. Local pages can submit bounded experiments, view results and download configurations. A same-origin local server checks API/core versions and reports capabilities. GitHub Pages hosts only static content. No account service or arbitrary Python execution. Core checkpoint b84d0bf is on local main; documentation work is on docs.

## Evidence on Hand
ngc/docs, ngc/examples, ngc/src and cross-platform validation. The F-16 model is a longitudinal teaching subset, not a complete aircraft. Aerodynamics and materials courses are future topics, not completed lessons.

## Brand Commitments
Probabilistic Aeronautics; Chinese teaching content with English technical terminology. The existing code and legacy documentation remain available.

## Product Principles
Keep equations visible. Separate teaching tasks from physics. Label static and computed results accurately. Preserve readable content when disconnected. Keep Node.js builds independent of Python.
