# Presentation assets

This directory is reserved for the small set of high-value visual assets used by ScaleRL's README, documentation site, portfolio presentation, and final release.

See [../PRESENTATION_PLAN.md](../PRESENTATION_PLAN.md) for the full plan and exact user-input instructions.

## Expected assets

```text
scalerl-logo.svg
scalerl-logo.png
scalerl-banner.png
scalerl-social-preview.png

scenario-lab-demo.gif
scenario-lab-poster.png

architecture-beginner.svg
architecture-beginner.png
architecture-research.svg
architecture-research.png
experiment-lifecycle.svg
experiment-lifecycle.png

results-cost-vs-sla.png       # after #46, if selected
results-by-workload.png       # after #46, if selected
results-robustness.png        # after #46, if selected
results-sim-to-real.png       # after #76, if selected
```

Not every optional result file must exist. Keep only visuals that carry meaningful information.

## Raw capture staging

If a human screen recording is required for #93, the local/raw staging name is:

```text
docs/assets/raw/scenario-lab-demo.mp4
```

Do not commit a large raw MP4 merely to support the README. The committed deliverables are the optimized GIF and poster frame.

## Rules

- No font files.
- No personal UI/account information.
- No held-out test data in the demo.
- No manually typed scientific result values in final figures.
- No asset may advertise a feature before the feature exists.
- Action labels must match the frozen #79 contract.
- Final result graphics must be generated from canonical frozen outputs.
