# ScaleRL Presentation & Open-Source Polish Plan

This document is the **permanent implementation brief** for ScaleRL's final public presentation.

It exists so the presentation direction does not depend on chat history.

Tracking umbrella: #91  
Learning/framework umbrella: #85

## Final identity

ScaleRL should present itself as:

> **A visual, reproducible lab for learning, experimenting with, researching, and extending cloud autoscaling.**

Public progression:

```text
Learn → Experiment → Research → Extend
```

This identity must not be changed casually. It is the agreed end-state for v1.

## README information hierarchy

The final README should use this order:

```text
1. Hero banner / brand
2. Real CI/release/license badges
3. One-paragraph explanation
4. Scenario Lab demo GIF
5. Choose your path: Learn / Experiment / Research / Extend
6. Quick Start
7. How it works — beginner view
8. How it works — research architecture
9. Controllers
10. Scenario Lab
11. Research methodology overview
12. Bring Your Own Controller
13. Final results / evidence
14. Documentation map
15. Compact project structure
16. Development / contributing
17. Citation / license / release
```

The README is a **front door**, not the full research paper.

Deep methodology belongs in `docs/`.

## Required presentation assets

Final expected asset set:

```text
docs/assets/
├── scalerl-logo.svg
├── scalerl-logo.png
├── scalerl-banner.png
├── scalerl-social-preview.png
├── scenario-lab-demo.gif
├── scenario-lab-poster.png
├── architecture-beginner.svg
├── architecture-beginner.png
├── architecture-research.svg
├── architecture-research.png
├── experiment-lifecycle.svg
├── experiment-lifecycle.png
├── results-cost-vs-sla.png          # after #46 if useful
├── results-by-workload.png          # after #46 if useful
├── results-robustness.png           # after #46 if useful
└── results-sim-to-real.png          # after #76 if useful
```

Do not create decorative assets merely to fill this list. Keep only high-information visuals.

## Hero banner

Target message:

```text
ScaleRL
A visual lab for cloud autoscaling & reinforcement learning

Learn • Experiment • Research • Extend
```

The visual identity should combine:
- cloud/infrastructure;
- changing traffic;
- scaling capacity;
- sequential decision/control.

It may borrow the conceptual language of City View without looking like a children's game.

## Demo GIF

The primary demo must visually tell this causal story:

```text
traffic increases
      ↓
queue / latency react
      ↓
controller requests capacity
      ↓
replicas remain pending during startup delay
      ↓
pending replicas become active
      ↓
queue / latency recover
      ↓
cost is now higher
```

Use development/validation-style synthetic traffic only, never sealed held-out data.

The visible action wording must match final #79 semantics.

## Documentation architecture

Canonical source of truth:

```text
README.md       public landing page
docs/           canonical versioned docs
MkDocs          rendering/navigation
GitHub Pages    hosted docs
Wiki            optional informal community notes only
```

Target navigation:

```text
Home
Learn
Experiment
Research
Extend
Architecture
Reference
Contributing
```

GitHub Wiki must never be the only home of critical methodology or API documentation.

## Architecture storytelling

Maintain two architecture explanations.

### Beginner

```text
🚗 Traffic arrives
       ↓
👥 Requests wait
       ↓
☕ Replicas serve them
       ↑
       │
🧑‍💼 Controller decides
   ├─ Threshold
   ├─ Predictive
   └─ RL
```

Pair beginner language with exact technical terms.

### Research

```text
Synthetic / Azure traces
        ↓
Autoscaling simulator
        ↓
Gymnasium observation/action contract
        ↓
Controller contract
        ↓
Shared evaluation
   ├─ MLflow
   ├─ Optuna
   └─ Scenario Lab / Results
```

Add third-party controller only after #88/#89.
Add live Knative only after #73–#75.

## Architecture Decision Records

Final docs must explain the reasoning behind:
- why RL;
- why simulator first;
- controller contract;
- final #79 action semantics;
- #78 model-selection change;
- train/validation/test isolation;
- multi-seed evaluation;
- strong predictive baselines;
- robustness model;
- MLflow + Optuna responsibilities;
- sim-to-real with Knative.

See #96.

## Final results presentation

After #46:
- lead with service quality + normalized cost;
- preserve workload-specific evidence;
- show training-seed dispersion for learned policies;
- show robustness separately where useful;
- reward remains secondary;
- do not manufacture an overall winner if the evidence is mixed.

All figures must be generated from canonical frozen outputs.

## USER INPUT REQUIRED — permanent checklist

These are the only manual inputs currently expected from the user.

### 1. Brand-direction choice

**What:** choose one generated ScaleRL visual direction.

**How:** when #92 produces 2–3 concrete variants, reply with `1`, `2`, or `3`, plus at most one requested adjustment.

**File you provide:** none.

**When:** during #92.

The user is **not** expected to design the logo manually.

---

### 2. Raw Scenario Lab screen recording

**What:** one raw 45–60 second recording of the final Scenario Lab.

**How:**
1. Wait until the implementation issue gives the exact scenario/controller settings.
2. Open only the Scenario Lab browser/app window.
3. Prefer 1920×1080 or 1440×900.
4. Hide personal tabs, bookmarks, email/account UI, and notifications.
5. No microphone or narration is required.
6. Start recording **before** pressing Play.
7. Let the run show:
   - traffic increasing;
   - queue/latency response;
   - controller scaling decision;
   - pending replicas during startup delay;
   - pending → active transition;
   - recovery;
   - increased infrastructure cost.
8. Stop after recovery and cost are both visible.

**Filename:**

```text
scenario-lab-demo.mp4
```

**How to hand it off:** upload the MP4 in chat when requested. If working in the repository locally, the staging location is:

```text
docs/assets/raw/scenario-lab-demo.mp4
```

The user does **not** need to edit it, make the GIF, crop it, select frames, or add captions.

**When:** after final #79 action wording is stable; preferably after #80/#87 if those materially change the UI.

---

### 3. GitHub social preview upload

**What:** upload the already-created social preview.

**File:**

```text
docs/assets/scalerl-social-preview.png
```

**How:**

```text
GitHub repository
→ Settings
→ General
→ Social preview
→ Edit
→ Upload docs/assets/scalerl-social-preview.png
```

**When:** after #92 is merged.

---

### 4. GitHub Pages setting, only if required

**What:** set Pages to deploy through GitHub Actions.

**How:**

```text
GitHub repository
→ Settings
→ Pages
→ Build and deployment
→ Source: GitHub Actions
```

**File:** none.

**When:** after #95's Pages workflow is merged and green.

If GitHub already configures this automatically, no action is needed.

---

### 5. Citation author name

**What:** confirm the exact public spelling of the author name for `CITATION.cff`.

**How:** reply with the exact public name.

**File:** no upload; the implementation writes:

```text
CITATION.cff
```

**When:** during #98 / before v1.0.0.

---

### 6. ORCID, optional

**What:** provide an ORCID only if it should be public.

**How:** send the full ORCID URL or explicitly say `omit ORCID`.

**File:** `CITATION.cff`.

**When:** together with the citation-name confirmation.

---

### 7. GitHub Discussions, optional

Only if we decide to actively use Discussions for Q&A, community controllers, research ideas, and showcases.

**How:**

```text
GitHub repository
→ Settings
→ General
→ Features
→ Discussions
```

Do not enable it merely for appearance.

---

### 8. Manual release tag, fallback only

Only if available repository tooling cannot create the final tag/release.

The implementation must first provide a **verified final main SHA**.

Then:

```bash
git tag -a v1.0.0 <VERIFIED_SHA> -m "ScaleRL v1.0.0"
git push origin v1.0.0
```

Do not do this early.

## Things the user should NOT have to provide

The implementation should handle:
- README copy;
- architecture diagrams;
- plot generation;
- poster extraction;
- GIF conversion/compression;
- captions;
- MkDocs configuration;
- GitHub issue/PR templates;
- release notes;
- result tables;
- final architecture descriptions.

## Timing

Scientific critical path remains:

```text
#79
 ↓
#80 + #81
 ↓
#20
 ↓
#72
 ↓
#46
```

Presentation work can be prepared in parallel, but public claims/results must wait until their underlying features are complete.

## Tracking

- #91 — presentation umbrella
- #92 — brand kit / hero
- #93 — Scenario Lab demo
- #94 — final README
- #95 — MkDocs / Pages
- #96 — Architecture Decision Records
- #97 — architecture and result visuals
- #98 — repository/community/citation polish
- #99 — v1.0.0 release

Related:
- #85 — Learn / Experiment / Research / Extend umbrella
- #86 — learning path
- #87 — Guided Learn mode
- #88 — stable BYO Controller API
- #89 — researcher extension kit / CLI
