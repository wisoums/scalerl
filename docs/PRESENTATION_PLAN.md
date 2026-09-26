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
6. Try ScaleRL Live — public Streamlit demo
7. Quick Start — full local research stack
8. How it works — beginner view
9. How it works — research architecture
10. Controllers
11. Scenario Lab
12. Research methodology overview
13. Bring Your Own Controller
14. Final results / evidence
15. Documentation map
16. Compact project structure
17. Development / contributing
18. Citation / license / release
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


## Public deployment architecture

ScaleRL v1 has three deliberately different public/runtime surfaces:

```text
PUBLIC INTERACTIVE DEMO
Streamlit Community Cloud
  ├─ Learn
  ├─ Scenario Lab / Playground
  ├─ controller comparison
  └─ built-in/canonical model inference

DOCUMENTATION
GitHub Pages / MkDocs

FULL RESEARCH STACK
local Docker Compose
  ├─ Scenario Lab
  ├─ MLflow
  ├─ Optuna
  ├─ PostgreSQL
  └─ Garage

SIM-TO-REAL
Knative testbed / replay path
```

The hosted Streamlit app is **not** the canonical research backend.

Community mode must:
- reuse the same simulator/controller/dashboard code;
- run simulation and inference only;
- avoid training/tuning jobs;
- avoid dependence on PostgreSQL/Garage/MLflow server;
- avoid arbitrary user-supplied Python/controller execution;
- expose curated synthetic demo workloads;
- load only explicit compatible learned artifacts;
- fail gracefully if learned artifacts are unavailable;
- clearly identify itself as a simulation/inference demo.

The full local Docker stack remains the reproducibility path for experiments, tuning, tracked runs, and artifacts.

Tracking issue: #103.

### README call to action

Once #103 is actually deployed, the README should show near the top:

```text
▶ Try ScaleRL Live
<final *.streamlit.app URL>
```

Then keep the Docker Compose path as:

```text
Run the complete research stack locally
```

Never publish a placeholder public URL.

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

### 8. Streamlit Community Cloud deployment

**What:** authorize/connect `wisoums/scalerl` in Streamlit Community Cloud and create the public app if repository tooling cannot perform the deployment.

**How:**
1. Open Streamlit Community Cloud.
2. Sign in with the GitHub account that can access `wisoums/scalerl`.
3. Create a new app.
4. Repository: `wisoums/scalerl`.
5. Branch: `main`.
6. Main file path: use the exact entrypoint created by #103. Expected direction is `streamlit_app.py`, but do not guess before implementation.
7. Add only environment/secrets explicitly required by #103.
8. Deploy.
9. Send/confirm the final public `*.streamlit.app` URL if it is not visible through repository tooling.

**File you provide:** none.

**When:** after #103's implementation PR is merged and green.

---

### 9. Manual release tag, fallback only

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
- #103 — public Streamlit Community Cloud demo

Related:
- #85 — Learn / Experiment / Research / Extend umbrella
- #86 — learning path
- #87 — Guided Learn mode
- #88 — stable BYO Controller API
- #89 — researcher extension kit / CLI
