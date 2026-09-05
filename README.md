# URSC — Uncertainty-Aware Re-sampling with Spatial-Consistency Calibration

Reference implementation of the **URSC** framework from the paper (ICASSP-2027):

> **URSC: Uncertainty-Aware Re-sampling with Spatial-Consistency Calibration for
> Multimodal Feature Fusion**

URSC couples **uncertainty-aware 2D refinement** with **uncertainty-modulated
re-sampling** and **spatial-consistency filtering**. It targets the
language-to-3D interface of a modular perception pipeline: a language-conditioned
2D model locates the target, back-projects the region into a point-cloud feature
field, and forwards it to a **frozen** downstream module (here a 6-DoF grasp
generator).

---

## What this repository is

A **reference structure** for the URSC method. It defines the two-stage control
flow and the intermediate representations, and it is *evaluation-oriented*: the
frozen inputs, protocols, and paired-bootstrap statistics that the paper reports
are described and mirrored here.

It is **not** a training repository. It contains **no** model weights, **no**
datasets, and **no** third-party implementations (HiFi-CS, SAM-B, FGC-GraspNet,
OVGNet / GraspNet submodule) — those are injected through the lightweight
interfaces in `ursc/pipeline.py`.

---

## The two interface-level failures URSC addresses

1. **Boundary errors, missing depth, or clutter truncate the 3D feature field.**
   The downstream module then has no valid response → C_b = ∅.
2. **Anchor-only consistency checks do not ensure whole query-support
   consistency.** A query whose anchor lies on the target can still place its
   support on a neighbour.

### Stage I — Language-conditioned target understanding

`ursc/stage1.py` refines the language-conditioned mask with SAM and retains the
cross-model disagreement as the feature-uncertainty estimate:

```
s_k^sel = λ_sel · IoU(m_k, m_h) + λ_conf · c_k
k*      = argmax_k s_k^sel
m_hat   = m_{k*}
U       = 1 − IoU(m_h, m_hat)      # cross-model feature-uncertainty
```

**U never triggers re-sampling by itself.** It only adjusts the re-sampled
point-cloud composition *after* an empty nominal-branch response.

### Stage II — Disagreement-modulated re-sampling & spatial-consistency filtering

`ursc/stage2.py` runs the nominal branch first, re-samples **only on empty
responses**, and filters survivors by query-support consistency:

```
r_ctx(U)    = clip(β0 + β1·U, r_ctx^min, r_ctx^max)
r_bdry      = ρ
r_core      = 1 − r_bdry − r_ctx(U)
```

Then each candidate's **complete** support region (fingers + closing + swept
approach volume) is transformed by (R_i, t_i), projected into m_hat, and scored:

```
p_i = (1/|V_i|) · Σ_{x∈V_i} m_hat(Π_K(x))
S_i = (1 − λ) · s_i + λ · p_i
```

Sorting by S_i yields the calibrated, ranked set C with g* = g_1. Identical
filtering in both branches ensures a non-empty nominal output is never replaced.

---

## Structure

```
ursc_opensource/
├── ursc/
│   ├── config.py      # URSCConfig — paper hyperparameters + validation
│   ├── stage1.py      # Prompt-constrained mask refinement + cross-model
│   │                  #   feature-uncertainty U (Stage I)
│   ├── stage2.py      # Disagreement-modulated re-sampling + spatial-
│   │                  #   consistency calibration (Stage II)
│   └── pipeline.py    # nominal-first control flow on a frozen backend
├── evaluation/
│   ├── protocol.py    # RoboRefIt / OCID-VLG / GraspNet protocol definitions
│   └── bootstrap_ci.py# paired bootstrap CIs (20,000 reps, 95%)
├── configs/
│   ├── ursc_default.json       # Table 1 configuration
│   └── protocol_testb300.json  # frozen testB-300 + other benches
├── examples/
│   └── demo_pipeline.py        # self-contained mechanism demo
├── docs/
│   └── protocol.md             # evaluation protocol description
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and NumPy/SciPy. The frozen downstream modules are external
and injected through the protocols in `ursc/pipeline.py`.

## Usage

```python
from ursc.config import URSCConfig
from ursc.pipeline import URSC

cfg = URSCConfig().validated()
model = URSC(cfg, front_end=..., sam=..., backend=...)
out = model.forward(image, depth, intrinsics, instruction)
```

`front_end`, `sam`, and `backend` are user-provided objects implementing the
interfaces in `ursc/pipeline.py`; the reference implementation defines the
structure around them.

## Run the mechanism demo

```bash
python examples/demo_pipeline.py
```

The demo shows the empty-triggered re-sampling, the U-modulated context ratio,
and the spatial-consistency calibration over the complete query support.

## Evaluation protocol

See [`docs/protocol.md`](docs/protocol.md). The paper evaluates on:

- **RoboRefIt** — complete testA/B (Stage I REC/RES) and a frozen testB-300 list.
- **OCID-VLG** — novel (unseen-class) split.
- **GraspNet-1Billion** — official protocol over 23,040 RealSense frames.

FGC-GraspNet (frozen) is the primary downstream module and OVGNet's frozen
GraspNet submodule is the cross-module check. `testB-300` fixes the
downstream-module and query inputs, isolating the interface change from
downstream capacity.

## License

MIT License. See [`LICENSE`](LICENSE).

## Citation

If you use this architecture in your work, please cite the paper (to be
completed after publication).
