# FPX: Federation Process Explainability

Operator-facing explanations of what a federated learning (FL) run is doing: why a site was down-weighted, which site drifted, whether this round's model can be trusted.

Most FedXAI work explains a model's predictions, or uses explanations to steer aggregation. FPX explains the federation process itself to the person running it (a study coordinator or hospital ML lead) so they can decide whether to keep, drop, or investigate a site.

This repo contains the simulation testbed, the signal layer, the explanation layer, and the study kit for a formative evaluation.

> **Status:** system and study materials are done. Participant data collection is in progress. Every number in the "Results" section below is a **placeholder** and must be replaced before this repo is cited or the paper is submitted.

---

## What's in the box

| Component | What it does | File |
|---|---|---|
| Taxonomy | Four classes of events an FL operator needs explained, plus client-centric vs. server-centric view | `docs/taxonomy_draft_v1.md` |
| Simulation testbed | 8 simulated hospital sites, Dirichlet non-IID split, small MLP, FedAvg | `notebooks/02_fl_simulation.ipynb` |
| Signal layer | Per-round divergence z-scores, warmup + persistence rule, down-weighting of flagged sites, injected ground-truth fault | `notebooks/03_signal_computation.ipynb` |
| Explanation layer | Template-based translation of signals into plain-language round briefings | `notebooks/04_explanation_interface.ipynb` |
| Study kit | Counterbalanced scenarios, moderator script, response template | `notebooks/05_formative_evaluation.ipynb`, `docs/FPX_Moderator_Reference_Sheet.md` |
| Participant notebook | Self-contained notebook each participant runs on a call | `participant_study.ipynb` |
| Shared utilities | Drive checkpointing, signal and explanation modules | `utils/` |

## How it works

```
data split ──> FL simulation ──> signal layer ──> explanation layer ──> study
(seeded)       (8 sites,         (z-score,         (round briefings)     (raw vs.
               15 rounds,         warmup,                                 raw + briefing)
               1 faulty site)     persistence)
```

**Fault injection.** Site 0's labels are shuffled from round 8 onward, simulating a silent labeling failure at one hospital. This gives a known ground truth to check explanations against.

**Detection rule.** A site's update is compared to the mean update each round (cosine similarity, converted to a z-score across sites). A site is down-weighted only if it falls below the threshold for 2 consecutive rounds, and only after a 5-round warmup. Both guards exist because a single-round threshold produced false positives during early training in testing.

| Parameter | Value |
|---|---|
| Sites | 8 |
| Rounds | 15 |
| Non-IID alpha | 0.5 |
| Warmup rounds | 5 |
| Persistence rounds | 2 |
| z threshold | -1.0 |
| Down-weight factor | 0.05 |
| Fault site / start round | 0 / 8 |

**Determinism.** Data split, partition, model init, and fault RNG are all seeded, so every participant sees the identical underlying run. Only the display format differs by condition.

**Briefing behavior.** Briefings discover which sites look anomalous from the signals alone (no knowledge of which site is faulty). During warmup they say detection is inactive. After warmup they distinguish "just started looking unusual, not yet acted on" from "down-weighted, N consecutive rounds."

## Quickstart

1. Clone the repo and upload the notebooks to Colab (free T4 is enough; the simulation also runs on CPU).
2. Run `notebooks/01_setup_and_checkpoints.ipynb` once to mount Drive, create the `FPX_Project/` folder tree, and write the shared utilities.
3. Run notebooks 02 to 05 in order. Each starts with the same reload cell, so you can resume after a Colab disconnect.

To run the study with a participant, open `participant_study.ipynb` alone. It needs no Drive access.

```python
# Cell 1 of every researcher notebook (resume after a session restart)
from google.colab import drive
drive.mount('/content/drive')

import sys, os
DRIVE_ROOT = "/content/drive/MyDrive/FPX_Project"
sys.path.insert(0, os.path.join(DRIVE_ROOT, "utils"))
import checkpoint_utils as cu

config = cu.load_json("config", "project_config")
cu.project_status()
```

### Drive layout

```
FPX_Project/
├── config/              project_config.json, fault_config.json
├── utils/               checkpoint_utils.py, signal_utils.py, explanation_utils.py
├── data/                data_split.pkl, client_partition.pkl
├── checkpoints/
│   ├── fl_sim/          per-round state
│   ├── fl_sim_robust/   robust run with fault injection
│   ├── signals/         signal_timeline.pkl
│   ├── interface/       generated_explanations.json
│   └── eval/            scenario materials, assignments, collected responses
├── outputs/             plots, protocol text
└── logs/                project_log.txt
```

## Study design

Within-subjects, 8 participants, ML/CS background (proxy for clinical FL operators, see Limitations). Each participant sees four scenarios drawn from the same run:

| ID | Round | What the round shows |
|---|---|---|
| S1_quiet_warmup | 3 | Early training, no detection active |
| S2_early_signal | 8 | First anomalous round for Site 0, not yet acted on |
| S3_fresh_exclusion | 9 | Site 0 down-weighted for the first time |
| S4_sustained_exclusion | 15 | Site 0 down-weighted for 7 consecutive rounds |

Two conditions: `raw_numbers` (signal table with status flags, similar to existing FL monitoring dashboards) and `raw_plus_briefing` (the same table followed by the plain-language briefing). Each participant gets two scenarios in each condition. Assignment rotates through all six balanced 2-2 splits and alternates condition order, so every scenario appears in each condition equally often across participants.

Per scenario: Q1 free-text read of the round, Q2 down-weighted yes/no/unsure (scored against ground truth), Q3 trust 1-7 with reason, Q4 next action, Q5 confidence 1-7. One wrap-up question asks whether participants wanted the raw numbers next to the briefing.

### Response format

Each session exports one JSON file. Example (dummy values):

```json
[
  {
    "participant": 3,
    "scenario": "S3_fresh_exclusion",
    "round": 9,
    "condition": "raw_plus_briefing",
    "q1": "Site 0 is being down-weighted, probably bad data there",
    "q2": "Yes",
    "q3": "6, the system already handled it",
    "q4": "Check Site 0's labeling pipeline",
    "q5": "6"
  }
]
```

## Results (PLACEHOLDER: dummy values, not collected data)

> **Do not cite these numbers.** They show the shape of the output the analysis notebook will produce. Replace the whole section after data collection.

| Metric | raw_numbers | raw_plus_briefing |
|---|---|---|
| Q2 comprehension accuracy | 0.62 | 0.88 |
| Q3 trust (1-7), mean | 4.4 | 5.6 |
| Q5 confidence (1-7), mean | 4.1 | 5.9 |
| Time to correct reading (s), median | 48 | 21 |
| N participants | 8 | 8 |

Per-scenario Q2 accuracy (dummy):

| Scenario | raw_numbers | raw_plus_briefing |
|---|---|---|
| S1_quiet_warmup | 0.75 | 1.00 |
| S2_early_signal | 0.50 | 0.75 |
| S3_fresh_exclusion | 0.75 | 1.00 |
| S4_sustained_exclusion | 0.50 | 0.75 |

## Limitations

- Participants are ML/CS students and researchers, not clinical FL operators. Findings on comprehension of process signals may transfer; findings on domain-specific trust need follow-up with the real population.
- N = 8 supports a formative, largely qualitative claim, not statistical significance.
- One dataset (sklearn breast cancer, tabular) and one synthetic fault type (label shuffle). Other fault types and larger models are untested.
- The detection rule is a simple cosine/z-score heuristic. The contribution is the explanation layer and its evaluation, not the detector.
- Briefings are template-based. The wording is fixed so it can be tested as a stable instrument.

## Related work

- TraceFL: neuron-provenance debugging for FL, arXiv:2312.13632. Technical tool, no operator interface or user study.
- Towards Interpretable Federated Learning, arXiv:2302.13473. Method taxonomy for interpretable FL.
- Federated Explainable AI survey, arXiv:2607.13045. Systematic survey whose taxonomy has no category for operator-facing process explanation.

## Reproducing the pipeline

All randomness is seeded: `DATA_SEED=42`, `PARTITION_SEED=42`, `MODEL_SEED=7`, `FAULT_RNG_SEED=123`. Re-running the participant notebook should reproduce the same per-round signals every time. Check that rounds 3, 8, 9, and 15 match the scenario table above.

## Data and privacy

Raw participant response files and session recordings are not committed. `.gitignore` excludes them. Share only anonymized, aggregated outputs (participant IDs P01-P08, no names).

```
# .gitignore
participant_*_responses.json
recordings/
*.pkl
__pycache__/
```

## Citation

```bibtex
@misc{fpx2026,
  title  = {Federation Process Explainability},
  author = {Mahir},
  year   = {2026},
  note   = {Work in progress}
}
```

## License

Add before publishing (MIT or Apache-2.0 are common for research code).
