# SpecRA: Compressing Agentic LLM Recommenders via Functional Spectral Distillation

Reference implementation of **SpecRA** (*Spectral Reasoning Alignment*), the knowledge-distillation
framework described in the paper. SpecRA compresses an agentic LLM recommender by aligning the
**per-step eigenspectrum of the empirical Neural Tangent Kernel (NTK)** between teacher and student,
using a debiased **Sinkhorn divergence** on spectral measures estimated by **stochastic Lanczos
quadrature (SLQ)**, under a step-aware layer curriculum. SpecRA modifies only training, so distilled
students keep their native inference cost.

All code is in a single self-contained script: **`SpecRA_Agentic.py`**.

> **Anonymized for review.** This release accompanies an ICDM submission; author/affiliation details
> have been removed.

---

## 1. Method at a glance

For each minibatch and each aligned reasoning step `t` and layer `ℓ`:

1. **Functional NTK extraction** — the cosine-normalized NTK Gram matrix `Θ^{(t,ℓ)} ∈ R^{b×b}` is
   accessed *matrix-free* via one VJP + one JVP (`ntk_matvec_raw`, `cosine_ntk_matvec`); the Jacobian
   is never materialized.
2. **Differentiable spectral estimation** — `Θ`'s spectral measure is estimated with SLQ
   (`slq`, `_lanczos`, `measure`): `m` Hutchinson probes × `k` Lanczos steps.
3. **Spectral alignment** — teacher and student measures are compared with a debiased Sinkhorn
   divergence (`sinkhorn_divergence`), summed over `(t, ℓ)` with detached softmax curriculum weights
   and ε-annealing (`SpecRALoss`).

The total objective is `L_total = L_KD + λ · L_spec`, so SpecRA can run standalone or on top of any
baseline KD loss.

---

## 2. Repository layout

`SpecRA_Agentic.py` is organized into numbered sections:

| Section | Contents |
|---|---|
| 0 | Config dataclasses: `SpecRAConfig`, `TrainArgs`, `ModelPair`, and the `PAIRS` registry |
| 1 | Data: `Interaction`, 5-core filter, real-dataset loaders, ID/CS/CD split builders |
| 2 | Tokenization + InteRecAgent multi-step prompt (`T=5`), tool stubs, trajectory builder |
| 3 | `TrajDataset` + collate |
| 4 | `HFAgent` wrapper — returns per-step logits at reasoning-step boundary tokens |
| 5 | **SpecRA core**: NTK matvec, SLQ, Sinkhorn divergence, `SpecRALoss` |
| 6 | Baseline losses: `kl_kd`, `fdd`, `distillm`, `distillm2`, `minillm`, `gkd`, `sad`, `sharp_distill` |
| 7 | Model construction (HF + optional LoRA; tiny GPT-2 for `--dev`) |
| 8 | `HiddenCollector` hook (for FDD / Sharp-Distill) |
| 9 | Training loops: `train_teacher`, `distill_student` |
| 10 | Evaluation: `evaluate_ndcg`, `evaluate_coherence` (GPT-4o-mini judge), `measure_latency_memory` |
| 11 | `paired_bootstrap` (10,000 resamples) |
| 12 | `run_single` — one full `(pair, dataset, method, seed)` run |
| 13 | `aggregate_and_report` — grouped means + significance, prints per-`(pair, split)` tables |
| 14 | `main` / CLI |

---

## 3. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The OPT-IML pair needs `peft`; the GPT-4o-mini reasoning judge (`--with-coherence`) needs `openai`
and the `OPENAI_API_KEY` environment variable. Everything else runs without them.

---

## 4. Data preparation

Place each dataset at the path the loader (`load_real`) expects, under `--data-root` (default `./data`):

| Dataset | Expected file | Canonical source |
|---|---|---|
| MovieLens-25M | `ml-25m/ratings.csv` | GroupLens (grouplens.org/datasets/movielens/25m) |
| Amazon-Books | `amazon-books/reviews.json.gz` | Amazon Reviews dataset (McAuley lab) |
| Yelp | `yelp/yelp_academic_dataset_review.json` | Yelp Open Dataset (yelp.com/dataset) |
| Steam | `steam/steam_reviews.json` | Steam Reviews dataset (McAuley lab) |

All four are filtered with **5-core** (`five_core_filter`). Splits are built per the paper:

- **ID** (`build_id_split`): leave-last-out test, leave-second-last-out validation, all earlier
  prefixes as training.
- **CS / cold-start** (`build_cs_split`): test users with `<5` training interactions or items with
  `<3` training appearances.
- **CD / cross-domain** (`build_cd_splits`): leave-one-dataset-out over `{movielens25m, amazon-books,
  yelp}`; Steam is ID/CS only.

---

## 5. Smoke test (no data, no GPU)

Verify the pipeline runs end-to-end on synthetic data with a tiny GPT-2:

```bash
python SpecRA_Agentic.py --dev --n-epochs 1 --max-train 256 \
    --pairs llama --datasets movielens25m --methods specra kl-kd --seeds 0
```

`--dev` is a **smoke test only**; it does not reproduce paper numbers (see §9).

---

## 6. Running experiments

A single configuration (one teacher–student pair, one dataset, one method, one seed):

```bash
python SpecRA_Agentic.py \
    --pairs llama --datasets movielens25m --methods specra \
    --seeds 0 --data-root ./data --output results_llama_ml_specra.json
```

The full grid run by the default `main()`:

```bash
python SpecRA_Agentic.py \
    --pairs llama qwen opt \
    --datasets movielens25m amazon-books yelp steam \
    --methods sft kl-kd fdd distillm distillm-2 minillm gkd sad sharp-distill \
              specra kl-kd+specra fdd+specra distillm+specra distillm-2+specra \
    --seeds 0 1 2 3 4 \
    --with-coherence \
    --data-root ./data --output results_main.json
```

**Compute note.** Teachers are 6.7B–8B and the script trains on a **single device**; the LLaMA/Qwen
pairs need an 80GB-class GPU at the paper's batch size. The full grid above is `3 × 4 × 14 × 5` runs
and retrains the teacher on every call, so in practice you run subsets (one `--pairs`/`--datasets`
combination at a time) and aggregate the resulting JSON files. Reduce footprint with `--max-train`,
`--n-epochs`, smaller `--methods`, or fewer `--seeds`.

Available CLI flags: `--data-root`, `--output`, `--device`, `--seeds`, `--datasets`, `--pairs`,
`--methods`, `--with-coherence`, `--dev`, `--max-train`, `--n-epochs`.

---

## 7. Key configuration (paper defaults)

Set in `SpecRAConfig` / `TrainArgs` (Section 0 of the script):

| Symbol | Field | Default |
|---|---|---|
| `m` (Hutchinson probes) | `num_probes` | 8 |
| `k` (Lanczos steps) | `lanczos_steps` | 20 |
| `|L*|` (aligned layers) | `num_aligned_layers` | 2 |
| `T̃` (aligned steps) | `num_aligned_steps` | 5 |
| `ε` schedule | `eps_start → eps_end` over `eps_anneal_steps` | 1.0 → 0.05 over 500 |
| `τ` (curriculum temp.) | `curriculum_temperature` | 1.0 |
| `λ` (spectral weight) | `spec_weight` | 1.0 |
| epochs / batch | `n_epochs` / `batch_size` | 5 / 16 |
| LR (full / LoRA) | `lr_full` / `lr_lora` | 1e-4 / 5e-4 (LoRA rank 256) |

The ablation (paper Table IV) and hyperparameter-sensitivity (Table V) studies are driven by these
fields — e.g. set `cosine_normalise=False`, swap the Sinkhorn for the alternatives, or vary
`num_probes` / `lanczos_steps` / `num_aligned_layers` / `num_aligned_steps` / `spec_weight`. They are
**not** exposed as dedicated CLI flags; edit `SpecRAConfig` (or add flags) to run them.

---

## 8. Outputs

`run_single` returns `RunResult` records; `main` writes `--output` JSON containing the full config,
all per-run records, and a grouped `summary`. `aggregate_and_report` also prints, per `(pair, split)`,
a table of NDCG@10 / Recall@10 / coherence / latency / memory with the best method marked and a
paired-bootstrap significance flag (`p < 0.05`) against the next-best.

---

## 9. Scope of this reference implementation

This file is a faithful implementation of the **SpecRA training objective and the baseline losses**,
plus a lightweight harness to exercise them. Several pieces are deliberately simplified, so please
read this section before treating the script as a one-command reproduction of the reported tables:

- **Evaluation is a proxy.** `evaluate_ndcg` ranks the **first token of the target item string**
  against `n_neg=99` randomly sampled vocabulary tokens. It does **not** re-rank the `K=20` retrieved
  catalog pool described in the paper, so its absolute NDCG@10 values are a relative sanity check and
  will **not** match the paper's Table II. Reproducing the reported metrics requires wiring in the
  full first-stage retriever and a candidate pool of real catalog items.
- **Tools are deterministic stubs.** `tool_retrieve` (shuffle-and-take-k), `tool_rank` (identity), and
  `tool_explain` (template) define the *shape* of the teacher-forced trajectory text only; per the
  paper's teacher-forcing design, tool outputs do not receive gradients, but these stubs are not the
  production InteRecAgent tools.
- **Reasoning text is templated.** `build_trajectory` emits fixed reasoning/action strings per step,
  so training trajectories are structurally identical across queries (modulo item IDs).
- **Per-step targets use a single proxy token.** `_target_token_ids` uses the target item's first
  token at every step boundary, rather than a distinct per-step objective.
- **Single-device driver.** The teacher is retrained on each `run_single` call; a fuller pipeline
  would train each teacher once and reuse it, and parallelize across GPUs.
- **`--dev`** uses synthetic interactions and a tiny GPT-2 and is for smoke-testing only.

If you maintain a fuller internal pipeline that produced the paper's numbers (real catalog re-ranking,
production prompting, shared teachers), that pipeline — not this single file — should back the
reproducibility claim, and this script is best described in the paper/appendix as a *reference
implementation of the method*.

---
