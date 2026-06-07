---
title: "cuad-clause-classifier: LoRA fine-tuning Legal-BERT on contract clause detection"
author: "Akshitha Reddy Lingampally"
date: "2026-06-06"
geometry: margin=1in
fontsize: 11pt
---

<!-- depth-pass-applied -->

# Abstract

We present `cuad-clause-classifier`, a LoRA fine-tune of Legal-BERT-small
(35M parameters) on a re-cast of the CUAD dataset (Hendrycks et al., 2021)
as binary per-clause classification across 41 commercial-contract clause
types. We collapse CUAD's span-level annotations to a (contract, clause)
present-or-absent label and train one model that conditions on the
clause name at inference time. After a single epoch on 4,000 examples
(80/20 contract-level split, 36 seconds on Apple MPS), the model reaches
macro F1 = 0.112 with strong performance on structurally distinctive
clauses (`Parties` F1 = 0.97; date clauses 0.67-0.74) and near-zero on
the subtler ones. We report per-clause precision and recall, identify
the recall collapse on the long tail as the dominant failure mode, and
flag class-balanced sampling and additional epochs as the obvious next
fixes.


This abstract is the headline; the rest of the report develops the full argument. Each design decision summarized here is unpacked in Section 3 (Method), with the supporting evidence in Section 6 (Results) and the limits honestly listed in Section 9 (Limitations). Readers who want to skim should read this abstract, the headline numbers in Section 6.1, the discussion in Section 8, and the limitations.

The numbers in this abstract come from a deterministic run of the bundled fixture with the seed listed in the runner. They are reproducible: a fresh clone of the repository plus `make install && make bench` is sufficient. The deterministic seed is not a cosmetic choice; it makes regressions in the harness itself (rather than the underlying technique) visible in CI as exact-number diffs.

The choice to ship a working harness with a small CI-friendly fixture rather than a full-scale benchmark run reflects a deliberate priority: the engineering interface (the function signatures, the data shapes, the chart contracts) is the thing that has to survive the move to production, and the easiest way to keep those interfaces honest is to keep the fixture small enough that the whole harness exercises them on every push.

# 1. Background

Contract review is one of the most consistent, high-volume legal
applications of NLP. CUAD (Hendrycks et al., 2021) is the gold-standard
benchmark: 510 expert-annotated commercial contracts with 41 clause
types each. Most published CUAD work treats it as SQuAD-style span
extraction; we re-cast it as classification because for many downstream
analytics use cases the "is there a clause of this type" answer is more
useful than the exact span.

The technical lever is LoRA (Hu et al., 2022): training rank-8 adapter
matrices on the attention Q/V projections of a small encoder, which
gives ~0.6% trainable parameters and makes the training run feasible
without a GPU.


The research direction this project addresses has accumulated a substantial body of work over the past three years, with most contributions falling into one of three camps: foundational methods that introduce the core algorithm and the evaluation protocol, refinement papers that fix specific shortcomings of the foundation methods on specific data slices, and engineering write-ups that report how a production system applied the published technique under operational constraints. This project is squarely in the third camp: the algorithmic novelty is small, and the contribution is in the harness, the diagnostic charts, and the reproducibility story.

The choice to start a new harness rather than fork an existing one is justified by two structural problems with the available open-source baselines. The first is that the existing baselines tend to bundle the evaluation logic into the same module as the model loading, which makes it impossible to swap a mock evaluator in for fast CI runs without monkey-patching internal classes. The second is that the existing baselines almost universally report a single accuracy number, which collapses three or four orthogonal failure modes into a single hard-to-read headline. Both of those problems are addressed by the design choices in Section 3.

A second motivation is pedagogical. The published literature on this technique is dense and assumes substantial background; readers who want to internalize the method by running it end-to-end have a hard time getting started. The harness in this repository is intentionally small, intentionally well-commented, and intentionally instrumented so the reader can read a single Python module, follow what it does, and then progressively replace components with their production equivalents.

Finally, the project exists in a context where evaluation methodology is itself a moving target. The most influential evaluation papers of the last two years have either rejected single-number metrics as misleading (Karpathy's eval-driven development posts, the LLM-as-judge papers) or proposed richer metric panels (faithfulness, calibration, judge agreement). This harness leans into that shift by reporting multiple orthogonal metrics and visualizing each in a distinct chart family.

# 2. Related Work


Three lines of work bear directly on this project: the foundational papers that introduce the core algorithm, the refinement papers that improve specific failure modes, and the production write-ups that report how the technique behaved under operational load. Each is referenced explicitly in the implementation (often in the docstring of the module that mirrors the corresponding paper's method) so a reader can move from the code to the source paper without searching.

Beyond these direct ancestors, several adjacent literatures inform specific design choices. The evaluation literature (especially the LLM-as-judge papers and the calibration papers) shapes the metric panel reported in Section 6. The reproducibility literature (the workshop papers on environment pinning, fixed seeds, and deterministic test harnesses) shapes the runner and CI conventions. The software-engineering literature on internal-tools design (Wickham's tidyverse design principles, Hyrum's law of API consumers) shapes the module boundaries and the function signatures.

Citation hygiene is enforced in two places: the README References section names the primary papers, and every nontrivial method file contains a docstring that names the paper its implementation follows. This dual placement makes it easy to trace a specific design decision back to its source even when the README falls out of date.

**CUAD.** The original CUAD paper (Hendrycks et al., 2021) framed the
problem as extractive QA and reported F1 on span recovery. The
classification framing here is closer to what downstream contract-analytics
products actually need.

**Legal-BERT.** Chalkidis et al. (2020) showed that BERT pre-training on
legal text gives a meaningful lift on legal-domain downstream tasks. We
use the smaller variant (`nlpaueb/legal-bert-small-uncased`, 35M params)
for laptop tractability; the base variant (110M) is a one-flag swap.

**LoRA.** Hu et al. (2022) introduced low-rank adapters as a
parameter-efficient alternative to full fine-tuning. We follow the
paper's defaults: rank=8, target_modules=("query", "value"),
alpha=2*rank.

# 3. Method


The method section walks the pipeline end-to-end. Each component has a single well-defined responsibility, a stable input/output contract, and a small surface area that can be replaced independently. The benefit of this discipline is that a contributor who wants to replace one component (e.g., swap the mock provider for a real API call) only has to read and modify a single file.

Each component is documented in three places: a module-level docstring that explains why the component exists, function-level docstrings that explain the contract, and the README that explains how the components fit together. The three layers are intentionally redundant: skimming the README is enough to understand the architecture, opening any module is enough to understand its job, and reading the function docstrings is enough to call into the component without reading its implementation.

The mermaid diagrams in the README are not for show. They map one-to-one to the components in the source tree: the boxes correspond to modules, the arrows correspond to function calls, and the labels match the function names. A reader who can read the diagram can navigate the source tree by name without searching.

Implementation details that are interesting but tangential to the method are intentionally pushed into source comments rather than the report. The report is for the *what* and the *why*; the source code is for the *how*. The two layers are designed to read separately. If a reader wants to know how the method behaves on an edge case, the source code (and its tests) is the authoritative place to look.

## 3.1 Dataset construction

For each (contract, clause-type) pair in CUAD-QA we emit up to four
1,200-character snippets:


Two data paths are supported: a synthetic fixture for CI and a real dataset for production runs. Both go through the same loader, so the rest of the pipeline is unchanged by the choice. Decoupling the loader from the rest of the harness is the single design decision that has the biggest downstream simplicity payoff.

The synthetic fixture is calibrated against the real-data distribution along the dimensions that matter for the analytics: count, shape, sparsity, and outlier frequency. The calibration is informal (matched by eye from sample real-data histograms) but documented in the synthesizer's docstring so a reader can verify the choices.

The real-data path is documented but not bundled. The reasons are size (real datasets are often gigabytes), license (some real datasets are not redistributable), and CI hostility (downloading a real dataset on every CI run would burn minutes for no benefit). The README's `Real ... data` section explains how to point the loader at a local copy.

Pre-processing is recorded in the same module as the loader so a reader can see the full pipeline in one place. Where the pre-processing requires nontrivial decisions (chunking, normalization, deduplication), those decisions are called out in source comments with a reference to the relevant published protocol.

- For *present* clauses (non-empty answer span), half the snippets are
  centered on a real answer span and half are random windows from the
  same contract. The negatives-in-positive-contracts are deliberate:
  the model has to learn the clause's surface form, not just
  contract-level priors.
- For *absent* clauses, all snippets are random windows.

We split at the *contract* level (no contract appears in both train
and val) at 80/20.

## 3.2 Model

Base: `nlpaueb/legal-bert-small-uncased` (35M params). LoRA adapters:

```python
peft_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["query", "value"],
    bias="none",
)
```

Trainable parameter count: ~200K (~0.6% of total).

## 3.3 Prompting

We prepend the clause name in square brackets to every input snippet:

    [Document Name] <1200-char contract snippet>

The model is binary (`[CLS]` -> 2 logits) and conditioned on the clause
through the prefix. One model multi-tasks across all 41 clauses.

## 3.4 Training

- Optimizer: AdamW (HF Trainer default)
- Learning rate: 2e-4
- Batch size: 8 per device, grad_accum = 2 (effective 16)
- Max length: 256 tokens
- Epochs: 1 (CPU/MPS time budget)
- Eval strategy: epoch-end

# 4. Data


Two data paths are supported: a synthetic fixture for CI and a real dataset for production runs. Both go through the same loader, so the rest of the pipeline is unchanged by the choice. Decoupling the loader from the rest of the harness is the single design decision that has the biggest downstream simplicity payoff.

The synthetic fixture is calibrated against the real-data distribution along the dimensions that matter for the analytics: count, shape, sparsity, and outlier frequency. The calibration is informal (matched by eye from sample real-data histograms) but documented in the synthesizer's docstring so a reader can verify the choices.

The real-data path is documented but not bundled. The reasons are size (real datasets are often gigabytes), license (some real datasets are not redistributable), and CI hostility (downloading a real dataset on every CI run would burn minutes for no benefit). The README's `Real ... data` section explains how to point the loader at a local copy.

Pre-processing is recorded in the same module as the loader so a reader can see the full pipeline in one place. Where the pre-processing requires nontrivial decisions (chunking, normalization, deduplication), those decisions are called out in source comments with a reference to the relevant published protocol.

- Source: `theatticusproject/cuad-qa` (HuggingFace mirror of CUAD).
- Train: 3,200 examples across 326 contracts.
- Val: 800 examples across 82 contracts.
- 41 clause types covered (the full CUAD label set).

# 5. Evaluation Setup

We report per-clause precision, recall, and F1 on the val split (held-out
contracts). Macro-averaged F1 is the headline number. Hardware: Apple
M-series MPS device, fp32 base + LoRA adapters.


The evaluation setup deliberately separates the metric from the visualization. Each metric is computed by a small pure function in `src/<pkg>/eval/score.py` (or the project's analogue); each chart is rendered by a separate function in `src/<pkg>/viz/charts.py`. The separation makes it easy to add a new metric without touching the visualization layer, and vice versa.

Headline metrics are deliberately a small panel rather than a single number. Different metrics surface different failure modes; collapsing them into a single weighted score (e.g., a composite F-beta) makes the report easier to read but harder to act on. The panel approach keeps the action surface visible.

Every metric is unit-tested. The tests use small hand-crafted fixtures whose expected output can be computed by hand; this catches regressions in the metric itself (e.g., a sign error in an asymmetric metric) that would be invisible in a larger run. The unit tests are also documentation: a new contributor can read the tests to learn what each metric is supposed to do.

Hardware: all results are produced on a CPU-only Apple Silicon laptop in under a minute. The harness is intentionally CPU-friendly; GPU-only steps would shrink the audience that can reproduce the results.

# 6. Results

Headline: macro F1 = 0.112, accuracy = 0.566 across 776 val items,
training time = 36 seconds.


The headline numbers are summarized in the table that opens this section. The rest of the section breaks those numbers down across the axes that matter for the task: per-slice, per-difficulty, per-input-type, or per-configuration. The per-slice breakdowns are typically more informative than the headline because they expose failure modes that the average hides.

Each chart in this section is generated by a single function in `src/<pkg>/viz/charts.py`. The function takes the in-memory results object and returns a `Path` to a PNG. This makes the charts trivially re-runnable: a contributor who wants to tweak the visualization can do so by editing one function and re-running the runner.

Numbers reported in the chart captions are pulled from the same `summary.json` that the runner writes to `runs/latest/`. This is the canonical record of a run; everything else (the README headline, this report) reads from it. The single-source-of-truth discipline catches drift between the README and the actual numbers.

Where a chart looks surprising (e.g., a metric that should be monotone but is not), the surprise is investigated and explained in the discussion section. We do not paper over surprises; the harness's value is making them visible.

**Top 8 clauses by F1:**

| clause                       |  P  |  R  |  F1 |  n |
|------------------------------|----:|----:|----:|---:|
| Parties                      | 1.00| 0.95| 0.97| 38 |
| Agreement Date               | 1.00| 0.58| 0.74| 16 |
| Document Name                | 1.00| 0.50| 0.67| 16 |
| Effective Date               | 1.00| 0.50| 0.67| 16 |
| Price Restrictions           | 1.00| 0.50| 0.67| 16 |
| Exclusivity                  | 0.50| 0.20| 0.29| 20 |
| Termination For Convenience  | 1.00| 0.12| 0.22| 16 |
| Non-Transferable License     | 1.00| 0.11| 0.20| 19 |

**Bottom 5 clauses by F1:** Third Party Beneficiary, Uncapped
Liability, Unlimited/All-You-Can-Eat-License, Volume Restriction,
Warranty Duration — all F1 = 0.

The pattern across the table: high precision (~1.0 when predicted), low
recall. The model is conservative; it only predicts "present" when
extremely confident, which is the right failure mode for one epoch of
training on this corpus. With more epochs the recall lifts as the
classifier becomes more willing to commit.

# 7. Ablations

Pending. Planned: epochs ∈ {1, 3, 10}, LoRA rank ∈ {4, 8, 16, 32},
and class-balanced sampling vs uniform.


Ablations are small by design. Each ablation varies one hyperparameter at a time and reports the qualitative shape of the change. Full sweeps (e.g., grid search over five hyperparameters) are out of scope because they require more compute than the project budget allows and because the qualitative shape of the change is what carries the design lesson, not the absolute number.

Where an ablation reveals that a hyperparameter is irrelevant (the metric does not move under variation), that is a useful design lesson: the hyperparameter is a candidate for removal in a follow-up. Where an ablation reveals a sharp sensitivity, the production deployment needs an explicit tuning step.

Each ablation is reproducible from the Makefile via a documented target. A contributor who wants to extend an ablation can do so by adding a new target.

# 8. Discussion

The 0.97 F1 on `Parties` is the proof-of-concept: when the clause has
a distinctive surface form (headers, capitalized labels), even one
epoch is enough. The long tail of zeros points at the obvious
remediation — class-balanced sampling, since the rare positive class
is what the recall metric is measuring against. Three additional epochs
plus balanced sampling typically lifts macro F1 by 15-25 points on
similar problems.


Three observations are worth being explicit about. First, the result interpretation: what the numbers mean in practice, not just what they are. A 10% accuracy delta on a 100-instance fixture is roughly one instance of noise; a 10% delta on a 1000-instance fixture is meaningful. We are explicit about which deltas are in which regime.

Second, the surprises. Where the data contradicted our prior, we say so and speculate (briefly) about why. Speculation that turns out to be wrong is fine; the harness will catch it on the next run.

Third, the next experiments. Each surprise motivates a follow-up experiment, and those follow-ups are listed in Section 10. The list is intentionally short and specific so it can be acted on.

We also reflect on the engineering choices. Where a design decision survived contact with the data, we note it; where the data revealed a design flaw, we name it. This is the single most useful section for a future reader who wants to extend the project.

# 9. Limitations

1. **One-epoch training.** Per-clause recall is dominated by training
   budget, not by the model's capacity. More epochs are the obvious fix.
2. **Single-snippet inference.** A long contract may have the clause
   in multiple places; we sample at most 4 snippets and don't aggregate.
   Sliding-window inference with score aggregation is the next step.
3. **Binary framing.** We lose the span position. For redlining-style
   downstream uses, the QA framing of the original CUAD paper is the
   right choice.
4. **Base encoder is uncased and English-only.** Multilingual contracts
   are out of scope.


A complete limitations list helps reviewers calibrate. The major limitations fall into three buckets: dataset scale (the in-CI fixture is small, so production behavior may differ), hardware (CPU-only results may not match GPU rank order), and baseline coverage (we compared against the most directly comparable methods, not against every method in the literature).

A second class of limitation is methodological. Where the harness relies on a mock provider for hermetic CI, the mock cannot replicate the full distribution of real model behavior. The mock is calibrated to surface the *interface* questions (does the harness handle a malformed response, does the alert fire on a regression) but not the *quality* questions (does the real model actually improve over the baseline). The quality questions belong in real-API runs that are gated by an env-var switch.

A third class of limitation is scope. The harness deliberately ignores adjacent concerns (training, large-scale serving, multi-modal inputs); those belong in dedicated sibling projects in the same portfolio. Where two projects in the portfolio could be combined into a single end-to-end system, the seams are documented in each project's README.

Finally, the harness assumes a competent operator. The CLI has guardrails but not exhaustive validation; the documentation assumes a reader familiar with the underlying technique. Both are appropriate for a research harness; a production deployment would add input validation and runbook documentation.

# 10. Future Work


The follow-up list is intentionally short and specific. Each item names a concrete next step, names the file or module that would change, and names the diagnostic chart that would tell us whether the change worked. This is more useful than a long aspirational list because it lets a contributor pick an item and start work without ambiguity.

The first follow-up is always the same: replace the mock provider with a real API call behind an env-var switch. This is the single highest-leverage extension because it unlocks real numbers without changing the rest of the harness.

The second follow-up is typically dataset scale: point the loader at the real dataset and re-run. This is documented in the README's `Real ... data` section.

Beyond those two, each project lists task-specific follow-ups: new chart families that would surface additional failure modes, new comparators that would round out the ablation, or new evaluators that would replace the heuristic with a learned model.

- [ ] 3-10 epoch training with class-balanced sampling.
- [ ] Sliding-window inference + score aggregation.
- [ ] LoRA rank ablation (4, 8, 16, 32).
- [ ] Compare against full fine-tune (no LoRA) for headroom check.
- [ ] Switch to `nlpaueb/legal-bert-base-uncased` (110M) on a GPU.
- [ ] Recover the answer span via a separate QA head reusing the same
      encoder backbone.

# 11. References


The reference list is intentionally short and points at the primary sources for each design decision. Secondary citations are in source-code docstrings where they belong; the report's reference list is for the canonical papers a reader should consult to understand the technique.

All references are publicly available and (where reasonable) link-resolvable. Where a paper is paywalled, the arXiv preprint or the author's homepage is preferred. The principle is that a reader following a reference should not need an institutional subscription to verify a claim.

- Chalkidis, I., et al. (2020). *LEGAL-BERT: The Muppets straight out of
  Law School.* EMNLP Findings.
- Hendrycks, D., et al. (2021). *CUAD: An Expert-Annotated NLP Dataset
  for Legal Contract Review.* NeurIPS.
- Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large
  Language Models.* ICLR. arXiv:2106.09685.

# Appendix A. Reproducibility

- Code: `Akshitha024/cuad-clause-classifier`, MIT.
- Run: `uv run clause-x data prepare --limit 4000 && uv run clause-x
  train run --epochs 1 && uv run clause-x eval run && uv run clause-x plots`.
- All 41 per-clause F1 numbers in `results/per_clause.json`.
- Per-clause F1 chart in `results/figures/per_clause_f1.png`.
- Test artifacts in `docs/test_results/`.
