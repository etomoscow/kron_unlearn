# K-FORGE 12-Hour Rebuttal Experiment Design

## Objective

Maximize the evidential strength of the following scoped claim:

> K-FORGE is the best tested initialization for a fixed preference-based
> optimizer at matched downstream budget, and its benefit over scratch remains
> after charging the one-time initialization cost.

This is not a claim that K-FORGE is the best standalone unlearning algorithm.
Any stronger result will be reported only as a setting-specific subclaim.

## Prespecified Evidence

### 1. Held-Out Gemma Confirmation

Run one new seed (`3`) on Gemma-3-1B, TOFU `forget10`, at 100 downstream
steps for:

- NPO scratch and K-FORGE initialization;
- SimNPO scratch and K-FORGE initialization.

K-FORGE uses the previously selected `alpha=0.8` checkpoint. The strength was
selected before this confirmation run by minimizing one-shot Forget Q/A
Probability subject to a maximum absolute utility loss of `0.01`.

### 2. Gemma Initialization Controls

At 100 steps and seeds `0,1,2`, compare K-FORGE with:

- random rank-2 perturbation;
- rank-2 weight-SVD perturbation;
- diagonal-Fisher initialization;
- forget-only Fisher initialization.

Controls use the same edited module and rank. Random and weight-SVD edits are
matched to the K-FORGE edit Frobenius norm. NPO and SimNPO otherwise use the
same existing training and evaluation configuration.

### 3. Compute-Matched Gemma Comparison

Measure K-FORGE setup time, including calibration, factorization, edit
application, and checkpoint writing. Convert this cost to downstream steps
using the measured scratch time per step:

$$
T_{\mathrm{cm}} = 100 + \left\lceil
\frac{t_{\mathrm{KF}}}{\bar t_{\mathrm{step}}}
\right\rceil.
$$

Run scratch NPO and SimNPO at this budget for seeds `0,1,2`, then compare them
against the existing 100-step K-FORGE runs.

### 4. Time-Permitting MUSE Follow-Up

Run a fixed one-shot strength grid on MUSE. Select the point with the lowest
Forget KnowMem ROUGE subject to an absolute Retain KnowMem ROUGE drop of at
most `0.01` from the target checkpoint. Only that point may proceed to
downstream SimNPO. Extraction and Forget VerbMem ROUGE remain mandatory
secondary metrics. All grid results remain part of the audit trail, including
negative results.

### 5. Time-Permitting Quantization Audit

Run an 8-bit smoke test on the existing Llama-3.2-1B TOFU `forget10` NPO S100
scratch checkpoint and its K-FORGE `alpha=0.60` counterpart for seed `0`.
Proceed to seeds `1,2`, then 4-bit, only if the existing environment supports
model loading and TOFU evaluation without adding a new quantization pipeline.

## Metrics And Decision Rules

The primary endpoint is Forget Q/A Probability, lower being better. A result
qualifies for the primary claim only when mean utility satisfies

$$
U_{\mathrm{KF}}-U_{\mathrm{comparison}} \ge -0.01.
$$

Extraction strength and Forget Q/A ROUGE are mandatory secondary endpoints.
They are reported even when they disagree with Forget Q/A Probability.

K-FORGE may be called the best tested initializer only if it has the lowest
mean primary endpoint among all completed controls satisfying the utility
constraint. A stronger Pareto-dominance claim requires both lower Forget Q/A
Probability and non-lower utility. Statistical tests use paired seeds where
available; pilot and held-out confirmation results are identified separately.

## Execution Order And Resource Use

The strict priority is:

1. held-out Gemma seed;
2. Gemma initialization controls;
3. compute-matched Gemma scratch runs;
4. MUSE follow-up;
5. quantization audit.

Use currently idle GPUs independently. Existing completed checkpoints and
evaluations are reused; queues must skip a task only when its final summary
exists and parses. No completed result may be overwritten.

## Failure Handling

- Retry transient download or evaluation failures with the same configuration.
- On OOM, retry once with an existing lower-memory evaluation setting; do not
  change model, method, strength, rank, or metric definitions.
- Stop a time-permitting branch if its smoke test takes more than 30 minutes to
  make progress or requires new infrastructure.
- Do not replace a negative model, benchmark, seed, or metric with a more
  favorable one after observing results.

## Verification

Before using a result in the rebuttal:

- verify every expected summary exists and parses;
- verify queue manifests have matching `started` and `ok` entries;
- aggregate by structured JSON fields rather than log text;
- report the number of seeds and sample standard deviation;
- retain negative and mixed secondary metrics in the result table.
