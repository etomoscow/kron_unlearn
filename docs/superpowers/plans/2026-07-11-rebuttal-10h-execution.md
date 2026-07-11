# K-FORGE 10-Hour Rebuttal Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert all completed rebuttal experiments into verified reviewer-facing evidence and use the remaining GPU window to close the matched-operating-point robustness gap.

**Architecture:** Reuse completed checkpoints and OpenUnlearning evaluation paths. Add only reproducible aggregation and attack runners, launch independent GPU queues, and record every decision and result in one Markdown audit log.

**Tech Stack:** Bash, Python standard library/SciPy, PyTorch, Transformers, bitsandbytes, OpenUnlearning, tmux.

## Global Constraints

- Deadline: `2026-07-12 00:57 UTC`.
- Primary endpoint: Forget Q/A Probability with utility non-inferiority margin `-0.01`.
- Always report extraction and Forget Q/A ROUGE, including adverse results.
- Never overwrite completed summaries or replace an unfavorable arm after observing it.
- Preserve all unrelated worktree changes.

---

### Task 1: Verify and aggregate completed additions

**Files:**
- Create: `open-unlearning/scripts/summarize_rebuttal_additions.py`
- Modify: `docs/rebuttal/2026-07-11-autonomous-10h-log.md`

**Interfaces:**
- Consumes: completed TOFU/MUSE `*_SUMMARY.json` files.
- Produces: terminal Markdown tables plus a nonzero exit on missing or malformed inputs.

- [ ] Implement fixed, structured aggregation for Gemma controls, Gemma compute matching, MUSE follow-up, held-out Gemma seed, and 4/8-bit quantization.
- [ ] Run `python open-unlearning/scripts/summarize_rebuttal_additions.py --check` and require all expected files to parse.
- [ ] Record means, sample standard deviations, paired deltas, and paired tests in the audit log.

### Task 2: Add matched-operating-point relearning runner

**Files:**
- Create: `open-unlearning/scripts/kforge_rebuttal_matched_relearning.sh`
- Modify: `docs/rebuttal/2026-07-11-autonomous-10h-log.md`

**Interfaces:**
- Consumes: Llama-3.2-1B NPO scratch S100 and K-FORGE `alpha=0.60` S50 checkpoints for seeds `0,1,2`.
- Produces: one-epoch and three-epoch TOFU summaries for both arms without changing attack hyperparameters between arms.

- [ ] Add skip-on-valid-summary logic and a TSV manifest.
- [ ] Run `bash -n open-unlearning/scripts/kforge_rebuttal_matched_relearning.sh`.
- [ ] Launch one-epoch and three-epoch queues on separate idle GPUs.
- [ ] Monitor every ten minutes and retry only transient failures with identical settings.

### Task 3: Add matched-operating-point quantization runner

**Files:**
- Create: `open-unlearning/scripts/kforge_rebuttal_matched_quantization.sh`
- Modify: `docs/rebuttal/2026-07-11-autonomous-10h-log.md`

**Interfaces:**
- Consumes: the same matched scratch S100 and K-FORGE S50 checkpoints.
- Produces: three-seed 8-bit and 4-bit TOFU summaries for both arms.

- [ ] Reuse the working bitsandbytes configuration from `logs/review_quantization/run_quantization_revert_v1.sh`.
- [ ] Run `bash -n open-unlearning/scripts/kforge_rebuttal_matched_quantization.sh`.
- [ ] Launch on an idle GPU and verify every manifest row reaches `ok` or `skipped` with a valid summary.

### Task 4: Spend remaining capacity on the highest-value confirmation

**Files:**
- Modify: `docs/rebuttal/2026-07-11-autonomous-10h-log.md`

**Interfaces:**
- Consumes: results from Tasks 1-3 and current free-GPU state.
- Produces: one prespecified follow-up selected before launching.

- [ ] If matched robustness is complete with time remaining, select exactly one of: an additional Gemma control seed, a stronger matched relearning duration, or an additional quantized budget.
- [ ] Record the choice and decision rule before launch.
- [ ] Run and verify the selected follow-up without suppressing negative results.

### Task 5: Produce final rebuttal text and audit

**Files:**
- Modify: `REBUTTAL.md`
- Modify: `docs/rebuttal/2026-07-11-autonomous-10h-log.md`

**Interfaces:**
- Consumes: verified aggregate output only.
- Produces: concise reviewer-specific responses with exact tables, limitations, and artifact paths.

- [ ] Replace stale future-tense or incomplete claims with completed evidence.
- [ ] State the strongest defensible subclaims and their metric boundaries.
- [ ] Run number-consistency checks between source JSON and `REBUTTAL.md`.
- [ ] Run Markdown/math delimiter checks and `git diff --check`.
- [ ] Verify no required experiment sessions remain running at the deadline and write the final report.

