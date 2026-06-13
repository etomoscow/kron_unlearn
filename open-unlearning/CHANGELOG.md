# Changelog

## 2026-06-13

- Completed the 3B pilot queue (`queue_I_3b_pilot`) for Llama-3.2-3B forget10 scratch vs K-FORGE layer15 s0.62 S50, seed0.
- Completed the MUSE News pilot queue (`queue_J_muse_news_pilot`) for Llama-2-7b-hf NPO vs SimNPO seed sweep.
- Fixed the MUSE pilot launch path in `logs/review_queue/queue_J_muse_news_pilot_after_3b.sh` by:
  - using the local cached `Llama-3.2-3B-Instruct` snapshot,
  - removing the invalid collator CLI override,
  - and pointing `retain_logs_path` at the existing baseline `saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json`.
- Fixed the shared queue idle gate in `logs/review_queue/queue_common.sh` so `WAIT_FOR_IDLE` now defaults to `0`, allowing free GPUs to launch immediately unless a script opts into waiting.
- Verified the completed MUSE pilot outputs:
  - `privleak = 24.097397140197867`
  - `extraction_strength = 0.014195035635481518`
- Confirmed several GPU1 queue scripts now skip cleanly because their summaries/checkpoints already exist.
