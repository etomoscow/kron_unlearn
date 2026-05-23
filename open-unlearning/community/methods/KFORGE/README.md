# K-FORGE

K-FORGE is a one-shot unlearning method that estimates forget and retain
Kronecker-Fisher factors for selected linear layers and applies a low-rank edit
without gradient-ascent fine-tuning.

The code keeps the original `legacy_v1` heuristic for ablation and adds
`wiener_v2`, the retain-budgeted Wiener construction. For new experiments, set
`trainer.method_args.edit_variant=wiener_v2` and choose
`trainer.method_args.lambda_tradeoff` explicitly.

The default OpenUnlearning configuration edits MLP `down_proj` layers. Increase
`trainer.method_args.target_modules_regex`, `rank`, and
`max_calibration_batches` for stronger edits after checking memory use. Edited
checkpoints default to fp32 so small updates survive serialization.

Example:

```bash
bash community/methods/KFORGE/run.sh
```
