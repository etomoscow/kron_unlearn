# K-FORGE

K-FORGE is registered as the `KFORGE` trainer and can be selected through Hydra
with `trainer=KFORGE`.

## Method

For every regex-selected `torch.nn.Linear` module, K-FORGE estimates empirical
Kronecker factors from forget and retain calibration batches:

- `A = E[x x^T]` from linear-layer inputs.
- `B = E[g g^T]` from linear-layer output gradients.

Calibration uses valid causal-LM loss positions only, sums token-loss
gradients before forming `B`, and runs in eval mode. Factor damping is relative
to each matrix's own scale.

Two edit variants are available:

- `legacy_v1`: the original asymmetric heuristic used in the first diagnostic
  runs. It is retained for ablation only.
- `wiener_v2`: the theorem-backed Wiener edit. It simultaneously diagonalizes
  forget and retain Kronecker factors, applies a `lambda_tradeoff` retain budget,
  truncates in the rescaled `Y` basis, and maps back through retain whitening.

Both variants remain training-free: the only backward passes are calibration
passes used to collect factors.

## Configuration

Use the default TOFU example:

```bash
python src/train.py --config-name=unlearn experiment=unlearn/tofu/kforge
```

Important knobs:

- `trainer.method_args.target_modules_regex`: full-match regex over module
  names. The default is `.*mlp\.down_proj$`.
- `trainer.method_args.rank`: truncated SVD rank per layer.
- `trainer.method_args.strength`: multiplier on the negative edit.
- `trainer.method_args.damping`: Tikhonov damping applied before Cholesky.
- `trainer.method_args.damping_floor`: positive floor for scale-relative
  damping, default `1e-12`.
- `trainer.method_args.max_calibration_batches`: number of forget and retain
  batches used for factor estimation.
- `trainer.method_args.skip_modules_larger_than`: optional parameter-count
  guard for very large linears.
- `trainer.method_args.max_target_modules`: optional cap on matched modules.
  The default config uses `4` as a memory guard; set it to `null` for all
  matched modules.
- `trainer.method_args.factor_mode`: `kron` for full Kronecker factors or
  `diagonal` for the diagonal-Fisher ablation.
- `trainer.method_args.use_retain_fisher`: set to `false` for the forget-only
  Fisher ablation.
- `trainer.method_args.edit_variant`: `legacy_v1` or `wiener_v2`.
- `trainer.method_args.lambda_tradeoff`: retain-Fisher penalty used only by
  `wiener_v2`; this is distinct from factor damping.
- `trainer.method_args.edit_weight_dtype`: `float32` keeps the edited
  checkpoint in fp32 so small updates survive serialization; `preserve`
  retains the model's current dtype for ablation.

## Current Scope

The implementation supports single-process calibration. Existing results from
the initial `legacy_v1` bf16 runs should be treated as historical diagnostics;
headline experiments should be rerun with corrected calibration, fp32 edited
checkpoints, and explicit `edit_variant` selection.
