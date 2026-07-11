#!/usr/bin/env python3
"""Reproduce the analytical K-FORGE setup and per-step FLOP estimates."""

import argparse
import json


PRESETS = {
    "llama-1b": {"parameters": 1_235_814_400, "rows": 2_048, "columns": 8_192},
    "llama-3b": {"parameters": 3_212_749_824, "rows": 3_072, "columns": 8_192},
}


def estimate_setup_flops(
    *,
    parameters: int,
    calibration_input_tokens: float,
    factor_token_rows: float,
    rows: int,
    columns: int,
) -> dict[str, float]:
    """Estimate calibration plus dense factor-algebra FLOPs for one edited matrix.

    Calibration uses the conventional 6PT transformer-training approximation.
    Factor accumulation counts two multiply-add FLOPs for each activation and
    output-gradient outer product. The 10(m^3+n^3) term is the paper's
    conservative envelope for Cholesky, whitening, full SVD, and edit
    application. This is an analytical approximation, not a hardware profiler.
    """
    if min(parameters, calibration_input_tokens, factor_token_rows, rows, columns) <= 0:
        raise ValueError("all compute-estimate inputs must be positive")
    calibration = 6.0 * parameters * calibration_input_tokens
    factor_accumulation = 2.0 * factor_token_rows * (rows**2 + columns**2)
    dense = float(10 * (rows**3 + columns**3))
    total = calibration + factor_accumulation + dense
    return {
        "calibration_flops": calibration,
        "factor_accumulation_flops": factor_accumulation,
        "dense_flops": dense,
        "total_flops": total,
        "calibration_fraction": calibration / total,
    }


def estimate_step_flops(
    *, parameters: int, forget_input_tokens: float, retain_input_tokens: float
) -> dict[str, float]:
    """Estimate one SimNPO or NPO optimizer step under the paper convention."""
    if min(parameters, forget_input_tokens, retain_input_tokens) <= 0:
        raise ValueError("all step-estimate inputs must be positive")
    simnpo = 6.0 * parameters * (forget_input_tokens + retain_input_tokens)
    npo = simnpo + 2.0 * parameters * forget_input_tokens
    return {"simnpo_flops": simnpo, "npo_flops": npo}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=PRESETS)
    parser.add_argument("--calibration-input-tokens", type=float, default=47_727.9)
    parser.add_argument("--factor-token-rows", type=float, default=18_614)
    parser.add_argument("--forget-step-input-tokens", type=float, default=3_042.32)
    parser.add_argument("--retain-step-input-tokens", type=float, default=2_923.67)
    args = parser.parse_args()

    config = PRESETS[args.model]
    setup = estimate_setup_flops(
        **config,
        calibration_input_tokens=args.calibration_input_tokens,
        factor_token_rows=args.factor_token_rows,
    )
    steps = estimate_step_flops(
        parameters=config["parameters"],
        forget_input_tokens=args.forget_step_input_tokens,
        retain_input_tokens=args.retain_step_input_tokens,
    )
    output = {
        "model": args.model,
        "parameters": config["parameters"],
        "edited_weight": [config["rows"], config["columns"]],
        "calibration_input_tokens": args.calibration_input_tokens,
        "factor_token_rows": args.factor_token_rows,
        **setup,
        **steps,
        "setup_in_npo_steps": setup["total_flops"] / steps["npo_flops"],
        "setup_in_simnpo_steps": setup["total_flops"] / steps["simnpo_flops"],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
