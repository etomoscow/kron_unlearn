#!/usr/bin/env python
"""Create matched-norm low-rank init checkpoints for K-FORGE controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_linear_modules(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            yield name, module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--template-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["random_low_rank", "weight_svd"], required=True)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-delta-norm", type=float, default=1e-8)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    template = AutoModelForCausalLM.from_pretrained(
        args.template_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    base.eval()
    template.eval()

    template_modules = dict(iter_linear_modules(template))
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    edited = []
    with torch.no_grad():
        for name, module in iter_linear_modules(base):
            if name not in template_modules:
                continue
            w = module.weight.data.float()
            delta_template = template_modules[name].weight.data.float() - w
            target_norm = torch.linalg.vector_norm(delta_template)
            if target_norm.item() <= args.min_delta_norm:
                continue

            if args.mode == "random_low_rank":
                left = torch.randn(
                    w.shape[0], args.rank, device=device, dtype=torch.float32, generator=generator
                )
                right = torch.randn(
                    args.rank, w.shape[1], device=device, dtype=torch.float32, generator=generator
                )
                delta = left @ right
            else:
                u, s, vh = torch.linalg.svd(w, full_matrices=False)
                r = min(args.rank, s.numel())
                delta = -((u[:, :r] * s[:r]) @ vh[:r, :])

            delta_norm = torch.linalg.vector_norm(delta).clamp_min(1e-12)
            delta = delta * (target_norm / delta_norm)
            module.weight.data.copy_((w + delta).to(dtype=module.weight.dtype))
            edited.append((name, float(target_norm.item())))

    if not edited:
        raise RuntimeError("No edited linear modules found from template/base delta.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "matched_init_meta.txt").open("w") as f:
        f.write(f"mode={args.mode}\n")
        f.write(f"rank={args.rank}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"base_model={args.base_model}\n")
        f.write(f"template_model={args.template_model}\n")
        for name, norm in edited:
            f.write(f"edited_module={name}\ttemplate_delta_norm={norm:.10g}\n")


if __name__ == "__main__":
    main()
