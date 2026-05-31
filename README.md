# K-FORGE

This repository contains the anonymous code release for K-FORGE, implemented on top of OpenUnlearning.

For review, the relevant implementation files are:

```text
open-unlearning/src/trainer/unlearn/kforge.py
open-unlearning/configs/trainer/KFORGE.yaml
open-unlearning/configs/experiment/unlearn/tofu/kforge.yaml
open-unlearning/docs/kforge.md
```

The main trainer is registered as `KFORGE`. The method estimates forget/retain Kronecker-Fisher factors and applies a closed-form low-rank Wiener edit that can be used directly or as initialization for downstream unlearning methods.

Minimal entry point:

```bash
cd open-unlearning
python src/train.py --config-name=unlearn experiment=unlearn/tofu/kforge
```

The code inherits OpenUnlearning setup and dependencies. See `open-unlearning/README.md` for the base framework and `open-unlearning/LICENSE` for the upstream MIT license.
