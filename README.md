# K-FORGE

K-FORGE is implemented on top of OpenUnlearning.

implementation is at: 

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
