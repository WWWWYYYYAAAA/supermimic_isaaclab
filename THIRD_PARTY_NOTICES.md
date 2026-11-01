# Third-party code

`supermimic/normalization.py` adapts `EmpiricalNormalization` from
[RSL-RL v5.5.2](https://github.com/leggedrobotics/rsl_rl/blob/v5.5.2/rsl_rl/modules/normalization.py).
Copyright ETH Zurich, NVIDIA and Preferred Networks as retained in the source;
see [BSD-3-Clause license](licenses/rsl_rl-BSD-3-Clause.txt). Local changes add
leading-dimension flattening, no-gradient updates and empty-batch handling.

The v7 feature and integration conventions follow
[SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced);
normalization and optional analytic termination differ as described in NETWORK_INPUTS.md.
