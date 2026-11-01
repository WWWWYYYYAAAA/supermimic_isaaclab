# Network and motion-goal contract (v7)

New training uses the input and dynamics semantics of `SuperMimic-Reproduced`,
with `EmpiricalNormalization` replacing its fixed input scaling. This is an
implementation alignment, not a claim of reproducing the paper's results.

| Network | Input (default K=10, H=10, context=33) | Output |
|---|---|---|
| Motion encoder | 68*K reference features + 9 anchor features = 689 | 32, tanh |
| Actor decoder | latent 32 + term-major proprioception history 930 | 29 actions |
| Critic | canonical state 71 + goal 689 + context 33 + root error 4 = 797 | scalar value |
| Dynamics | canonical state 71 + action 29 + context 33 = 133 | 70 residuals |
| Neural termination | signed height/tilt errors and their magnitudes (4) | terminal logit |

Encoder/decoder hidden widths are 2048/1024/512/256, critic 256/128,
dynamics 2048/2048/2048, termination 64/64. All use SiLU. Dynamics has no hidden
LayerNorm. Decoder output uses orthogonal initialization with gain 0.01;
dynamics output is zero-initialized.

## Inputs and normalization

Actor observations per frame are `[q + bias - q0 (29), qd (29), omega_body (3),
projected_gravity (3), previous_action (29)]`. Optional configured term scaling
and noise still apply. History is oldest first **within each term**: all H joint
position frames, then all H velocity frames, etc. Call `MDP.flatten_history()`;
a direct flatten of `[B,H,93]` has the wrong order.

`EmpiricalNormalization` follows RSL-RL v5.5.2: sample-count-weighted running
mean/variance, `(x-mean)/(sqrt(var)+0.01)`, no learned affine parameters, no
implicit updates in `forward`, and no default sample-count cutoff. The local
implementation additionally flattens leading batch/time dimensions and updates
under `no_grad`; distributed statistics merge across ranks when initialized.
Actor observation history (930) and goal (689), critic input (797), dynamics
input (133), and termination input (4) have independent normalizers. The bounded
32D latent is concatenated with normalized observations at the decoder.

Actor/dynamics/termination statistics update explicitly from real collection;
critic statistics update from its detached training inputs. Optimization and
imagined forward calls do not update statistics. Playback and exported modules
use saved statistics. `--freeze_policy_norm_at N` freezes only actor statistics.

Canonical state is `[height (1), heading-removed rot6d (6), linear_velocity_heading
(3), angular_velocity_body (3), joint_position (29), joint_velocity (29)]`.
Global XY and yaw are removed. Context preserves the Reproduced friction/bias/CoM
scaling before the joint empirical normalizer. With optional PD randomization,
context is 91D: dynamics input 191, critic input 855 (anchor enabled).

Critic root error is `[robot_position-reference_position in reference heading
frame (3), wrap(robot_yaw-reference_yaw) (1)]`. Critic therefore receives explicit
yaw drift even with `anchor_obs=False`.

## Dataset to motion goal

CSV files retain their original root position/quaternion and joint columns;
no dataset rewrite or cached legacy goal is required. `MotionLib` still resamples
30 Hz CSV to 50 Hz. `MotionLib.goal(f,K)` now builds frames **f..f+K-1**:

| Per-frame slice | Meaning |
|---|---|
| 0:29 | reference joint position minus default joint pose |
| 29:58 | reference joint velocity, no 0.1 multiplier |
| 58:61 | reference projected gravity in its body frame |
| 61:64 | reference angular velocity in its body frame |
| 64:67 | reference linear velocity in its heading frame |
| 67:68 | reference root height |

Tracking ends at current frame f when f+K-1 >= the last reference frame.
For K=10 the final state is f=last-9; no action is executed from that state.
Random resets exclude this frame and later frames; imagination requires a complete
goal through its final bootstrap state. Evaluation maximum return uses N-K steps
per N-frame clip, and clips with N<=K are rejected. This is a clip timeout, not a
failure label for the learned termination head.

The low-level MotionLib.goal() still clamps out-of-range standalone queries to
the complete last frame as a defensive fallback. Normal training/play/evaluation
ends before such padding is needed. No reference window crosses into another clip.

`MDP.goal(S,f,K,anchor_obs=True)` appends the current reference **torso_link**
position (3) and orientation (6) in the robot torso frame, using differentiable
FK. The 6D rotation is the first two matrix columns flattened row-major. Anchor
features are dynamic, not cached in the reference dataset. In noisy training,
anchor uniform half-widths are 0.25 for position and 0.05 for orientation;
playback/evaluation use clean features. The actor and critic receive this goal.

## Dynamics output and termination

Output order is `[dv_heading (3), dw_body (3), dqd (29), dp_heading (3),
dtheta_body (3), dq (29)]`, scaled by `[0.3,0.5,2.0,0.01,0.02,0.02]` per group.
First update velocities; then integrate position, rotation and joints at dt=0.02
and add the last three residual groups. Zero output extrapolates constant velocity.
Replay storage stays `(p,R,v_world,w_world,q,qd)`; the network converts angular
velocity to/from body coordinates at the boundary.

State loss sums 70 squared errors divided by fixed scales
`[0.05,0.1,0.5,1.0,0.1,2.0]` for position/rotation/linear velocity/body angular
velocity/joints/joint velocity, then averages batch and time. Episode masks remove
transitions after a reset. Neural termination adds masked BCE with default weight
1.0 and gradients through the predicted state into dynamics.

`--termination_head neural` is default. Its signed features are reference minus
robot height and reference minus robot projected-gravity-z, divided by the hard
thresholds before concatenation with their magnitudes and empirical normalization.
The neural sigmoid supplies imagination survival/critic bootstrapping; parameter
freezing during actor updates preserves derivatives with respect to state.
`--termination_head analytic` instead uses the existing smooth analytic rule and
creates no learned head/BCE loss. Real simulator resets always use the hard
height/tilt rule. Replay stores `term` separately from `done`: time limits and clip
ends are not positive failure labels. Predicted next state and reference f+1 are
paired consistently with this project's simulator transition timing.

## Options and compatibility

Defaults: `--anchor_obs` enabled, `--termination_head neural`,
`--term_loss_weight 1.0`. Disable with `--no-anchor_obs` (or `--no-anchor-obs`);
choose analytic termination with `--termination_head analytic`.
Without anchor, goal=680 and critic=788. Checkpoint `args` saves both switches;
`task_settings` also saves the command's anchor flag.

All new checkpoints use `model_input_version=7`. Old weights require a fresh
training run: shapes alone cannot identify the changed coordinate conventions.
Historical v2-v6 BatchNorm policies retain separate inference/export adapters,
including their original goal layout, frame-major history and bias sign. The
BatchNorm classes in the source exist only for these adapters, not new networks.
New TorchScript policy/encoder/decoder exports include empirical statistics and
metadata describing v7 term-major history, goal slices and anchor suffix.
Existing exported files and historical training data are not rewritten.

This change keeps existing reward formulas, physical PD actions, randomization,
reference sampling and BPTT schedule. It does not port the entire Reproduced
simulator or training algorithm. In particular, the local actuator's calibration
offset convention is retained; the actor observation bias sign is aligned to
Reproduced. GPU long-run training and convergence must be evaluated separately.

Validation (2026-11-01): main project 78 CPU tests and 15 tool tests; Isaac project
76 CPU tests and 3 legacy motion-goal tests. Eight cross-checkout feature/prediction
comparisons passed exact CPU equality. No real GPU training was run.
