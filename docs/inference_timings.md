# Policy and steering processing times

Measured from existing local logs, reviewed on November 25, 2026. NaVILA
prediction averaged 668 ms without steering and 796 ms with steering in the
sampled runs. Shared hardware contention may explain part of the difference,
but these runs do not isolate its cause.

## Processing times excluding startup

All values are milliseconds, rounded to the nearest millisecond. Startup
exclusion removes the first logged policy timing and first recorded steering
frame of each episode. This is an operational exclusion, not proof that every
subsequent sample is fully warmed up.

| Configuration | Policy mean | Policy median | Policy P95 | SteerNav mean | SteerNav median | SteerNav P95 |
|---|---:|---:|---:|---:|---:|---:|
| OmniVLA | 600 | 590 | 670 | — | — | — |
| OmniVLA + SteerNav | 647 | 630 | 780 | 414 | 406 | 536 |
| NaVILA | 668 | 662 | 694 | — | — | — |
| NaVILA + SteerNav | 796 | 821 | 893 | 671 | 597 | 1,038 |
| TIC-VLA | 67 | 60 | 90 | — | — | — |
| TIC-VLA + SteerNav | 80 | 70 | 120 | 479 | 470 | 569 |

Policy and steering columns are separate wall-clock measurements. They cannot
be added to obtain measured end-to-end latency because the nodes run
independently. Steering timings exclude visualization, frame saving, and final
publication.

TIC-VLA policy calls use asynchronously generated model state. Their fast action
updates are not equivalent to fresh reasoning generation. Before startup
exclusion, separately logged background reasoning generation averaged 5,283 ms
standalone and 6,505 ms with steering; those values are not the policy timings
in this table.

## Sources and measurement boundaries

- OmniVLA and TIC-VLA: pooled easy, medium, and hard crossover v2 runs whose
  batch paths contain `crossover_v2_20261117_221003_`, using `attempt1` only,
  under `/home/jim/Projects/OmniVLA/logs/orchestration/` and
  `/home/jim/Projects/TIC-VLA/logs/orchestration/`.
  Policy samples use periodically logged `last=` values, not every policy call.
  OmniVLA times `model.run()`; TIC-VLA times `model.predict_async()`.
- NaVILA without steering: `logs/orchestration/20261125_193659/`, normal mode,
  `icra_scenarios_v4/scenario3_5.yaml`. There were 25 retained predictions after
  excluding the first prediction (882.7 ms). Episode 1 ended in a pedestrian
  collision; episode 2 had no local inference samples at review time.
- NaVILA with steering: `logs/orchestration/20261125_192011/`, steered mode,
  the same scenario file. There were 59 retained predictions across two
  episodes after excluding each episode's first prediction, and 38 retained
  steering frames from episode 1. The run did not finish normally: remote
  status was 141, and episode 2 had no completion result in the local log.
- NaVILA times each `policy.predict()` call, including image preprocessing,
  model generation, and text decoding. Frame insertion, path processing, and
  publication are outside this policy measurement.
- Steering samples use `frame_result.processing_seconds` from
  `receiver_timing.jsonl`. P95 uses the nearest-rank percentile.

Retained policy sample counts: OmniVLA normal 928, OmniVLA steered 2,213,
NaVILA normal 25, NaVILA steered 59, TIC-VLA normal 1,641, and TIC-VLA steered
2,931. Retained steering counts: OmniVLA 3,613, NaVILA 38, and TIC-VLA 3,652.

These are historical comparisons across different trajectories, dates, and
sample counts, rather than a controlled benchmark on identical observations.

## Why steering may increase policy time

Separate processes can still share GPU compute, memory bandwidth, CPU time,
and transfer resources. NaVILA and SteerNav's depth and detection models
default to CUDA. If placed on the same GPU, concurrent work can delay policy
execution; a wall-clock timer around prediction includes those waits.

NaVILA's mean increased by about 128 ms (19%) with steering in these samples.
Resource contention is a hypothesis, not an established causal finding.
Steering also changes the robot's trajectory and camera observations, which
can change model work. GPU placement and utilization were not captured in
this analysis.

To isolate contention, replay identical images through each policy with and
without SteerNav running, record GPU placement and utilization, and measure
individual preprocessing, generation, depth, detection, optimization, and
visualization stages.
