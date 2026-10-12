# Assignment 2 Part 2: Pedestrian flow matching, design choices

Notebook: [`assignments/Assignment_2_part2.ipynb`](assignments/Assignment_2_part2.ipynb).
Outputs (plots, `results.txt`, best checkpoints): [`assignments/part2_outputs/`](assignments/part2_outputs/).
Final run on Snellius, job 27956341 (12 Oct), 1x A100, 31 min ([`assignments/run_part2.job`](assignments/run_part2.job)).
An earlier version used 1D convolutions over time and frame-9 edge features for the future frames (jobs 27780417 and 27948583). Its results are kept below for comparison.

## Problem setup

- Every pedestrian is one 30-frame sequence: frames 0-9 are observed, frames 10-29 are generated.
- The model generates the 20 future **velocities** (normalized), not positions. Positions are recovered by integration:
  `p^{9+k} = p^9 + dt * sum_{j<=k} v^{9+j}` with `dt = 0.5 s`.
- **Time bundling:** all 20 future frames come out of one model call.
- Normalization statistics (velocity, relative position, absolute position) are computed on the training split only.

## Architecture: space-time velocity field

- **Input grid of pedestrians x frames.**
  - Observed frames carry the position relative to frame 9, the velocity and an "observed" flag.
  - Generated frames carry the noisy velocity `x_tau`.
  - Every frame also gets the absolute position at frame 9 and a learned frame embedding.
- **Temporal blocks:** pre-norm self-attention along the 30 frames of each pedestrian (4 heads), followed by a residual MLP (64 -> 128 -> 64, SiLU).
  - Every frame sees every other frame of the same pedestrian. The learned frame embedding is the positional encoding.
  - This replaced the first version's 1D convolutions (kernel 5, two per block). Their ±16-frame receptive field meant frames 26-29 never saw the observed frames.
- **Spatial blocks:** residual message passing between pedestrians using Practical 4's corrected `NeuralMessagePassingLayer` (mean aggregation). The same weights are applied to every frame.
- **Interaction graph:**
  - Directed kNN graph with k = 8, built from the positions at frame 9 only, so no future information leaks in.
  - Edges never cross scenes.
  - The graph is built per scene in the dataset's `make_edge_index`, replacing the template's fully connected graph, and the model reads it from `batch.edge_index`.
  - Edge features are the neighbour's relative position and its distance.
  - For the observed frames, the edge features come from the true positions. For the future frames, they come from the positions implied by `x_tau` (`p^9 + dt * cumsum(v(x_tau))`), recomputed in every forward pass. The neighbour set itself stays the one of frame 9.
- **Flow-time conditioning:** sinusoidal tau embedding, then an MLP, added before every block. tau is sampled once per scene.
- **Size:** hidden width 64, 4 temporal + spatial blocks, 264,258 parameters.
- **No EGNN:** we chose temporal attention plus plain message passing instead.

## Training and sampling (identical for both priors)

- Conditional flow matching with the linear path `x_tau = (1 - tau) x_0 + tau x_1`, MSE loss on `x_1 - x_0`.
- AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, gradient clipping 1.0, 30 epochs, seed 42. About 5.7 min of training per model.
- The checkpoint with the best validation loss is kept (epoch 28 for both).
- Sampling: 16 Euler steps from tau = 0 to 1, from fresh prior samples. The step count was fixed before the Euler step sweep (bonus 6) and was not tuned on it.
- Data: 10,000 of the 100,000 scenes (the notebook's dev default), split 7000 / 1000 / 2000 with seed 42.

## Part I: uninformed prior

- `x_0 ~ N(0, sigma^2 I)` with `sigma = 1.0` in normalized velocity units, independent of the observed frames.

## Part II: informed prior

- `x_0 = gamma_k * v^9 + sigma * eps`: every pedestrian starts from its last observed velocity, damped more for frames further ahead.
- `gamma_k` is a least-squares fit per future frame on the training split, going from 0.80 down to 0.40.
  - Among priors of this form, this minimizes the transport cost `E||x_0 - x_1||^2`.
  - `gamma = 0` is the uninformed prior, so the cost can only go down.
- The prior uses only the observed frames, stays stochastic, and is sampled the same way in training and inference.
- Same sigma, model, training budget and sampler as Part I; only `sample_prior` changes.

## Evaluation

- 2000 test scenes, M = 20 futures each, metrics on frames 10-29 only.
- **Pointwise metrics:** ADE, FDE, min20 ADE and min20 FDE.
  - The min is taken per scene (averaged over its pedestrians), not per pedestrian.
- **Distributional metrics:** histograms of speed, acceleration, tortuosity and closest-pedestrian distance.
  - Velocities and accelerations are backward differences, so frame 10 uses the observed frames 8 and 9.

## Main results

| | ADE [m] | FDE [m] | min20 ADE [m] | min20 FDE [m] |
|---|---|---|---|---|
| Uninformed | **0.914** | **1.850** | **0.644** | **1.217** |
| Informed | 0.919 | 1.860 | 0.646 | 1.221 |
| *First version (convolutions), uninformed* | *0.971* | *2.069* | *0.668* | *1.307* |
| *First version (convolutions), informed* | *0.927* | *1.869* | *0.657* | *1.244* |

- In the final version the two priors are practically equal. The uninformed prior is ahead by 0.3-0.5%, which is within single-seed noise.
- In the first version (convolutions), the informed prior was clearly better.
- Switching to attention improved the uninformed model a lot (FDE 2.07 -> 1.85 m) and the informed one only a little. So most of the informed advantage came from passing `v^9` to frames 26-29, which the convolutions could not reach.
- The two models' histograms are almost the same (see the binned KL in bonus 3).

## Bonus: extended evaluation

The assignment gives bonus points for extending the evaluation with more informative measures. Notebook cell 20 defines the functions and cell 21 runs them. All plots are in `assignments/part2_outputs/`.

### 1. Error over the prediction horizon (`bonus1_error_over_horizon.png`)

- **What:** the mean displacement error at every future frame, with a 25th-75th percentile band across scenes. Averaging over frames gives ADE, and the last frame gives FDE.
- **Why:** ADE hides *where* in the horizon a model gains or loses.
- **Result:**
  - The two curves lie on top of each other over the whole horizon: at frame 29, 1.85 m (uninformed) against 1.86 m (informed).
  - In the first version (convolutions), the curves split after about frame 20, and the informed prior ended at 1.87 m against 2.07 m.

### 2. Distance travelled and kinematics over time (`bonus2_distance_and_time_profiles.png`)

- **What:**
  - A histogram of the path length over frames 9 -> 29.
  - The median and 25th-75th percentile band of speed and acceleration per frame.
- **Why:** the required histograms pool all frames, so they cannot show drift over the horizon.
- **Result:**
  - The mean distance travelled is 4.62 m for the ground truth, 4.35 m uninformed and 4.34 m informed. Both models move pedestrians slightly too little.
  - Both models are smoother than the data: their median speed is about 0.11 m/s against 0.135 m/s for the ground truth, and their median acceleration about 0.25 m/s² against 0.32 m/s².
  - For the generated futures, the upper speed quantile drops over the horizon, while the ground truth stays wider.
  - There is no drift in the medians.

### 3. Binned KL divergence (printed table)

- **What:** `KL(model || ground truth)` for the four required histograms. The method follows Practical 3:
  - 60 equal-width bins over the central 99% of the ground truth, plus two overflow bins.
  - A pseudo count on the ground-truth bins.
  - 0-2 m for the closest-neighbour distance.
- **Why:** turns the histogram comparison into numbers.

| KL [nats] | speed | acceleration | tortuosity | closest |
|---|---|---|---|---|
| Uninformed | 0.0210 | 0.0880 | 0.0133 | 0.0322 |
| Informed | 0.0197 | 0.0879 | 0.0193 | 0.0328 |

- **Result:**
  - Neither prior is consistently better: the informed prior is slightly better on speed, the uninformed one on tortuosity.
  - Acceleration is the worst-matched quantity, slightly worse than in the first version (0.075).
  - Part of that is the float16 position quantization in the dataset, which makes the ground-truth acceleration histogram spiky. Positions along the platform are rounded to 3.1 cm, so accelerations come in steps of 0.125 m/s².

### 4. Collision rate (`bonus4_collision_rate.png`)

- **What:** the fraction of pedestrian-frames whose nearest neighbour (same scene, same future) is closer than r.
  - Reported at r = 0.3 m.
  - Also plotted as a function of r, so the conclusion does not depend on one threshold.
- **Why:** a physically meaningful check of the interaction modelling that a closest-distance histogram only shows indirectly.
- **Result:**
  - At 0.3 m: ground truth 0.64%, uninformed 2.57%, informed 2.61%, so the models collide about 4x more often.
  - The gap holds for all r below about 0.8 m.
  - The prior makes no difference here.
  - The future edge features from `x_tau` did not help: the first version, which reused the frame-9 offsets, had the same 2.57-2.58%. For small tau the implied positions are mostly noise, and the neighbour set is still fixed at frame 9.

### 5. Transport cost of each prior (`bonus5_transport_cost.png`)

- **What:** a Monte Carlo estimate of `E||x_0 - x_1||^2` per pedestrian, as in Practical 4.
  - Computed on the training split, because it motivates the prior.
  - Compared with the noise floor `2 K sigma^2 = 40`.
- **Why:** this is the quantity the informed prior is designed to reduce, so it checks that the prior does what it should.
- **Result:**
  - Uninformed 77.5 (37.5 above the floor), informed 65.5 (25.5 above the floor): about 1/3 less transport above the noise floor.
  - Per frame, the reduction is largest at frame 10 (2.75 against 4.0) and shrinks to about 3.5 against 3.8 at frame 29, as `gamma_k` decays.
  - The prior is the same in both versions, so these numbers hold for both. In the final version, the lower transport cost does not translate into lower errors (bonus 1).

### 6. Euler step sweep (`bonus6_euler_step_sweep.png`)

- **What:** ADE, min20 ADE, KL speed and KL acceleration for 1, 2, 4, ..., 64 Euler steps.
  - Run on 200 test scenes, with the same seed for every step count.
  - It is diagnostic only; the main results use 16 steps, chosen beforehand.
- **Why:** shows the trade-off between sample cost and sample quality, and whether 16 steps is enough.
- **Result:**
  - With 1-2 steps, ADE is lowest (about 0.70 m), but the samples are collapsed towards the conditional mean: KL speed is about 0.3-0.4 and KL acceleration about 0.9-1.6.
  - With more steps, the KL drops to about 0.01 (speed) and 0.05 (acceleration) and has converged by 16-32 steps. ADE rises because the samples become realistically spread out.
  - Pointwise error alone would favour too few steps, which is why we judge quality with the distributional metrics.
  - The uninformed prior has a slightly lower ADE at every step count, so the shorter transport of the informed prior does not pay off even with few steps.

## Limitations and possible improvements

- Train on more than 10k scenes.
- Rebuild the neighbour set (not only the edge features) from the generated positions, or add a repulsion term, to reduce collisions.
- Train several seeds per model, to tell whether the 0.5% difference between the priors is real.
- Float16 positions in the dataset limit how well the acceleration statistics can be matched.
