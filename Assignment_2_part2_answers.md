# Assignment 2, Part 2: Pedestrians on Eindhoven Centraal, answers

> Notes for me (delete before submitting):
> - Model numbers and plots are from the final Snellius run of 12 Oct (job 27956341, 1x A100): temporal attention and future edge features from x_τ. It is saved in `assignments/Assignment_2_part2.ipynb`, with the plots and checkpoints in `assignments/part2_outputs/`.
> - The first version (1D convolutions over time, frame-9 edge features, jobs 27780417 and 27948583) is used as a comparison in 12a, 12c, 13b, 14a and 14c. Its numbers: ADE/FDE/min20 ADE/min20 FDE uninformed 0.971/2.069/0.668/1.307, informed 0.927/1.869/0.657/1.244.
> - The data analysis figures (9e) come from Snellius job 27944192. The loss curves and timing (13b, 13c, 12d) come from the final run. All extra figures are in `assignments/part2_outputs/ans_figures/`.
> - 9c was not in the screenshots. I assumed it is the "explain why" question that 9b refers to.
> - Word limits: 9a max 200, 9d and 9e max 500, 12b max 1000.
---

## 9a. Contents of the dataset (1p)

`eindhoven_centraal.pkl` is a pickled Python list of 100,000 NumPy arrays. Every array is one scene (a data sample) with dtype float16 and shape (N, 30, 2): N pedestrians, 30 frames, 2 coordinates.

- N is between 5 and 100 pedestrians per scene (mean 23.6, median 17), so the samples have different sizes. In total there are 2,358,008 pedestrian trajectories, which is about 283 MB of float16 positions.
- The values are positions in metres in a fixed coordinate frame of the station. The first coordinate runs across the platform (from -9.8 to 4.1 m) and the second along the platform (from -5.3 to 75.3 m).
- Frames are Δt = 0.5 s apart, so a scene covers 15 s. Every pedestrian is present in all 30 frames, and there are no missing values (NaNs).

The data represents tracked pedestrians walking and waiting on platforms 3 and 4 of Eindhoven Centraal. In the task, frames 0-9 are observed and frames 10-29 have to be generated. The data loader converts each array to a float32 tensor and wraps it in a PyG `Data` object.

---

## 9b. Relevant symmetries (2p)

Selected: **Permutation of information along the pedestrian indices.**

Not selected: rotation, reflection and translation of positions, and permutation along the time indices.

## 9c. Why these symmetries (assumed question)

**Pedestrian permutation:** The order of the pedestrians in an array is just the order of the tracker IDs. If we shuffle the rows of (N, 30, 2), it is still exactly the same scene. A model should therefore be permutation equivariant: if the inputs are permuted, the outputs are permuted in the same way.

**Rotation, reflection and translation:** These are not symmetries of this dataset, because the positions are absolute coordinates in one fixed station. The platform has walls, track edges, stairs on the left, a building block in the middle and benches. The occupancy map in 9e shows that pedestrians only walk in specific lanes and almost always move along the platform. If we rotate a scene by 90 degrees, translate it by 10 m or mirror it, pedestrians end up walking through walls or onto the tracks, so the sample no longer represents what happens on this platform. The platform is also not mirror-symmetric (stairs on one side, a different exit on the other). The interaction between two pedestrians on its own is roughly invariant to these transformations, which is why we describe neighbours with relative offsets (see 10a). The full sample, however, is not invariant.

**Time permutation:** Reordering the frames destroys the trajectories: velocities and accelerations come from consecutive frames, and the task is to predict the future from the past. So time order matters.

---

## 9d. Properties of the dynamical system (2p)

Notation: p_i^t ∈ R² is the position of pedestrian i at frame t, with t = 0, ..., 29, Δt = 0.5 s and N_s pedestrians in scene s.

1. **Scene size** N_s. The number of pedestrians changes per sample, so the model has to work on sets of variable size.
2. **Velocity and speed:** v_i^t = (p_i^t − p_i^{t−1}) / Δt and the speed ‖v_i^t‖₂. This shows whether people walk or wait, and it is the quantity our model generates.
3. **Acceleration:** a_i^t = (v_i^t − v_i^{t−1}) / Δt and ‖a_i^t‖₂. Measures how smooth the motion is and how quickly people change pace or direction.
4. **Heading:** θ_i^t = atan2(v_{i,y}^t, v_{i,x}^t), only for moving steps (‖v_i^t‖ > 0.5 m/s). Shows whether there are preferred walking directions, which tells us whether rotation is a symmetry (9c).
5. **Velocity memory:** the correlation ρ(k) = corr(v_i^{9+k}, v_i^9) for k = 1, ..., 20 (per coordinate, normalized velocities, averaged over the two coordinates), and the least-squares coefficient γ_k = Σ_i v_i^{9+k} · v_i^9 / Σ_i ‖v_i^9‖². This measures how much the last observed velocity says about the future, which is the basis of our informed prior.
6. **Tortuosity:** χ_i = Σ_{t=10}^{29} ‖p_i^t − p_i^{t−1}‖ / max(‖p_i^{29} − p_i^9‖, 0.01). Measures how straight the paths are.
7. **Distance to the closest pedestrian:** d_i^t = min_{j≠i} ‖p_i^t − p_j^t‖. Shows personal space and how often near-collisions happen.
8. **Local density:** n_i = |{ j ≠ i : ‖p_i^9 − p_j^9‖ < 2 m }|, and the distance to the 8th nearest neighbour at frame 9. Tells how many pedestrians are close enough to interact, which drives the choice of the interaction graph (10b).
9. **Stationarity in time:** the mean speed per frame, (1/P) Σ_i ‖v_i^t‖, for every t. Checks whether the statistics change over the 15 s window.
10. **Spatial occupancy and mean velocity field:** histograms of p_i^t over the platform and the mean of v_i^t per spatial cell. Shows where people are and where they go, so how much the absolute position tells about the motion.

These properties cover the three axes of the problem: motion of individual pedestrians (2, 3, 4, 5, 6), interactions (7, 8) and the effect of the environment (4, 10). Properties 2, 3, 6 and 7 are also the distributional metrics we use to evaluate the generated futures, so the analysis tells us what the ground truth looks like before we compare models against it.

---

## 9e. Exploratory data analysis plots (2p)

Upload: `assignments/part2_outputs/ans_figures/9e_eda_overview.png` and `assignments/part2_outputs/ans_figures/9e_occupancy_velocity_map.png`

(Computed on all 100,000 scenes, except the neighbour statistics, which use 10,000 random scenes.)

**Scene size:** Most scenes have 5-20 pedestrians, with a long tail up to 100, so graph sizes in a batch vary a lot.

**Two groups of pedestrians:** The speed histogram is bimodal. 43.6% of all steps are slower than 0.1 m/s (people waiting for a train), and there is a second peak around 1.2-1.4 m/s (normal walking speed). 54% of the pedestrians have a mean speed below 0.2 m/s and 28.5% are above 0.8 m/s. The future is therefore multimodal: a waiting person may stay or start walking, and the model has to represent both.

**Strong directional structure:** 87.5% of moving steps are within 30 degrees of the platform axis. The velocity spread is 0.70 m/s along the platform and only 0.20 m/s across it. The map shows walking lanes on both sides of the central building, flows to and from the stairs on the left, and dense waiting spots (for example at x ≈ 60 m). So the absolute position carries a lot of information about where a pedestrian will go.

**Velocity memory decays slowly:** The correlation between v^9 and v^{9+k} is 0.80 one frame ahead and still 0.44 after 20 frames (10 s). The least-squares γ_k follows the same curve (0.79 → 0.39). So v^9 is a strong predictor, but it must be damped with k.

**Sparse interactions:** The median distance to the closest pedestrian is 1.45 m. Only 0.29% of the pedestrian-frames are closer than 0.3 m and 3% are closer than 0.5 m, so people keep a personal space of about 0.5-0.7 m. At frame 9, a pedestrian has on average only 1.4 others within 2 m, while the 8th nearest neighbour is a median 7.6 m away. Most pairs in a scene do not interact.

**Tortuosity:** The median is 3.5 (57% have χ > 2). This comes from waiting pedestrians, where jitter of a few centimetres on almost zero net displacement gives a large ratio. Walkers have χ ≈ 1.

**Data quality:**
- The positions are float16. Along the platform (positions of 32-64 m), the values are rounded to steps of 3.1 cm, so velocities come in steps of 0.0625 m/s and accelerations in steps of 0.125 m/s². This explains the spiky acceleration histogram.
- A few speeds are impossible (up to 47 m/s), probably tracker ID switches.
- The mean speed per frame only drifts from 0.49 to 0.45 m/s over the 30 frames, so the data is close to stationary in time.

**Modelling challenges:**
- A stochastic, multimodal future (waiting versus walking).
- A variable number of pedestrians.
- Interactions that are local and sparse, but where near-collisions must stay rare.
- A strong dependence on absolute position, so rotation and translation equivariance would hurt.
- Information that fades over a 10 s horizon.
- Quantization noise and outliers in the targets.

---

## 10a. Data structure and features (2p)

**Data structure:** Every scene is a PyTorch Geometric `Data` object. The nodes are pedestrians, and the node features `x` hold the full trajectories with shape (N, 30, 2) in float32. We replaced the template's fully connected graph in `make_edge_index` with a kNN graph built from the last observed frame (see 10b), so `edge_index` holds the interaction graph of every scene. Scenes are batched with the PyG `DataLoader` (16 scenes per batch), which gives:

- `x`: (ΣN, 30, 2)
- `edge_index`: (2, ΣE), offset per scene by PyG, so edges never connect different scenes
- `batch`: (ΣN,), the scene index of every pedestrian
- `ptr`: the start index of every scene

Only frames 0-9 of `x` are used as input. Frames 10-29 are only used as the training target and for evaluation.

**Normalization:** All statistics are computed on the 7000 training scenes only: the mean and standard deviation of the velocity, the standard deviation of positions relative to frame 9, and the mean and standard deviation of the absolute position at frame 9.

**Features we compute from the positions** (in `PedestrianFlowModel.condition` and `SpaceTimeVelocityField.forward`):

- **Velocities:** v^t = (p^t − p^{t−1}) / Δt for t = 1, ..., 9, normalized with the training mean and std. v^0 cannot be computed and is set to 0.
- **Relative positions:** (p^t − p^9) / std for t = 0, ..., 9. The observed track is expressed relative to where the pedestrian is now.
- **Observed flag:** 1 for frames 0-9 and 0 for frames 10-29, so the network knows which frames are given and which are generated.
- **Absolute position at frame 9,** normalized, given to every frame of the pedestrian. It tells the model where on the platform the pedestrian is (lanes, stairs, waiting areas).
- **Last velocity v^9** (normalized), used by the informed prior.
- **Edge features** for every edge j → i and frame t: the offset (p_j^t − p_i^t) / std and the distance ‖p_j^t − p_i^t‖ in metres.
  - For t = 0, ..., 9 they come from the observed positions.
  - For the future frames t = 10, ..., 29 the true positions are unknown. We use the positions implied by the current noisy state: p̂^{9+k} = p^9 + Δt Σ_{j≤k} v^{9+j}(x_τ), recomputed in every forward pass.
- **Target:** the normalized future velocities v^10, ..., v^29, shape (N, 20, 2). The model generates velocities, not positions. Positions are recovered with p^{9+k} = p^9 + Δt Σ_{j≤k} v^{9+j}.

So the input of the network is a grid of pedestrians × 30 frames. Each token has 7 features: relative position (2), velocity (2), observed flag (1) and absolute position at frame 9 (2). In the 20 future tokens, the relative-position slots and the flag are 0, and the velocity slots hold the noisy flow-matching state x_τ.

---

## 10b. Modelling interactions between pedestrians (2p)

**How:** Each pedestrian receives messages from its k = 8 nearest neighbours in the same scene. The directed kNN graph j → i is built per scene in the dataset's `make_edge_index`, with `torch.cdist` on the positions at frame 9, the last observed frame, so no future information is used. Scenes with fewer than 9 pedestrians get k = N − 1. The messages use Practical 4's `NeuralMessagePassingLayer`: an MLP on [h_j^t, h_i^t, e_ij^t], mean aggregation over the neighbours, and a node MLP on [h_i^t, m_i^t].

**When:**
- The same set of neighbours is used in every frame (0-29) and in every one of the 4 spatial blocks. Each spatial block comes after a temporal attention block.
- Interactions are modelled within a frame: pedestrian i at frame t talks to its neighbours at the same frame t. Between two spatial blocks, the temporal attention mixes information along the frames of each pedestrian.
- The edge features change per frame. In the observed frames they are the true offsets. In the generated frames they are the offsets between the positions implied by x_τ, so they are updated at every Euler step during sampling. The neighbour set itself is built once per sample when the data is loaded.

**Expressiveness:**
- Four blocks give a 4-hop receptive field, so information can travel through chains of neighbours. Together with the temporal blocks, a pedestrian can react to what a neighbour did in any other frame.
- The edge features contain the relative offset and the distance, so the messages can depend on how close and in which direction a neighbour is, also in the generated future.
- Interactions are limited to the 8 nearest neighbours at frame 9. From 9e, these are often far away (median 7.6 m for the 8th neighbour) while only 1.4 pedestrians are within 2 m. The relevant neighbours are almost always included, but mean aggregation also mixes in messages from people who do not interact.
- A pedestrian who is not among the 8 nearest at frame 9 but comes close later is never connected.

**Computational complexity:** The number of edges is at most kN instead of N(N−1) for the fully connected template graph. Every spatial block computes messages for every edge in every frame, so one block costs O(T·k·N·d) instead of O(T·N²·d).
- For a scene with 100 pedestrians, that is 800 instead of 9,900 edges per frame.
- For 1000 pedestrians it would be 8,000 instead of about 1,000,000, which is what makes large scenes feasible (12d).

Building the graph costs O(N²) distance computations, but only once per sample and with almost no feature dimension, so it is negligible. Recomputing the future edge features costs O(K·k·N) per forward pass, which is small next to the message MLPs.

---

## 11a. Why a probabilistic model, and how this data differs from Part 1 (2p)

**Why probabilistic:**
- The future of a pedestrian is not determined by the 10 observed frames. Intentions are not observed: where someone wants to go, when their train arrives, whether a waiting person starts to walk. From 9e, more than half of the pedestrians are standing still, and from the same 5 s of standing they can stay or leave in either direction along the platform.
- So the true future is a distribution with several modes. A deterministic model trained with MSE predicts the conditional mean, which for "go left or go right" is "stand still", a trajectory nobody would actually walk.
- A generative model can sample several plausible futures, which is also what min20 ADE/FDE and the distributional metrics evaluate. It also gives a measure of uncertainty, which matters for uses such as crowd management on a platform.

**How this differs from Part 1 (cylinder flow):**
- The flow simulation is a deterministic physical system: the Navier-Stokes equations fix the next state given the current one, so one prediction per input is enough.
- Pedestrians are agents with their own unobserved goals and no known governing equation, so the future is uncertain.
- The flow data lives on a fixed mesh with the same nodes in every sample. The pedestrian scenes are sets of 5 to 100 agents that move around, so the graph changes per sample and over time.
- The pedestrian state is only partially observed (positions only, no goals), and the positions are quantized and noisy.
- The interactions are sparse and short-range, and the environment (walls, stairs) strongly breaks the symmetries.

---

## 11b. Uninformed and informed prior (2p)

**Uninformed prior:** x_0 ~ N(0, σ² I), with x_0 ∈ R^{N×20×2} in normalized velocity units and σ = 1. It does not depend on the observed frames.

**Informed prior:** x_{0,i}^k = γ_k · v̂_i^9 + σ ε_i^k with ε ~ N(0, I), for k = 1, ..., 20 (future frame 9+k). Here v̂_i^9 is the normalized last observed velocity of pedestrian i. The coefficients γ_k come from a least-squares fit on the training split:

γ_k = Σ_i v̂_i^{9+k} · v̂_i^9 / Σ_i ‖v̂_i^9‖²

They go from 0.80 for frame 10 down to 0.40 for frame 29. In words: every pedestrian starts from its last observed velocity, damped towards the mean velocity the further ahead the frame is. The noise level σ, the model, the training and the sampler are identical to the uninformed version.

**Why it should help in theory:**
- In flow matching, the network learns the velocity field that moves samples from x_0 to x_1 along the line x_τ = (1−τ)x_0 + τ x_1, with target x_1 − x_0. The closer the prior samples are to the data, the shorter these paths are.
- Our γ_k minimize E‖x_0 − x_1‖² among all priors of the form γ_k v̂^9 + σε, and γ = 0 is exactly the uninformed prior, so the transport cost cannot go up. On the training split it drops from 77.5 to 65.5 per pedestrian, against a noise floor of 2Kσ² = 40. That is a third less transport above the noise.
- Shorter, less crossing paths give a simpler, straighter velocity field. That field is easier to learn with the same budget and has a smaller discretization error with a fixed number of Euler steps (16).
- The network only has to learn a correction on top of a physically sensible guess (pedestrians keep moving the way they move now), instead of building the whole future from pure noise.
- The prior stays stochastic through ε, so the model can still produce several futures.

---

## 11c. Inputs, outputs and the learned task (1p)

**Inputs:**
1. The observed positions of all pedestrians in a scene for frames 0-9. From these we compute the relative positions, velocities, absolute position at frame 9, the kNN graph and the observed edge features (10a).
2. The current noisy state x_τ of the future velocities, shape (N, 20, 2).
3. The flow time τ ∈ [0, 1], one value per scene.

**Output:** a velocity field v_θ(x_τ, τ | observed frames) of shape (N, 20, 2): for every pedestrian and every future frame, the direction in which x_τ should move.

**In simple terms:** The model learns to turn a rough random guess of how everyone will walk into a realistic future, step by step. At τ = 0 we start from noise (or from "keep your current velocity, plus noise" for the informed prior). The model repeatedly says "move this guess a bit in this direction", taking into account each person's past motion, their position on the platform and where their neighbours are going according to the current guess. After 16 steps (τ = 1), the guess is a plausible set of velocities for the next 10 s, which we integrate into positions. Different starting noise gives different plausible futures.

---

## 11d. Time bundling: advantages, disadvantages and this dataset (3p)

**In general.** With time bundling, the surrogate predicts K future steps in one model call instead of one step per call.

Advantages:
- **Fewer model calls:** K times fewer sequential calls at inference, so it is faster and the calls are easier to parallelize.
- **Less error accumulation:** an autoregressive model feeds its own predictions back in, so small errors are amplified at every step (e_t ≈ J_t e_{t−1}). A bundled model only rolls out once per K steps.
- **Coherent trajectories:** the frames of a bundle are predicted jointly, so the model sees the whole bundle at once and can make the frames consistent with each other.
- **Training matches inference:** the model is trained on what it outputs at test time, without the train/test mismatch of feeding back its own predictions.

Disadvantages:
- **Harder learning problem:** the output is K times larger and frames far ahead are harder to predict from the same input.
- **Fixed horizon:** horizons longer than K need a rollout of bundles anyway, and other horizon lengths do not fit naturally.
- **No feedback:** the model cannot react to its own intermediate predictions. Within a bundle, it cannot use the predicted state at step t to decide step t+1, which is natural in an autoregressive model.
- **Higher memory use** per forward pass.

**For this dataset, time bundling is more effective than one-step autoregressive modelling:**
- **Cost:** a generative model must be integrated over τ at every prediction step. One-step autoregressive flow matching would need 20 sequential generations × 16 Euler steps = 320 forward passes per sample, against 16 for our bundled model. That is about 2.4 s instead of about 0.12 s per minibatch (13c).
- **Error accumulation:** each autoregressive step would sample new noise and feed its own sample back. Errors and sampling noise in the velocities would compound over 20 steps, and the position errors already grow with the horizon because positions integrate velocities.
- **The horizon fits in one bundle:** the horizon is short (20 frames, 10 s) and from 9e the motion is smooth with slowly decaying velocity memory, so one bundle covers it without a rollout.
- **Joint sampling stays consistent:** a pedestrian who starts walking at frame 15 keeps walking. Separate stochastic steps could switch between modes from one frame to the next.

The main disadvantage for this dataset is interactions in the future. An autoregressive model can rebuild the neighbour graph from the predicted positions after every step. Inside a bundle we approximate this by computing the future edge features from the positions implied by x_τ, but the neighbour set stays the one of frame 9, and early in sampling the implied positions are mostly noise. The collision rate stays about 4× too high (14b).

---

## 12a. Convolutional, transformer and graph-based architectures for this data (2p)

**Convolutional:**
- **2D convolution on a rasterized occupancy or velocity grid of the platform:** it works with the fixed geometry, but it loses the identity of individual pedestrians. You can no longer say where pedestrian i goes. It also needs a fine grid for 0.5 m personal space over a 80 × 15 m platform, and most cells are empty. Not effective.
- **1D convolution along the time axis of each pedestrian:** reasonable. A trajectory is a regular sequence with a fixed Δt, local patterns in time (velocity, acceleration, turning) are translation-invariant in time, the cost is linear in T, and the weights are shared across pedestrians, so it handles any N and is permutation equivariant. Its receptive field grows only with depth, though. Our first version with 4 blocks of two kernel-5 convolutions reached ±16 frames, so frames 26-29 could not see the observed frames. On its own, it also sees no other pedestrians.

**Transformer:**
- **Attention over time per pedestrian** (30 tokens): effective. It has a global temporal receptive field, so frame 29 attends directly to the observed frames, and 30 tokens is cheap. This is what we use.
- **Attention over pedestrians within a frame** (set transformer without positional encoding): permutation equivariant and handles variable N. It is O(N²) and attends to everyone in the scene, while 9e shows that most pairs do not interact. Still fine for N ≤ 100, but not for 1000 pedestrians.
- **Full space-time attention over N × 30 tokens:** up to 3000 tokens, so 9 million attention scores per head. Too expensive and not needed.

**Graph-based:**
- **Message passing over a kNN or radius graph of pedestrians:** effective for interactions. It is permutation equivariant, works for any N, focuses on local neighbours (personal space and collision avoidance are local), and costs O(kN) instead of O(N²).
- **Time:** as discussed in Practical 4, message passing alone does not model time. Putting the frames in the node features and processing them with an MLP ignores the order and locality of the frames.

So graph-based models fit the space axis and attention (or convolutions) fit the time axis. The most effective choice is a combination.

---

## 12b. Implementation and forward pass (5p)

The network is `SpaceTimeVelocityField` (hidden size d = 64, 4 blocks, 4 attention heads, 264,258 parameters in total). It is wrapped in `PedestrianFlowModel`, a Lightning module that builds the conditioning, samples the prior, computes the loss and runs the sampler. A forward pass computes v_θ(x_τ, τ | c) for all pedestrians of a batch.

**Step 1, conditioning (observed part).** `condition(batch)` only reads `batch.x[:, :10]`. For every pedestrian it computes the normalized velocities and the positions relative to frame 9 for frames 0-9, plus an observed flag of 1:

```python
observed = torch.cat([rel, vel_n, torch.ones_like(p[..., :1])], dim=-1)  # (ΣN, 10, 5)
```

It takes the kNN graph from `batch.edge_index` (built by `make_edge_index` from frame 9, k = 8) and computes the edge features of the observed frames: the offset to the neighbour and its distance. It also returns the normalized absolute position at frame 9, p^9 in metres and v̂^9. During sampling, the conditioning is computed once and reused for all Euler steps.

**Step 2, building the token grid.** The noisy future velocities x_τ (ΣN, 20, 2) become the 20 future tokens, with zeros in the relative-position slots and the flag:

```python
future = torch.cat([zeros(n, K, 2), x_tau, zeros(n, K, 1)], dim=-1)        # (ΣN, 20, 5)
frames = torch.cat([cond["observed"], future], dim=1)                      # (ΣN, 30, 5)
static = cond["position"][:, None].expand(-1, 30, -1)                      # (ΣN, 30, 2)
h = self.input(torch.cat([frames, static], dim=-1)) + self.frame_embedding # (ΣN, 30, 64)
```

Every pedestrian is one sequence of 30 tokens. The observed and generated parts go through the same layers and are only told apart by their features. A linear layer maps the 7 features to 64, and a learned embedding per frame index tells each token which frame it is.

**Step 3, future edge features.** From x_τ we compute where everyone would be in frames 10-29, and from that the offsets and distances to the neighbours:

```python
vel = x_tau * cond["vel_std"] + cond["vel_mean"]                 # m/s
p_future = cond["p9"][:, None] + DT * torch.cumsum(vel, dim=1)   # (ΣN, 20, 2)
offset = p_future[src] - p_future[dst]                           # (E, 20, 2)
```

They are concatenated with the observed edge features into (E, 30, 3).

**Step 4, processing τ.** τ (one value per scene) gets a sinusoidal embedding with 16 frequencies between 1 and 1000 (32 values), and an MLP (Linear, SiLU, Linear) maps it to 64 features. Before every block, a separate linear layer per block projects it, and the result is added to all 30 tokens of the pedestrian:

```python
for temporal, spatial, tau_proj in zip(...):
    h = h + tau_proj(t)[:, None]
    h = temporal(h)
    h = spatial(h, cond["edge_index"], edge_attr)
```

**Step 5, time: temporal block.** `TemporalBlock` is a pre-norm Transformer layer over the 30 frames of one pedestrian: LayerNorm, multi-head self-attention (4 heads), residual, then LayerNorm, MLP (64 → 128 → 64, SiLU), residual. The pedestrians are the batch dimension, so a pedestrian only attends to its own frames. There is no causal mask: every frame attends to every other frame, so all future tokens read the observed motion directly, and the observed tokens see the current guess of the future.

**Step 6, space: spatial block.** `SpatialBlock` mixes information between pedestrians within the same frame. After LayerNorm, the (ΣN, 30, 64) tensor is flattened into 30·ΣN nodes. The kNN edges are copied 30 times with an offset per frame, so pedestrian i at frame t is only connected to its neighbours at frame t, with that frame's edge features:

```python
y = self.norm(h).transpose(0, 1).reshape(f * n, d)              # node (i, t) -> t*n + i
frame_edges = edge_index.repeat(1, f) + offsets                 # same graph in every frame
y = self.mp(y, frame_edges, frame_attr)
return h + y.reshape(f, n, d).transpose(0, 1)
```

The layer computes m_ij^t = MLP([h_j^t, h_i^t, e_ij^t]), averages the messages per receiving node, and updates with a node MLP on [h_i^t, mean_j m_ij^t], with a residual connection. The weights are shared over all frames and pedestrians.

**Combining space and time.** The temporal and spatial blocks alternate 4 times. After a temporal block, every token contains information from all frames of its pedestrian. The next spatial block passes that to the neighbours, so a future token can depend on a neighbour's observed velocity.

**Step 7, output.** Only the 20 future tokens are read out. A head (LayerNorm, Linear, SiLU, Linear) maps each to 2 values, giving v_θ of shape (ΣN, 20, 2).

**Training and sampling.** Training draws x_0 from the prior and one τ ~ U(0, 1) per scene, sets x_τ = (1−τ)x_0 + τx_1 and minimizes MSE(v_θ, x_1 − x_0). Sampling takes 16 Euler steps x ← x + v_θ(x, i/16, c) / 16 from a new prior sample, then integrates the velocities to positions from p^9.

---

## 12c. Motivation of the architecture and alternatives (5p)

**Separate time and space blocks instead of one big layer.** The data has two different axes. Along time, every pedestrian has a regular sequence of 30 frames. Across pedestrians, the scene is an unordered set of variable size with sparse, local interactions (9e). Using a layer that fits each axis is more efficient than treating all N × 30 tokens the same, and alternating them lets information flow between both axes.

**Self-attention over time:**
- Each pedestrian has only 30 frames, so the attention matrix is 30 × 30 and the quadratic cost of attention, which made full attention expensive on the grids of Practical 1 and Assignment 1, does not matter here.
- In exchange, every frame sees every other frame in one layer. The future frames need the observed velocity (the velocity memory of 9e lasts the full 10 s horizon), and the last generated frame is 20 frames away from the last observed one.
- We first used 1D convolutions over time (4 blocks of two kernel-5 convolutions). Their receptive field was ±16 frames, so frames 26-29 could not see the observed frames at all. Replacing them with attention lowered the uninformed model's FDE from 2.07 m to 1.85 m with 10% fewer parameters (14a).
- Attention is not causal on purpose. For generation, the whole sequence is available at once (time bundling), so future tokens may use each other and the observed past. The learned frame embedding gives the order of the frames, as the positional features did for the Transformer in Practical 1.
- Attention is permutation equivariant over frames, so without the frame embedding it would not know the order. The embedding is cheap and also tells each token how far ahead it is, which matters because the uncertainty grows with the horizon.

**Message passing over a kNN graph for interactions:**
- It is permutation equivariant and works for any number of pedestrians. Only local neighbours matter for personal space and avoidance, and the cost grows with kN instead of N².
- We reuse Practical 4's corrected message passing layer: messages from [h_j, h_i, e_ij] and a node update on [h_i, mean of messages]. It is a general and well-tested layer.
- Mean aggregation keeps the scale of the messages the same for pedestrians with fewer neighbours, for example in small scenes.
- **kNN instead of a radius graph:** every pedestrian always gets some context, even in empty areas, and the number of edges is bounded, so memory is predictable.
- **Graph from frame 9:** this is the last frame that is known at inference, so nothing leaks from the future.
- **Future edge features from x_τ:** without them, the generated frames would only see the neighbours' frame-9 offsets, and the model would not see two pedestrians getting closer in the future it is generating.

**Relative features plus the absolute position.** Neighbours are described by relative offsets and distances, and each pedestrian's own track is relative to its position at frame 9. That makes the interaction part independent of where on the platform it happens. The absolute position at frame 9 is still given to every token, because the platform is not symmetric (9c): lanes, stairs and waiting areas depend on the position. We therefore deliberately did not make the model rotation or translation equivariant.

**Generating velocities instead of positions.** The velocities have a similar scale in every frame and are close to stationary (9e). Positions grow over the horizon. With velocities, the informed prior "keep your velocity" is simple to write, and integrating guarantees continuous trajectories that start at p^9.

**Residual pre-norm blocks and a τ embedding added per block.** These are standard choices for stable training of diffusion and flow models. The τ embedding lets every block behave differently for nearly pure noise (τ ≈ 0) and nearly clean data (τ ≈ 1).

**Alternatives:**
- **1D convolutions over time,** with dilation or more blocks to cover all 30 frames. Cheaper per layer, but they need depth for long-range information.
- **Attention over neighbours** (graph attention) instead of mean aggregation, so the model can weight the one pedestrian at 0.5 m much more than the one at 7 m.
- **A radius graph** (for example 2-3 m) instead of kNN, which matches the interaction range from 9e better, possibly with a minimum k.
- **Rebuilding the neighbour set** from the positions implied by x_τ, not only the edge features, so pedestrians who approach in the future get connected.
- **A recurrent encoder** (GRU or LSTM) for the observed frames with a separate decoder for the future, as in many trajectory prediction models.

---

## 12d. Scaling to 1000 pedestrians or 300 timesteps (2p)

**1000 pedestrians: yes.** The cost is linear in the number of pedestrians, because each pedestrian has at most k = 8 edges and all other layers act per pedestrian. Building the kNN graph needs a 1000 × 1000 distance matrix at frame 9 (1 million distances), which is negligible.

We measured this on one A100 with a random scene of 1000 pedestrians:

| Scene | Forward pass | Sampling (16 Euler steps) | Peak GPU memory |
|---|---|---|---|
| 1000 pedestrians | 18.4 ms | 294 ms | 0.30 GB |

For comparison, a forward pass on a normal test minibatch with about 390 pedestrians takes 6.4 ms. Training on such scenes needs more memory for the stored activations, but at 0.30 GB for inference this still fits easily on one GPU.

With the fully connected template graph, the same scene would have about 1 million edges per frame, so 30 million messages of 64 features per block. That costs several GB per layer and much more time, which is the main reason we use kNN.

**300 timesteps: probably yes, with small changes.**
- Message passing runs per frame, so that part is linear in T.
- The temporal attention is quadratic in T: 300 × 300 = 90,000 scores per head and pedestrian, against 900 now. For a typical scene of 24 pedestrians with 4 heads, that is about 8.6 million scores per layer, which a single GPU handles easily. We estimate a forward pass well below 100 ms, but we did not measure it.
- For scenes with both many pedestrians and many frames, the attention memory grows quickly, and a sliding-window or slice-based attention over time would be needed.
- The learned frame embedding only has 30 entries and would have to be extended. The graph fixed at frame 9 would also be meaningless after 145 s.

So the cost is fine, but the architecture needs small changes to stay accurate.

---

## 13a. Training procedure, hyperparameters, time and hardware (1p)

Both models are trained in exactly the same way (`train_flow_model`). Only the prior differs.

- **Data:** 10,000 of the 100,000 scenes, split at scene level with seed 42 into 7000 train, 1000 validation and 2000 test scenes. Normalization statistics come from the training split only.
- **Loss:** conditional flow matching. For each batch: a fresh prior sample x_0, one τ ~ U(0, 1) per scene, x_τ = (1−τ)x_0 + τx_1, and the MSE between v_θ and x_1 − x_0 over all pedestrians and future frames.
- **Hyperparameters:**
  - Batch size 16 scenes (438 steps per epoch), 30 epochs.
  - AdamW with lr 1e-3 and weight decay 1e-4, cosine annealing of the lr to 0 over 30 epochs.
  - Gradient clipping at norm 1.0.
  - Hidden size 64, 4 blocks, 4 attention heads, k = 8 neighbours, σ = 1, seed 42.
- **Checkpoint:** the one with the lowest validation loss, which was epoch 28 for both models.
- **Hardware and time:** Snellius, 1× NVIDIA A100 (40 GB), 18 CPU cores. Each model took about 5.7 minutes for 30 epochs (about 11 s per epoch). The whole notebook took 31 minutes, including the evaluation with 20 futures for 2000 test scenes per model and the bonus analyses.

---

## 13b. Training and validation loss curves (1p)

Upload: `assignments/part2_outputs/ans_figures/13b_loss_curves.png`

**Validation loss:** The validation loss goes down steadily for both models:

| Model | Epoch 0 | Best (epoch 28) | Epoch 29 |
|---|---|---|---|
| Uninformed | 0.770 | 0.662 | 0.662 |
| Informed | 0.800 | 0.663 | 0.664 |

Most of the gain is in the first 5 epochs. After that the loss decreases slowly, and the last few epochs, with the small learning rate at the end of the cosine schedule, still give a small improvement.

**Training loss:** The training curve is noisy because Lightning logs one batch every 50 steps, so each epoch point is a mean of only about 8 single-batch losses. Flow-matching losses also have a high variance anyway, because each batch draws new noise and new τ values. Its trend follows the validation loss, and the training loss is not systematically below the validation loss. There is no overfitting, and both models could probably improve slightly with more data or epochs.

**The loss values of the two priors are not directly comparable.** The target x_1 − x_0 depends on the prior, and the flow-matching loss contains an irreducible part that comes from how x_0 and x_1 are paired. A lower loss therefore does not mean better samples: in the convolutional version, the informed model had the higher validation loss but clearly better test metrics. That is why we judge convergence on the validation loss of each model separately, and compare the models on the sample metrics (ADE/FDE, histograms, KL). The Euler step sweep (bonus) is another check that the learned flow converges: the KL of speed and acceleration stops improving from 16-32 steps.

---

## 13c. Inference procedure, hyperparameters and runtime (1p)

**Procedure:** For each test minibatch, the conditioning (features, kNN graph, observed edge features) is computed once from frames 0-9. Then, for each of the M futures:

1. Draw x_0 from the prior (Gaussian, or γ_k v̂^9 + σε for the informed model).
2. Integrate the learned ODE dx/dτ = v_θ(x, τ | c) from τ = 0 to 1 with the explicit Euler method. The future edge features are recomputed from x at every step.
3. Un-normalize the velocities and integrate them to positions from p^9.

**Hyperparameters:** 16 Euler steps, M = 20 futures per scene, σ = 1, the same for both models. The number of steps was fixed before evaluation and not tuned on the test set.

**Runtime:** Measured on one A100 (40 GB), with test minibatches of 16 scenes (on average 387 pedestrians), after 10 warm-up batches, mean over 50 batches:

| Model | One future (M = 1) | 20 futures (M = 20) | One forward pass of v_θ |
|---|---|---|---|
| Uninformed | 121.4 ± 21.3 ms | 2.39 s | 6.4 ms |
| Informed | 122.3 ± 21.4 ms | 2.42 s | 6.4 ms |

The prior makes no difference in runtime. One future for one minibatch takes about 0.12 s, almost all of it in the 16 forward passes. The 20 futures are generated one after another, so the time grows linearly with M.

---

## 14a. Pointwise error metrics (4p)

Upload: `assignments/part2_outputs/bonus1_error_over_horizon.png`

2000 test scenes, 20 futures per scene, 16 Euler steps:

| Model | ADE [m] | FDE [m] | min20 ADE [m] | min20 FDE [m] |
|---|---|---|---|---|
| Uninformed prior | **0.914** | **1.850** | **0.644** | **1.217** |
| Informed prior | 0.919 | 1.860 | 0.646 | 1.221 |

**Additional pointwise measure: displacement error per frame.** For every generated frame t ∈ {10, ..., 29}:

DE^t = (1/S) Σ_s (1/(M N_s)) Σ_m Σ_i ‖p̂_{s,m,i}^t − p_{s,i}^t‖₂

with the 25th-75th percentile across scenes as a band. Averaging DE^t over t gives ADE, and DE^29 is FDE. It shows where in the horizon the error comes from.

**Results:**
- The two models are practically equal. The uninformed model is ahead by 0.3-0.5% on all four metrics, which is within the variation we would expect from a different training seed (we trained one seed per model).
- The per-frame curves lie on top of each other over the whole horizon.
- The error grows faster than linearly over the horizon, because positions are integrated velocities, so errors in the velocity add up.
- The min20 metrics are much lower than the mean metrics (0.64 against 0.91 m). The 20 futures are spread out and one of them is usually close to the truth, which is what a multimodal model should do.

**Are the results as expected?** Not fully.
- We expected the informed prior to win, because the data has strong velocity memory and the prior reduces the transport cost by a third above the noise floor.
- With our first architecture (1D convolutions over time) it did: ADE 0.927 against 0.971 m and FDE 1.869 against 2.069 m, with the gain growing late in the horizon. Those convolutions could not see the observed frames from frames 26-29.
- After switching to temporal attention, the uninformed model improved strongly (FDE 2.069 → 1.850 m), the informed one only a little (1.869 → 1.860 m), and the difference disappeared.
- So most of the informed prior's advantage came from passing the last velocity to frames the network could not see itself (14c).

**In real life on the platform:**
- At 0.5 s ahead the error is about 0.1 m, at 1 s about 0.25 m (less than a shoulder width), at 5 s about 0.9 m, and at 10 s about 1.85 m.
- A walking pedestrian covers about 13 m in 10 s, so an FDE of 1.85 m means we usually get the lane and the direction right, but not the exact spot. Many pedestrians stand still, which lowers the average.
- The model is accurate enough for crowd-level questions, such as which parts of the platform will be busy in the next 5-10 s or where flows go to and from the stairs.
- It is not accurate enough for an individual 10 s ahead, or for anything that depends on 0.3-0.5 m precision, such as predicting collisions.

---

## 14b. Distributional comparison (4p)

Upload: `assignments/part2_outputs/histograms.png`, `assignments/part2_outputs/bonus2_distance_and_time_profiles.png`, `assignments/part2_outputs/bonus4_collision_rate.png`

**Additional properties (not in the PDF):**
- **Path length** over the prediction window: L_i = Σ_{t=10}^{29} ‖p_i^t − p_i^{t−1}‖₂ (the numerator of the tortuosity).
- **Speed and acceleration per frame:** the median and 25th-75th percentile of ‖v_i^t‖ and ‖a_i^t‖ for every t, to see drift over the horizon.
- **Collision rate:** C(r) = fraction of (scene, future, pedestrian, frame) with d_i^t < r, where d_i^t is the distance to the closest pedestrian in the same scene and future. Reported at r = 0.3 m and as a curve over r.
- **Binned KL divergence:** KL(model ‖ ground truth) per histogram, with 60 equal-width bins over the central 99% of the ground truth plus two overflow bins, and a pseudo count of 1 on the ground-truth bins (as in Practical 3).

| KL [nats] | speed | acceleration | tortuosity | closest |
|---|---|---|---|---|
| Uninformed | 0.0210 | 0.0880 | 0.0133 | 0.0322 |
| Informed | 0.0197 | 0.0879 | 0.0193 | 0.0328 |

| Collisions (r = 0.3 m) | Ground truth | Uninformed | Informed |
|---|---|---|---|
| Rate | 0.64% | 2.57% | 2.61% |

**Speed:** Both models reproduce the bimodal speed distribution: a peak for waiting people near 0 m/s and a walking peak around 1.2-1.3 m/s with the right tail. The models put too much mass in the lowest bin, though. Their standing pedestrians stand more still than real ones, who show some tracking jitter. The median speed per frame is about 0.11 m/s against 0.135 m/s for the ground truth.

**Acceleration:** The overall shape is right, but the models are smoother: the median is about 0.25 m/s² against 0.32 m/s². The spikes of the ground truth at 0.125 and 0.25 m/s² come from the float16 quantization (9e), and the models cannot (and should not) reproduce them. This also inflates the acceleration KL.

**Tortuosity:** Matched well, including the spike at χ ≈ 1 (straight walkers) and the long tail from people standing still. The uninformed model matches it best (KL 0.013 against 0.019).

**Distance travelled:** The mean is 4.62 m for the ground truth, 4.35 m for the uninformed model and 4.34 m for the informed model. Both models move people slightly too little.

**Closest neighbour:** The models get the peak at about 0.7 m, but they have too much mass below 0.5 m, and their collision rate is 4× higher than the ground truth for every radius below about 0.8 m.

**Are the results as expected?** Partly.
- The distributional metrics are almost the same for both models. That is expected, because both learn the same target distribution, and in the ideal case the prior only changes the path, not the end distribution.
- We expected the future edge features from x_τ to lower the collision rate. They did not: it is 2.57-2.61%, the same as in the version that reused the frame-9 offsets (2.57-2.58%). Early in sampling the implied positions are dominated by noise (σ = 1 corresponds to about 0.7 m/s along the platform), and the neighbour set is still fixed at frame 9.

**What human behaviour is captured:**
- The two groups of pedestrians, waiting and walking, at a realistic walking speed.
- Smooth motion that keeps its direction (realistic tortuosity), and walking along the lanes of the platform.
- Keeping a typical personal space of about 0.7 m.

**What is not captured well:**
- Collision avoidance at short range. Two people approaching each other walk into each other's personal space too often.
- The small fidgeting of people who are waiting.
- Part of the long trips: some walkers stop or slow down too early.

---

## 14c. Which prior performed better and why (2p)

With our final architecture, **neither prior performed clearly better.** The uninformed prior is ahead by 0.3-0.5% on ADE, FDE, min20 ADE and min20 FDE, and the informed one is slightly ahead on the speed KL. These differences are within what we would expect from a different training seed, and the runtime is the same.

**Hypotheses for why:**
1. **The network no longer needs the information in the prior.** The informed prior's main content is v̂^9, the last observed velocity. With temporal attention, every future token reads the observed frames directly, so the network already has this information at every flow time τ. With our first architecture (convolutions reaching only ±16 frames), frames 26-29 could not see the observed frames, and the informed prior was the only way v̂^9 reached them. That version showed a clear informed advantage (FDE 1.87 against 2.07 m) that grew towards frame 29. With attention, the uninformed model caught up (FDE 1.85 m) and the gap disappeared. This is the clearest evidence that the gain came from the information path, not from the transport.
2. **Shorter transport helps less than expected with 16 steps.** The informed prior still reduces the transport cost from 77.5 to 65.5 (37.5 → 25.5 above the noise floor). The Euler sweep shows that both models have converged by 16-32 steps, so the straighter paths of the informed prior no longer give a measurable gain in discretization error. With 1-2 steps, the informed model is even slightly worse in ADE.
3. **The noise dominates the prior.** The informed mean γ_k v̂^9 is about 1-1.5 normalized units for a walker and close to 0 for the many waiting pedestrians, and σ = 1 noise is added on top. The informed and uninformed priors therefore overlap a lot, so the difference between the two flows is small to begin with. A smaller σ for the informed prior would make it more informative, but would no longer be a comparison with only the mean changed.

So an informed prior is most useful when it adds information the network cannot easily get on its own, or when very few sampling steps are used.

---

## 14d. Non-obvious limitations (1p)

- **Collisions do not go away:** using the positions implied by x_τ for the future edge features did not reduce the 4× too high collision rate. The implied positions are mostly noise for small τ, and the neighbour set is still the one of frame 9.
- **kNN neighbours are often far away:** k = 8 reaches a median of 7.6 m, while only about 1.4 pedestrians are within 2 m. With mean aggregation, the one neighbour that matters is averaged with 7 who do not.
- **Tied to this station:** because we use the absolute position, the model does not transfer to another platform or station.
- **Noisy training targets:** float16 quantization and tracker glitches (speeds up to 47 m/s) are in the targets, and the MSE loss is sensitive to such outliers.
- **Metrics can be misleading:** fewer Euler steps give a lower ADE (about 0.70 m with 1-2 steps against about 0.89 m with 16, on the 200-scene sweep subset) because the samples collapse towards the mean, while the KL gets 10-20× worse. ADE alone would favour wrong samples.
- **One seed per model:** the difference between the two priors (0.5%) is smaller than the spread we would expect between training seeds, which we did not measure.
- **Only 10% of the data was used** (10,000 of the 100,000 scenes).

---

## 14e. Most effective changes for efficiency (1p)

1. **Generate all M futures in one batch.** Our sampler loops over the 20 futures one after another, and a forward pass for 387 pedestrians takes only 6.4 ms, so the GPU is mostly waiting on kernel launches. Stacking the 20 prior samples into the batch dimension, with the conditioning and graph repeated, would make generating 20 futures close to the cost of one (about 0.1-0.2 s instead of 2.4 s per minibatch).
2. **Fewer Euler steps through straighter flows.** Rectified flow (reflow) or distillation would allow 2-4 steps instead of 16 without the distribution collapse seen in the sweep.
3. **A sparser, more relevant graph.** A radius graph (2-3 m) in `make_edge_index` instead of kNN would have fewer edges and better neighbours.
4. **Training on the full dataset** rather than more epochs, with mixed precision and `torch.compile` to reduce the overhead of the many small operations.
