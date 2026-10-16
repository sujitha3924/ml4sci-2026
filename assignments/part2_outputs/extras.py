"""Extra material for the Ans answers: loss curves, exploratory data analysis, inference timing."""
import json, pickle, time, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import torch

OUT = "extras"
import os
os.makedirs(OUT, exist_ok=True)

# 1. Loss curves
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), dpi=150, sharey=True)
for ax, (version, name) in zip(axs, [(0, "Uninformed prior"), (1, "Informed prior")]):
    rows = list(csv.DictReader(open(f"lightning_logs/version_{version}/metrics.csv")))
    def per_epoch(key):
        d = {}
        for r in rows:
            if r.get(key): d.setdefault(int(r["epoch"]), []).append(float(r[key]))
        e = sorted(d); return np.array(e), np.array([np.mean(d[k]) for k in e])
    te, tv = per_epoch("train_loss"); ve, vv = per_epoch("val_loss")
    ax.plot(te, tv, lw=2, label="train (epoch mean of logged steps)")
    ax.plot(ve, vv, lw=2, label="validation")
    best = int(ve[vv.argmin()])
    ax.axvline(best, color="gray", ls=":", lw=1, label=f"best val, epoch {best}")
    ax.set_title(name); ax.set_xlabel("Epoch"); ax.grid(alpha=0.4, ls="--", lw=0.5); ax.legend(frameon=False, fontsize=8)
    print(f"{name}: train first/last {tv[0]:.4f}/{tv[-1]:.4f}, val first/last/best {vv[0]:.4f}/{vv[-1]:.4f}/{vv.min():.4f} at {best}")
    print("  val per epoch:", [round(float(v), 4) for v in vv])
    print("  train per epoch:", [round(float(v), 4) for v in tv])
axs[0].set_ylabel("Flow matching loss (MSE)")
fig.tight_layout(); fig.savefig(f"{OUT}/13b_loss_curves.png"); plt.close(fig)

# 2. Exploratory data analysis on the full dataset
with open("eindhoven_centraal.pkl", "rb") as f:
    raw = pickle.load(f)
print("container:", type(raw), "len", len(raw), "element type", type(raw[0]), "dtype", getattr(raw[0], "dtype", None), "shape", raw[0].shape)
Ns = np.array([s.shape[0] for s in raw])
print("pedestrians per scene: min", Ns.min(), "max", Ns.max(), "mean", Ns.mean().round(2), "median", np.median(Ns), "total", Ns.sum())
print("frames:", set(s.shape[1] for s in raw[:1000]), "coords:", set(s.shape[2] for s in raw[:1000]))
print("approx memory float16 MB:", Ns.sum() * 30 * 2 * 2 / 1e6)
DT = 0.5
allp = np.concatenate([np.asarray(s, dtype=np.float32) for s in raw])          # (P, 30, 2)
print("x range", allp[..., 0].min(), allp[..., 0].max(), "y range", allp[..., 1].min(), allp[..., 1].max())
nan = np.isnan(allp).sum(); print("NaNs:", nan)
vals = np.unique(np.asarray(raw[0], dtype=np.float32)[..., 1])
print("example distinct y spacing near 30 m:", np.diff(np.unique(np.abs(allp[:20000, :, 1].ravel())))[-5:])
vel = np.diff(allp, axis=1) / DT                    # (P, 29, 2)
speed = np.linalg.norm(vel, axis=-1)
acc = np.diff(vel, axis=1) / DT
accm = np.linalg.norm(acc, axis=-1)
print("speed: mean %.3f median %.3f p90 %.3f p99 %.3f max %.3f" % (speed.mean(), np.median(speed), *np.percentile(speed, [90, 99]), speed.max()))
print("fraction speed < 0.1 m/s: %.3f, < 0.3: %.3f, > 1.0: %.3f" % ((speed < 0.1).mean(), (speed < 0.3).mean(), (speed > 1.0).mean()))
mean_speed_ped = speed.mean(1)
print("fraction pedestrians with mean speed < 0.2 m/s: %.3f, > 0.8: %.3f" % ((mean_speed_ped < 0.2).mean(), (mean_speed_ped > 0.8).mean()))
print("acc: mean %.3f median %.3f p99 %.3f" % (accm.mean(), np.median(accm), np.percentile(accm, 99)))
print("distinct speed values in first 1e6:", len(np.unique(speed.ravel()[:1000000])))
print("mean velocity x %.4f y %.4f, std x %.4f y %.4f" % (*vel.reshape(-1, 2).mean(0), *vel.reshape(-1, 2).std(0)))
moving = speed > 0.5
heading = np.degrees(np.arctan2(vel[..., 1], vel[..., 0]))[moving]
h_hist, _ = np.histogram(heading, bins=np.linspace(-180, 180, 13))
print("heading histogram (30 deg bins from -180) of moving steps:", (h_hist / h_hist.sum()).round(3).tolist())
# speed over the 30 frames (stationarity)
print("mean speed per frame:", speed.mean(0).round(3).tolist())
# velocity autocorrelation in normalized units, per lag from frame 9
vm, vs = vel.reshape(-1, 2).mean(0), vel.reshape(-1, 2).std(0)
vn = (vel - vm) / vs
v9 = vn[:, 8]
gam = [(vn[:, 8 + k] * v9).sum() / (v9 ** 2).sum() for k in range(1, 21)]
corr = [np.mean([np.corrcoef(vn[:, 8 + k, c], v9[:, c])[0, 1] for c in range(2)]) for k in range(1, 21)]
print("lag-k corr of normalized velocity with v^9 (k=1..20):", np.round(corr, 3).tolist())
# tortuosity over frames 9-29
path = np.linalg.norm(np.diff(allp[:, 9:], axis=1), axis=-1).sum(1)
net = np.linalg.norm(allp[:, 29] - allp[:, 9], axis=-1)
tort = path / np.maximum(net, 0.01)
print("tortuosity median %.3f p75 %.3f p90 %.3f; fraction > 2: %.3f" % (np.median(tort), *np.percentile(tort, [75, 90]), (tort > 2).mean()))
print("path length 9->29 mean %.2f median %.2f" % (path.mean(), np.median(path)))
# nearest neighbour distance and local density on 10k scenes
rng = np.random.default_rng(0)
idx = rng.choice(len(raw), 10000, replace=False)
nn, close, dens = [], 0, []
for i in idx:
    p = np.asarray(raw[i], dtype=np.float32)
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)       # (N, N, 30)
    d[np.arange(len(p)), np.arange(len(p))] = np.inf
    nn.append(d.min(1).ravel())
    dens.append((d[:, :, 9] < 2.0).sum(1))
nn = np.concatenate(nn); dens = np.concatenate(dens)
print("nn distance: median %.3f p5 %.3f p1 %.3f; frac < 0.3: %.4f, < 0.5: %.4f, < 1: %.3f" % (np.median(nn), *np.percentile(nn, [5, 1]), (nn < 0.3).mean(), (nn < 0.5).mean(), (nn < 1).mean()))
print("neighbours within 2 m at frame 9: mean %.2f, median %.1f, p90 %.1f, max %d" % (dens.mean(), np.median(dens), np.percentile(dens, 90), dens.max()))
print("fraction of pedestrians with >= 8 neighbours within 2 m: %.3f" % (dens >= 8).mean())
# kNN radius: distance to 8th neighbour at frame 9
k8 = []
for i in idx[:3000]:
    p = np.asarray(raw[i], dtype=np.float32)[:, 9]
    d = np.sort(np.linalg.norm(p[:, None] - p[None], axis=-1), axis=1)
    if len(p) > 8: k8.append(d[:, 8])
k8 = np.concatenate(k8)
print("distance to 8th neighbour at frame 9: median %.2f p10 %.2f p90 %.2f" % (np.median(k8), *np.percentile(k8, [10, 90])))

# EDA figure
fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), dpi=150)
axs[0, 0].hist(Ns, bins=np.arange(4.5, 101.5, 2), color="C0")
axs[0, 0].set_title("Pedestrians per scene"); axs[0, 0].set_xlabel("N"); axs[0, 0].set_ylabel("Scenes")
axs[0, 1].hist(speed.ravel(), bins=np.linspace(0, 2.5, 101), density=True, color="C0")
axs[0, 1].set_yscale("log"); axs[0, 1].set_title(r"Speed $\|v_i^t\|$"); axs[0, 1].set_xlabel("m/s")
axs[0, 2].hist(accm.ravel(), bins=np.linspace(0, 3, 121), density=True, color="C0")
axs[0, 2].set_yscale("log"); axs[0, 2].set_title(r"Acceleration $\|a_i^t\|$ (float16 steps)"); axs[0, 2].set_xlabel(r"m/s$^2$")
axs[1, 0].plot(np.arange(1, 21), corr, marker="o", label="corr($v^{9+k}$, $v^9$)")
axs[1, 0].plot(np.arange(1, 21), gam, marker="s", label=r"least-squares $\gamma_k$")
axs[1, 0].set_title("Velocity memory"); axs[1, 0].set_xlabel("Lag k [frames]"); axs[1, 0].legend(frameon=False); axs[1, 0].set_ylim(0, 1)
axs[1, 1].hist(nn, bins=np.linspace(0, 4, 81), density=True, color="C0")
axs[1, 1].axvline(0.3, color="gray", ls=":"); axs[1, 1].set_title("Distance to closest pedestrian"); axs[1, 1].set_xlabel("m")
axs[1, 2].hist(np.clip(tort, 1, 5), bins=np.linspace(1, 5, 81), density=True, color="C0")
axs[1, 2].set_yscale("log"); axs[1, 2].set_title(r"Tortuosity $\chi_i$ (frames 9-29, clipped at 5)")
for ax in axs.ravel(): ax.grid(alpha=0.4, ls="--", lw=0.5)
fig.tight_layout(); fig.savefig(f"{OUT}/9e_eda_overview.png"); plt.close(fig)

# occupancy and mean velocity map on the station background
EXTENT = (-6.309, 75.091, -16.539, 10.909)
sub = allp[rng.choice(len(allp), 300000, replace=False)]
fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)
ax.imshow(plt.imread("Eindhoven_centraal_platform_3_4.png"), extent=EXTENT, origin="upper", zorder=0)
h = ax.hexbin(sub[..., 1].ravel(), sub[..., 0].ravel(), gridsize=120, bins="log", mincnt=1, cmap="viridis", alpha=0.75, zorder=1)
gx = np.linspace(EXTENT[0], EXTENT[1], 40); gy = np.linspace(EXTENT[2], EXTENT[3], 14)
pos = allp[:, :-1].reshape(-1, 2); vv = vel.reshape(-1, 2)
sel = rng.choice(len(pos), 2000000, replace=False); pos, vv = pos[sel], vv[sel]
ix = np.digitize(pos[:, 1], gx); iy = np.digitize(pos[:, 0], gy)
U = np.zeros((len(gy) + 1, len(gx) + 1)); V = np.zeros_like(U); C = np.zeros_like(U)
np.add.at(U, (iy, ix), vv[:, 1]); np.add.at(V, (iy, ix), vv[:, 0]); np.add.at(C, (iy, ix), 1)
m = C > 200
cx, cy = np.meshgrid(np.r_[gx[0], (gx[1:] + gx[:-1]) / 2, gx[-1]], np.r_[gy[0], (gy[1:] + gy[:-1]) / 2, gy[-1]])
ax.quiver(cx[m], cy[m], (U / np.maximum(C, 1))[m], (V / np.maximum(C, 1))[m], color="white", scale=8, width=0.002, zorder=2)
ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
ax.set_title("Occupancy (log) and mean velocity of all pedestrians"); fig.colorbar(h, ax=ax, label="count (log)")
fig.tight_layout(); fig.savefig(f"{OUT}/9e_occupancy_velocity_map.png"); plt.close(fig)
del allp, vel, acc, speed, accm

# 3. Inference timing with the trained checkpoints
nb = json.load(open("Assignment_2_part2.ipynb", encoding="utf-8"))
src = lambda i: "".join(nb["cells"][i]["source"])
g = {}
for i in [1, 5, 7, 11]:
    exec(src(i), g)
cell15 = src(15).split("gamma = fit_horizon_gamma")[0]
exec(cell15, g)
gamma = torch.tensor([0.798, 0.776, 0.728, 0.697, 0.658, 0.624, 0.601, 0.571, 0.552, 0.53, 0.515, 0.496, 0.483, 0.469, 0.454, 0.444, 0.431, 0.42, 0.408, 0.398])
models = {"Uninformed": g["PedestrianFlowModel"](g["norm"]), "Informed": g["InformedPriorFlowModel"](g["norm"], gamma)}
ckpts = {"Uninformed": "lightning_logs/version_0/checkpoints/uninformed-epoch=28-val_loss=0.6576.ckpt",
         "Informed": "lightning_logs/version_1/checkpoints/informed-epoch=28-val_loss=0.6598.ckpt"}
n_params = sum(p.numel() for p in models["Uninformed"].parameters())
print("parameters:", n_params)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
loader = g["data_module"].test_dataloader()
batches = [b for _, b in zip(range(60), loader)]
print("mean pedestrians per test minibatch of 16 scenes:", np.mean([b.num_nodes for b in batches]))
for name, model in models.items():
    model.load_state_dict(torch.load(ckpts[name], map_location="cpu")["state_dict"])
    model = model.cuda().eval()
    for M in [1, 20]:
        for b in batches[:10]:
            model.sample(b, M=M, n_steps=16)
        torch.cuda.synchronize()
        times = []
        for b in batches[10:]:
            t0 = time.perf_counter(); model.sample(b, M=M, n_steps=16); torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        print(f"{name}: M={M}, 16 Euler steps, batch of 16 scenes: mean {1000*np.mean(times):.1f} ms, std {1000*np.std(times):.1f} ms")
    # one forward pass of the velocity field
    b = batches[10].to("cuda"); cond = model.condition(b); x = model.sample_prior(cond); tau = torch.zeros(x.shape[0], device="cuda")
    with torch.no_grad():
        for _ in range(10): model.vf(x, tau, cond)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(100): model.vf(x, tau, cond)
        torch.cuda.synchronize()
    print(f"{name}: one velocity field forward pass: {1000*(time.perf_counter()-t0)/100:.2f} ms")
    # scaling test: one scene with 1000 pedestrians and a long sequence is not possible (fixed 30 frames), so time N = 1000
p = torch.randn(1000, 30, 2) * torch.tensor([3.0, 20.0]) + torch.tensor([-2.0, 30.0])
from torch_geometric.data import Batch, Data
big = Batch.from_data_list([Data(x=p, num_nodes=1000)]).to("cuda")
model = models["Informed"]
with torch.no_grad():
    for _ in range(3): model.sample(big, M=1, n_steps=16)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): model.sample(big, M=1, n_steps=16)
    torch.cuda.synchronize()
print(f"1000 pedestrians, one scene, 16 Euler steps: {1000*(time.perf_counter()-t0)/10:.1f} ms; peak memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
