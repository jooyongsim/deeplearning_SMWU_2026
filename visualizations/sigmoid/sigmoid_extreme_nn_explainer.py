import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

rng = np.random.default_rng(42)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

z_structured = np.array([
    -50, -40, -30, -20, -10, -8, -6, -4, -2, -1, -0.5, 0,
     0.5, 1, 2, 4, 6, 8, 10, 20, 30, 40, 50
], dtype=float)
z_random = rng.uniform(-50, 50, size=90)
z = np.concatenate([z_structured, z_random])
p = sigmoid(z)
n = len(z)
jitter_top = rng.uniform(-0.05, 0.05, size=n)

def p_to_display(prob):
    return -50 + 100 * prob

z_display = z
p_display = p_to_display(p)

fig = plt.figure(figsize=(10, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.8, 1.3], hspace=0.42)

ax_top = fig.add_subplot(gs[0])
ax_curve = fig.add_subplot(gs[1])
ax_hist = fig.add_subplot(gs[2])

fig.suptitle("How sigmoid compresses logits into probabilities", fontsize=16)

ax_top.set_xlim(-55, 55)
ax_top.set_ylim(-0.28, 1.35)
ax_top.set_yticks([])
ax_top.set_title("Scattered logits move from the z-axis to the σ(z)-axis")
ax_top.axhline(1.0, linewidth=1)
ax_top.axhline(0.0, linewidth=1)

ax_top.text(-54, 1.08, "logit axis  z ∈ [-50, 50]", fontsize=11, va="bottom")
ax_top.text(-54, 0.08, "probability axis  σ(z) ∈ [0, 1]", fontsize=11, va="bottom")

for tick in [-50, -25, 0, 25, 50]:
    ax_top.plot([tick, tick], [0.97, 1.03], linewidth=1)
    ax_top.text(tick, 1.15, f"{tick:g}", ha="center", va="bottom", fontsize=9)

for prob_tick in [0, 0.25, 0.5, 0.75, 1.0]:
    x_tick = p_to_display(prob_tick)
    ax_top.plot([x_tick, x_tick], [-0.03, 0.03], linewidth=1)
    ax_top.text(x_tick, -0.12, f"{prob_tick:g}", ha="center", va="top", fontsize=9)

top_scatter = ax_top.scatter(z_display, 1.0 + jitter_top, s=28, alpha=0.8)

status_text = ax_top.text(
    0.5, 0.52, "",
    transform=ax_top.transAxes,
    ha="center", va="center", fontsize=11
)

ax_top.text(-37, 0.55, "negative logits collapse toward 0", fontsize=10)
ax_top.text(8, 0.55, "positive logits collapse toward 1", fontsize=10)
ax_top.text(-4, 0.33, "values near z=0 map near 0.5", fontsize=10)

x = np.linspace(-50, 50, 1200)
y = sigmoid(x)
ax_curve.plot(x, y, linewidth=2)
ax_curve.scatter(z, p, s=20, alpha=0.35)
curve_highlight = ax_curve.scatter([0], [0.5], s=80)

ax_curve.set_xlim(-50, 50)
ax_curve.set_ylim(-0.02, 1.02)
ax_curve.set_xlabel("logit z")
ax_curve.set_ylabel("σ(z)")
ax_curve.set_title("Sigmoid curve: σ(z) = 1 / (1 + e⁻ᶻ)")
ax_curve.grid(alpha=0.25)
ax_curve.axvline(0.0, linewidth=1, linestyle="--", alpha=0.5)
ax_curve.axhline(0.5, linewidth=1, linestyle="--", alpha=0.5)

ax_curve.text(
    0.02, 0.93,
    "Large negative z → ~0,  z≈0 → 0.5,  large positive z → ~1",
    transform=ax_curve.transAxes,
    fontsize=10, va="top"
)

bins_z = np.linspace(-50, 50, 26)
bins_p = np.linspace(0, 1, 26)

counts_z, edges_z = np.histogram(z, bins=bins_z)
counts_p, edges_p = np.histogram(p, bins=bins_p)

counts_z = counts_z.astype(float)
counts_p = counts_p.astype(float)
max_count = max(counts_z.max(), counts_p.max())
if max_count > 0:
    counts_z /= max_count
    counts_p /= max_count

centers_z = (edges_z[:-1] + edges_z[1:]) / 2
centers_p = (edges_p[:-1] + edges_p[1:]) / 2
centers_p_display = p_to_display(centers_p)

widths_z = np.diff(edges_z)
widths_p_display = 100 * np.diff(edges_p)

bars = ax_hist.bar(centers_z, counts_z, width=widths_z * 0.9, alpha=0.55, align='center')

ax_hist.set_xlim(-55, 55)
ax_hist.set_ylim(0, 1.1)
ax_hist.set_title("Distribution view: broad logits become saturated probabilities")
ax_hist.set_ylabel("normalized count")
ax_hist.set_xlabel("shared display axis  (top: z, bottom target: σ(z))")
ax_hist.grid(alpha=0.2, axis='y')

hist_text = ax_hist.text(
    0.02, 0.93,
    "",
    transform=ax_hist.transAxes,
    fontsize=10, va="top"
)

ax_hist.text(-50, -0.14, "z-scale", fontsize=9, va="top")
ax_hist.text(50, -0.14, "σ(z)-scale mapped to same width", fontsize=9, va="top", ha="right")

frames = 140

def ease_in_out(t):
    return t * t * (3 - 2 * t)

highlight_path = np.linspace(-50, 50, frames)
highlight_probs = sigmoid(highlight_path)

def update(frame):
    t_raw = frame / (frames - 1)
    t = ease_in_out(t_raw)

    x_now = (1 - t) * z_display + t * p_display
    y_now = (1 - t) * (1.0 + jitter_top) + t * (0.02 + 0.35 * jitter_top)
    top_scatter.set_offsets(np.column_stack([x_now, y_now]))

    if t < 0.1:
        status_text.set_text("Start: logits are spread across a wide range")
    elif t < 0.45:
        status_text.set_text("Applying σ(z): every z is mapped into [0, 1]")
    elif t < 0.8:
        status_text.set_text("Extreme values saturate: negative → 0, positive → 1")
    else:
        status_text.set_text("End: a wide logit range has been squashed into probabilities")

    zh = highlight_path[frame]
    ph = highlight_probs[frame]
    curve_highlight.set_offsets(np.array([[zh, ph]]))

    for i, bar in enumerate(bars):
        x0 = centers_z[i]
        x1 = centers_p_display[i]
        w0 = widths_z[i] * 0.9
        w1 = widths_p_display[i] * 0.9
        h0 = counts_z[i]
        h1 = counts_p[i]

        x_bar = (1 - t) * x0 + t * x1
        w_bar = (1 - t) * w0 + t * w1
        h_bar = (1 - t) * h0 + t * h1

        bar.set_x(x_bar - w_bar / 2)
        bar.set_width(w_bar)
        bar.set_height(h_bar)

    if t < 0.2:
        hist_text.set_text("Initial distribution in z-space")
    elif t < 0.8:
        hist_text.set_text("Distribution is being compressed by the nonlinear sigmoid")
    else:
        hist_text.set_text("Final distribution in probability-space: mass piles up near 0 and 1")

    return [top_scatter, curve_highlight, status_text, hist_text, *bars]

anim = FuncAnimation(
    fig, update,
    frames=140,
    interval=55,
    blit=False,
    repeat=True
)

anim.save("sigmoid_extreme_nn_explainer.gif", writer=PillowWriter(fps=18))
plt.show()
