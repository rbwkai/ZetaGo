"""Regenerate every figure in `Docs/figures/` from committed result files.

This is the single entry point behind `Docs/figures/figures.md`. It covers both
the three figures that appear in the paper (which have their own scripts, reused
here rather than reimplemented -- a second implementation is a second thing that
can disagree with the paper) and the analysis figures that exist only in
`Docs/figures/`, for the deck's backup slides, the internal record, and Q&A.

Everything except `class_distribution` and `recall_heatmap` reads only from
`results/*.json` and runs in seconds. Those two are marked "heavy": one reads
`train.h5`, the other loads the champion checkpoint and re-predicts the whole
validation split (~2 min). `--skip-heavy` omits exactly those two.

Run from the repo root:
    venv/bin/python -m data.make_docs_figures                  # all, to Docs/figures/*.png
    venv/bin/python -m data.make_docs_figures --skip-heavy     # results-only figures
    venv/bin/python -m data.make_docs_figures --only factor_a_interaction
    venv/bin/python -m data.make_docs_figures --list
"""

import argparse
import json
import os

import numpy as np

MODELS = ["logreg", "rf", "knn", "svm", "cnn"]
MODEL_COLOURS = {
    "logreg": "#4C72B0", "rf": "#DD8452", "knn": "#55A868",
    "svm": "#C44E52", "cnn": "#8172B3",
}
MARKERS = {"logreg": "o", "rf": "s", "knn": "^", "svm": "D", "cnn": "*"}
FULL_VOLUME = 107969
SIDE_TO_MOVE_TEST = 0.8644276284719553  # DATASET.md SS1, recomputed in final_test_metrics.json


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    _plt().close(fig)
    print(f"wrote {out_path}")


def _load(name):
    return json.load(open(os.path.join("results", name)))


def _cells(rows):
    """Sweep rows only -- drops the three baseline rows, which use seed=-1."""
    return [r for r in rows if r.get("seed", -1) != -1]


def _mean_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _p_mathtext(p):
    """Format a p-value as mathtext. Plain `f"{p:.1e}"` renders as `3.4e-64`,
    which mathtext then sets as `3.4e - 64` (the hyphen becomes a spaced minus)."""
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


# --------------------------------------------------------------- paper figures

def fig_class_distribution(out_path, args):
    from data.plot_class_distribution import plot

    plot(args.train_h5, out_path, "train")


def fig_learning_curve(out_path, args):
    from data.plot_learning_curve import load_curve, plot

    plot(load_curve(os.path.join("results", "supervised_track_a_metrics.json")), out_path)


def fig_recall_heatmap(out_path, args):
    from data.plot_recall_heatmap import compute_recall, plot

    recall, counts = compute_recall(args.champion, 4, args.val_h5, 0)
    plot(recall, counts, out_path)


# ------------------------------------------------------------ Track A figures

def fig_factor_a_interaction(out_path, args):
    """The paper's central claim, which has no figure in the paper itself: the
    N=2->7 effect depends on which model receives it. Full volume, 5 seeds."""
    plt = _plt()
    rows = _cells(_load("supervised_track_a_metrics.json"))
    sel = [r for r in rows if r["data_volume_games"] == FULL_VOLUME and r["dedup"] == "none"]
    encodings = sorted(set(r["encoding"] for r in sel))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in MODELS:
        means, sds = [], []
        for enc in encodings:
            vals = [r["move_top1"] for r in sel if r["model"] == m and r["encoding"] == enc]
            mu, sd = _mean_sd(vals)
            means.append(mu)
            sds.append(sd)
        spread = max(means) - min(means)
        ax.errorbar(
            encodings, means, yerr=sds, marker=MARKERS[m], capsize=3,
            color=MODEL_COLOURS[m], label=f"{m}  (spread {spread:.4f})",
        )
    ax.set_xticks(encodings)
    ax.set_xlabel("Factor A: feature-plane count $N$")
    ax.set_ylabel("mean top-1 accuracy (5 seeds)")
    ax.set_title("Feature richness is an interaction, not a main effect\n"
                 "full volume; error bars = across-seed SD")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def fig_factor_a_interaction_normalised(out_path, args):
    """Same data as `factor_a_interaction`, each model re-centred on its own
    N=2 score. The absolute plot is dominated by the between-model gap; this one
    isolates the within-model effect, which is what the interaction term tests."""
    plt = _plt()
    rows = _cells(_load("supervised_track_a_metrics.json"))
    sel = [r for r in rows if r["data_volume_games"] == FULL_VOLUME and r["dedup"] == "none"]
    encodings = sorted(set(r["encoding"] for r in sel))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in MODELS:
        means = [float(np.mean([r["move_top1"] for r in sel
                                if r["model"] == m and r["encoding"] == enc])) for enc in encodings]
        seed_sd = float(np.mean([
            np.std([r["move_top1"] for r in sel if r["model"] == m and r["encoding"] == enc], ddof=1)
            for enc in encodings
        ]))
        delta = [v - means[0] for v in means]
        ax.plot(encodings, delta, marker=MARKERS[m], color=MODEL_COLOURS[m],
                label=f"{m}  (seed SD {seed_sd:.5f})")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(encodings)
    ax.set_xlabel("Factor A: feature-plane count $N$")
    ax.set_ylabel(r"top-1 change from its own $N{=}2$ score")
    ax.set_title("Same data, each model re-centred on its own $N{=}2$\n"
                 "linear models gain ~+0.10; the CNN gains nothing measurable")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def fig_data_redundancy(out_path, args):
    """Why the learning curve's x-axis is unique positions, not games: two orders
    of magnitude more games buys one order of magnitude more unique positions.

    The exact ratio depends on how the per-seed subsamples are aggregated, and
    the project's documents do not agree: this figure uses the mean over 5 seeds
    (10.8x), `TRAINING_RESULTS.md` SS5.2's table is a single seed (10.9x), and
    `main.tex` SSIV's bullet quotes a third set of counts (10.3x). The title
    states which basis is plotted rather than leaving it implicit."""
    plt = _plt()
    rows = _cells(_load("supervised_track_a_metrics.json"))
    volumes = sorted(set(r["data_volume_games"] for r in rows))
    games, rows_n, uniq = [], [], []
    for v in volumes:
        sel = [r for r in rows if r["data_volume_games"] == v]
        games.append(v)
        rows_n.append(float(np.mean([r["data_volume_rows"] for r in sel])))
        uniq.append(float(np.mean([r["data_volume_unique"] for r in sel])))

    x = np.arange(len(volumes))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    w = 0.27
    ax.bar(x - w, games, w, label="games", color="#4C72B0")
    ax.bar(x, rows_n, w, label="rows (positions)", color="#DD8452")
    ax.bar(x + w, uniq, w, label="unique positions", color="#55A868")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" for v in volumes], rotation=15)
    ax.set_xlabel("Factor B level (games sampled)")
    ax.set_ylabel("count (log scale)")
    ax.set_title("Rows grow with games; unique positions do not")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    growth_games = [g / games[0] for g in games]
    growth_uniq = [u / uniq[0] for u in uniq]
    ax2.plot(x, growth_games, marker="o", color="#4C72B0", label="games")
    ax2.plot(x, growth_uniq, marker="s", color="#55A868", label="unique positions")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{v:,}" for v in volumes], rotation=15)
    ax2.set_xlabel("Factor B level (games sampled)")
    ax2.set_ylabel(r"growth relative to the 1k level ($\times$)")
    ax2.set_title(f"{growth_games[-1]:.0f}$\\times$ more games "
                  f"$\\rightarrow$ {growth_uniq[-1]:.1f}$\\times$ more unique positions\n"
                  "(mean over 5 seeds; see the docstring on aggregation)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_path)


def fig_compute_budget(out_path, args):
    """Where the sweep's wall-clock actually went. SVM cost more than the CNN."""
    plt = _plt()
    rows = _cells(_load("supervised_track_a_metrics.json"))
    train_h = {m: sum(r["train_seconds"] for r in rows if r["model"] == m) / 3600.0 for m in MODELS}
    infer_ms = {m: float(np.mean([r["infer_ms_per_sample"] for r in rows if r["model"] == m]))
                for m in MODELS}
    order = sorted(MODELS, key=lambda m: -train_h[m])
    total = sum(train_h.values())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax.barh(order, [train_h[m] for m in order], color=[MODEL_COLOURS[m] for m in order])
    for i, m in enumerate(order):
        ax.text(train_h[m], i, f"  {train_h[m]:.2f} h ({100 * train_h[m] / total:.0f}%)",
                va="center", fontsize=8)
    ax.set_xlabel("total training time over the 300-cell sweep (hours)")
    ax.set_title(f"Training cost by model (total {total:.1f} h)")
    ax.set_xlim(0, max(train_h.values()) * 1.35)
    ax.grid(alpha=0.3, axis="x")

    order2 = sorted(MODELS, key=lambda m: -infer_ms[m])
    ax2.barh(order2, [infer_ms[m] for m in order2], color=[MODEL_COLOURS[m] for m in order2])
    for i, m in enumerate(order2):
        ax2.text(infer_ms[m], i, f"  {infer_ms[m]:.4f}", va="center", fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("mean inference time (ms/sample, log scale)")
    ax2.set_title("Inference cost: k-NN pays at predict time")
    ax2.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------- Phase 4: test set and statistics

def fig_val_vs_test(out_path, args):
    """Finding 8: no generalisation gap. Same 25 checkpoints, both splits."""
    plt = _plt()
    val = _load("final_row_val_metrics.json")
    test = _load("final_test_metrics.json")

    x = np.arange(len(MODELS))
    vm, vs, tm, ts = [], [], [], []
    for m in MODELS:
        a, b = _mean_sd([r["move_top1"] for r in val if r.get("model") == m])
        c, d = _mean_sd([r["move_top1"] for r in test if r.get("model") == m])
        vm.append(a); vs.append(b); tm.append(c); ts.append(d)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    w = 0.36
    ax.bar(x - w / 2, vm, w, yerr=vs, capsize=3, label="validation", color="#4C72B0")
    ax.bar(x + w / 2, tm, w, yerr=ts, capsize=3, label="test (sealed)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("mean top-1 accuracy (5 seeds)")
    ax.set_ylim(0.75, 0.93)
    ax.set_title("Validation vs. sealed test, same checkpoints")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    delta = [t - v for t, v in zip(tm, vm)]
    ax2.bar(x, delta, 0.55, color=["#55A868" if d >= 0 else "#C44E52" for d in delta])
    ax2.errorbar(x, delta, yerr=vs, fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODELS)
    ax2.set_ylabel("test $-$ validation top-1")
    ax2.set_title("Every gap is inside its own across-seed SD\n(error bars = that model's seed SD)")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, out_path)


def fig_anova_variance(out_path, args):
    """The interaction term's effect size, which the p-value does not convey."""
    plt = _plt()
    a = _load("final_stats_tests.json")["factor_a_x_model_interaction"]
    terms = [
        ("model\n(main effect)", a["ss_a"], a["df_a"], a["F_a"], a["p_a"], "#4C72B0"),
        ("encoding\n(main effect)", a["ss_b"], a["df_b"], a["F_b"], a["p_b"], "#DD8452"),
        ("model $\\times$ encoding\n(interaction)", a["ss_interaction"], a["df_interaction"],
         a["F_interaction"], a["p_interaction"], "#C44E52"),
        ("residual\n(seed noise)", a["ss_error"], a["df_error"], None, None, "#999999"),
    ]
    total = a["ss_total"]
    labels = [t[0] for t in terms]
    shares = [100.0 * t[1] / total for t in terms]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.barh(labels, shares, color=[t[5] for t in terms])
    for bar, t, s in zip(bars, terms, shares):
        txt = f"  {s:.2f}% of SS"
        if t[3] is not None:
            txt += (f"   $F({t[2]},{a['df_error']}){{=}}{t[3]:,.0f}$,"
                    f"  $p{{=}}{_p_mathtext(t[4])}$")
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, txt, va="center", fontsize=8)
    ax.set_xlim(0, 108)
    ax.set_xlabel("share of total sum of squares (%)")
    ax.set_title("Two-way ANOVA on top-1 (full volume, 5 models $\\times$ 3 encodings $\\times$ 5 seeds)\n"
                 "the interaction explains more variance than the encoding main effect")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    _save(fig, out_path)


def fig_mcnemar_discordance(out_path, args):
    """Finding 10: the crossover holds position-by-position, replicated on two
    disjoint splits. Only discordant pairs carry information about which model
    is better, so only those are plotted."""
    plt = _plt()
    d = _load("final_mcnemar_cnn_vs_rf.json")
    splits = ["val", "test"]
    x = np.arange(len(splits))
    cnn_only = [d[s]["a_right_b_wrong"] for s in splits]
    rf_only = [d[s]["a_wrong_b_right"] for s in splits]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    w = 0.36
    ax.bar(x - w / 2, cnn_only, w, label="CNN right, RF wrong", color="#8172B3")
    ax.bar(x + w / 2, rf_only, w, label="RF right, CNN wrong", color="#DD8452")
    for i, s in enumerate(splits):
        ax.text(i - w / 2, cnn_only[i], f"{cnn_only[i]:,}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, rf_only[i], f"{rf_only[i]:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={d[s]['n']:,}, {d[s]['n_discordant']:,} discordant)" for s in splits])
    ax.set_ylabel("discordant positions")
    ax.set_ylim(0, max(cnn_only) * 1.28)  # headroom so the legend clears the bars
    ax.set_title("Exact McNemar, CNN vs. RF (seed 46)\n"
                 "$p<10^{-300}$ on both splits, independently")
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(alpha=0.3, axis="y")

    for i, s in enumerate(splits):
        r = d[s]
        parts = [r["both_right"], r["a_right_b_wrong"], r["a_wrong_b_right"], r["both_wrong"]]
        bottom = 0
        for val, lab, col in zip(
            parts,
            ["both right", "CNN only", "RF only", "both wrong"],
            ["#55A868", "#8172B3", "#DD8452", "#C44E52"],
        ):
            ax2.bar(i, 100.0 * val / r["n"], 0.5, bottom=bottom, color=col,
                    label=lab if i == 0 else None)
            bottom += 100.0 * val / r["n"]
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits)
    ax2.set_ylabel("share of positions (%)")
    ax2.set_ylim(0, 118)
    ax2.set_title("Most positions agree; the ~5% that disagree\nsplit about 4:1 toward the CNN")
    ax2.legend(fontsize=8, loc="upper center", ncol=4, columnspacing=0.8, handlelength=1.2)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, out_path)


def fig_value_vs_baseline(out_path, args):
    """Why baseline choice is a methodological decision, not a formality: two of
    five models lose to a one-line rule on the value task."""
    plt = _plt()
    test = _load("final_test_metrics.json")
    means = {m: _mean_sd([r["value_acc"] for r in test if r.get("model") == m])[0] for m in MODELS}
    sds = {m: _mean_sd([r["value_acc"] for r in test if r.get("model") == m])[1] for m in MODELS}
    order = sorted(MODELS, key=lambda m: -means[m])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colours = ["#55A868" if means[m] >= SIDE_TO_MOVE_TEST else "#C44E52" for m in order]
    ax.bar(order, [means[m] for m in order], 0.6,
           yerr=[sds[m] for m in order], capsize=4, color=colours)
    ax.axhline(SIDE_TO_MOVE_TEST, color="black", linestyle="--", linewidth=1.2,
               label=f'side-to-move baseline ({SIDE_TO_MOVE_TEST * 100:.2f}%)')
    ax.axhline(0.506, color="grey", linestyle=":", linewidth=1.2,
               label="marginal-majority baseline (50.6%) -- the wrong one")
    for i, m in enumerate(order):
        d = 100 * (means[m] - SIDE_TO_MOVE_TEST)
        ax.text(i, means[m] + sds[m] + 0.008, f"{d:+.1f} pts", ha="center", fontsize=8)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("win/loss sign accuracy (test, 5 seeds)")
    ax.set_title("Value task against the correct baseline\n"
                 "logreg and SVM lose to a one-line rule")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_path)


def fig_tournament_elo(out_path, args):
    """The largest effect in the project, and it is not a modelling result:
    search over identical frozen weights."""
    plt = _plt()
    t = _load("tournament_track_a.json")
    agents = ["random", "greedy", "mcts"]
    elo = [t["elo"][a] for a in agents]
    lo = [t["elo"][a] - t["elo_ci95"][a][0] for a in agents]
    hi = [t["elo_ci95"][a][1] - t["elo"][a] for a in agents]
    labels = ["random", "greedy\n(argmax)", f"mcts\n({t['n_simulations']} sim)"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax.bar(labels, elo, 0.55, yerr=[lo, hi], capsize=5,
           color=["#999999", "#4C72B0", "#8172B3"])
    for i, e in enumerate(elo):
        ax.text(i, e + hi[i] + 25, f"{e:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("Bradley--Terry Elo (anchored at random)")
    ax.set_title(f"{t['n_games']}-game round-robin, {t['opening_plies']} random opening plies\n"
                 "greedy and mcts wrap identical frozen weights")
    ax.grid(alpha=0.3, axis="y")

    h2h = t["head_to_head"]
    pairs = [("greedy_vs_mcts", "greedy", "mcts"), ("random_vs_mcts", "random", "mcts"),
             ("random_vs_greedy", "random", "greedy")]
    names, left, right = [], [], []
    for key, a, b in pairs:
        rec = h2h[key]
        names.append(f"{a} vs {b}")
        left.append(rec[f"{a}_score"])
        right.append(rec[f"{b}_score"])
    y = np.arange(len(names))
    ax2.barh(y, left, 0.5, color="#C44E52", label="first agent's score")
    ax2.barh(y, right, 0.5, left=left, color="#55A868", label="second agent's score")
    for i in range(len(names)):
        if left[i] > 0:  # a 0-width segment has no room for its own label
            ax2.text(left[i] / 2, i, f"{left[i]:.0f}", ha="center", va="center",
                     fontsize=8, color="white")
        ax2.text(left[i] + right[i] / 2, i, f"{right[i]:.0f}", ha="center", va="center",
                 fontsize=8, color="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("games won (50 per pair)")
    ax2.set_title("Head-to-head, reported beside Elo\n(an anchor swept without a loss breaks Elo)")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------------------------ Track B figures

def _probe_stats(analysis, latent):
    """Probe/control R^2 per statistic at one latent: (mean, SD, control mean)
    across the 3 autoencoder seeds.

    The SD matters and is plotted. `main.tex` Table VI quotes **seed 42 alone**
    while `TRAINING_RESULTS.md` SS17.2 quotes the **3-seed mean**, and neither
    states which -- for `num_groups` that is 0.161 vs 0.234. Showing the spread
    makes the figure agree with both rather than silently picking a side."""
    per_seed = analysis[str(latent)]["per_seed"]
    stats = sorted(next(iter(per_seed.values()))["probes"].keys())
    out = {}
    for s in stats:
        probe = [per_seed[k]["probes"][s]["probe_r2"] for k in per_seed]
        control = [per_seed[k]["probes"][s]["control_r2"] for k in per_seed]
        out[s] = (float(np.mean(probe)), float(np.std(probe, ddof=1)), float(np.mean(control)))
    return out


def fig_probe_r2(out_path, args):
    """Finding 6: reconstruction and supervision preserve different information.
    Every bar must clear its own control, or it measures probe capacity."""
    plt = _plt()
    analysis = _load("unsupervised_track_b_analysis.json")
    bridge = _load("unsupervised_track_b_bridge.json")
    ae64, ae32 = _probe_stats(analysis, 64), _probe_stats(analysis, 32)
    champ = {s: (v["probe_r2"], v["control_r2"]) for s, v in bridge["probes"].items()}

    stats = list(champ.keys())
    pretty = [s.replace("_", " ") for s in stats]
    x = np.arange(len(stats))
    w = 0.26

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w, [ae64[s][0] for s in stats], w, yerr=[ae64[s][1] for s in stats], capsize=3,
           label="autoencoder ($L{=}64$), 3 seeds", color="#4C72B0")
    ax.bar(x, [ae32[s][0] for s in stats], w, yerr=[ae32[s][1] for s in stats], capsize=3,
           label="autoencoder ($L{=}32$), 3 seeds", color="#55A868")
    ax.bar(x + w, [champ[s][0] for s in stats], w,
           label="champion CNN (64-d), single model", color="#8172B3")

    ctrl = [c for s in stats for c in (ae64[s][2], ae32[s][2], champ[s][1])]
    ax.axhspan(min(ctrl), max(ctrl), color="grey", alpha=0.35,
               label=f"control band ($R^2 \\in [{min(ctrl):.3f}, {max(ctrl):.3f}]$)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, rotation=18, ha="right")
    ax.set_ylim(-0.05, 1.18)
    ax.set_ylabel("linear-probe $R^2$")
    ax.set_title("Recovering board statistics from a 64-d latent\n"
                 "AE wins on raw statistics; every probe clears its shuffled-target control")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_path)


def fig_silhouette_sweep(out_path, args):
    """The other half of Finding 6, pointing the opposite way: the supervised
    representation clusters far more cleanly than any autoencoder run."""
    plt = _plt()
    analysis = _load("unsupervised_track_b_analysis.json")
    bridge = _load("unsupervised_track_b_bridge.json")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for latent, colour in (("64", "#4C72B0"), ("32", "#55A868")):
        for i, (seed, payload) in enumerate(sorted(analysis[latent]["per_seed"].items())):
            sweep = payload["kmeans"]["sweep"]
            ks = sorted(int(k) for k in sweep)
            ax.plot(ks, [sweep[str(k)] for k in ks], color=colour, alpha=0.75, linewidth=1.1,
                    label=f"autoencoder $L{{=}}{latent}$" if i == 0 else None)

    sweep = bridge["kmeans"]["sweep"]
    ks = sorted(int(k) for k in sweep)
    ax.plot(ks, [sweep[str(k)] for k in ks], color="#8172B3", linewidth=2.4,
            marker="o", markersize=4, label="champion CNN (64-d)")
    best_k, best_s = bridge["kmeans"]["k"], bridge["kmeans"]["silhouette"]
    ax.annotate(f"peak $k{{=}}{best_k}$, silhouette {best_s:.3f}",
                xy=(best_k, best_s), xytext=(best_k + 4.5, best_s - 0.055),
                arrowprops=dict(arrowstyle="->", linewidth=0.9), fontsize=8)
    ax.set_ylim(top=best_s * 1.12)  # keep the annotation clear of the title

    ax.set_xlabel("$k$ (k-means)")
    ax.set_ylabel("silhouette score")
    ax.set_title("Cluster geometry inverts the probe result\n"
                 "champion peaks at $k{=}3$; no AE run finds an interior peak up to $k{=}20$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def fig_ae_training_curves(out_path, args):
    """Finding 5: latent 32 is a genuine capacity bottleneck, not undertraining
    -- five orders of magnitude apart, with train and holdout tracking together."""
    plt = _plt()
    runs = _load("unsupervised_track_b_training.json")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for latent, colour in ((64, "#4C72B0"), (32, "#C44E52")):
        for i, run in enumerate([r for r in runs if r["latent"] == latent]):
            hist = run["history"]
            ep = [h["epoch"] for h in hist]
            ax.plot(ep, [h["holdout_loss"] for h in hist], color=colour, alpha=0.85, linewidth=1.2,
                    label=f"latent {latent} (holdout)" if i == 0 else None)
            ax.plot(ep, [h["train_loss"] for h in hist], color=colour, alpha=0.45, linewidth=0.9,
                    linestyle="--", label=f"latent {latent} (train)" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE reconstruction loss (log scale)")
    ax.set_title("Autoencoder reconstruction, 3 seeds per latent\n"
                 "$L{=}64$ reaches near-lossless; $L{=}32$ plateaus 5 orders of magnitude higher")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, out_path)


def fig_ae_init_ablation(out_path, args):
    """Finding 7, plotted the way it should be read: a consistent positive trend
    that does not reach significance at n=5. Paired by seed."""
    plt = _plt()
    d = _load("unsupervised_track_b_ae_init_ablation.json")
    rand = {r["seed"]: r for r in d["random_init"]}
    ae = {r["seed"]: r for r in d["ae_init"]}
    seeds = sorted(rand)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, name in ((axes[0], "move_top1", "top-1 accuracy"),
                          (axes[1], "value_acc", "value sign accuracy")):
        for s in seeds:
            ax.plot([0, 1], [rand[s][key], ae[s][key]], marker="o", markersize=5,
                    color="#999999", linewidth=1, alpha=0.8)
        rm = float(np.mean([rand[s][key] for s in seeds]))
        am = float(np.mean([ae[s][key] for s in seeds]))
        ax.plot([0, 1], [rm, am], marker="D", markersize=8, color="#C44E52", linewidth=2.4,
                label=f"mean  ({rm:.4f} $\\rightarrow$ {am:.4f})")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["random init", "AE init"])
        ax.set_ylabel(name)
        ax.set_title(f"{name}\n(1,000 games, N=2, 5 seeds, paired)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("AE-weight transfer: a consistent trend, not a significant result "
                 "($p{=}0.20$ / $p{=}0.11$)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_path)


def fig_seed_and_ci_scales(out_path, args):
    """The two uncertainties this project reports are different quantities, and
    a claim is only interesting if it clears both."""
    plt = _plt()
    test = _load("final_test_metrics.json")
    x = np.arange(len(MODELS))
    seed_sd = [_mean_sd([r["move_top1"] for r in test if r.get("model") == m])[1] for m in MODELS]
    ci_w = [float(np.mean([r["move_top1_ci_hi"] - r["move_top1_ci_lo"]
                           for r in test if r.get("model") == m])) for m in MODELS]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    w = 0.36
    ax.bar(x - w / 2, seed_sd, w, label="across-seed SD (retraining noise)", color="#4C72B0")
    ax.bar(x + w / 2, ci_w, w, label="mean 95% bootstrap CI width (sampling noise)", color="#DD8452")
    cnn_rf = abs(
        float(np.mean([r["move_top1"] for r in test if r.get("model") == "cnn"]))
        - float(np.mean([r["move_top1"] for r in test if r.get("model") == "rf"]))
    )
    ax.axhline(cnn_rf, color="#C44E52", linestyle="--", linewidth=1.4,
               label=f"CNN $-$ RF top-1 gap ({cnn_rf:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("top-1 accuracy units (test split)")
    ax.set_title("Two different uncertainties, and the effect they must clear\n"
                 "bootstrap CIs resample games, not positions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_path)


FIGURES = {
    # name: (function, heavy?)
    "class_distribution": (fig_class_distribution, True),
    "learning_curve": (fig_learning_curve, False),
    "recall_heatmap": (fig_recall_heatmap, True),
    "factor_a_interaction": (fig_factor_a_interaction, False),
    "factor_a_interaction_normalised": (fig_factor_a_interaction_normalised, False),
    "data_redundancy": (fig_data_redundancy, False),
    "compute_budget": (fig_compute_budget, False),
    "val_vs_test": (fig_val_vs_test, False),
    "anova_variance": (fig_anova_variance, False),
    "mcnemar_discordance": (fig_mcnemar_discordance, False),
    "value_vs_baseline": (fig_value_vs_baseline, False),
    "seed_and_ci_scales": (fig_seed_and_ci_scales, False),
    "tournament_elo": (fig_tournament_elo, False),
    "probe_r2": (fig_probe_r2, False),
    "silhouette_sweep": (fig_silhouette_sweep, False),
    "ae_training_curves": (fig_ae_training_curves, False),
    "ae_init_ablation": (fig_ae_init_ablation, False),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="Docs/figures")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--only", action="append", default=[],
                    help="generate just this figure (repeatable)")
    ap.add_argument("--skip-heavy", action="store_true",
                    help="skip the two figures that read h5/checkpoints")
    ap.add_argument("--list", action="store_true", help="list figure names and exit")
    ap.add_argument("--train-h5", default="data/processed/train.h5")
    ap.add_argument("--val-h5", default="data/processed/val.h5")
    ap.add_argument("--champion", default="models/supervised/cnn_enc4_seed46_vol107969_none.pt")
    args = ap.parse_args()

    if args.list:
        for name, (_, heavy) in FIGURES.items():
            print(f"{name}{'  [heavy]' if heavy else ''}")
        return

    unknown = [n for n in args.only if n not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s): {unknown}\nknown: {sorted(FIGURES)}")

    names = args.only or list(FIGURES)
    made = 0
    for name in names:
        fn, heavy = FIGURES[name]
        if heavy and args.skip_heavy:
            print(f"skipped {name} (heavy)")
            continue
        fn(os.path.join(args.outdir, f"{name}.{args.format}"), args)
        made += 1
    print(f"\n{made} figure(s) written to {args.outdir}/")


if __name__ == "__main__":
    main()
