"""Round-robin tournament across the three agent tiers (EXECUTION_Phase2.md task 2.4).

Pits a persisted champion CNN against itself under two decision rules -- plain
argmax (`GreedyAgent`) and PUCT search (`MCTSAgent`) -- plus a uniform-random
floor. Because both wrapped agents share the *same* weights, any Elo gap between
them is attributable to search alone, which is exactly what gate P2-G2 asks.

MCTS here is evaluation-time only: weights never change and no simulated game is
recorded as training data (this document's SS1).

Usage:
  python -m eval.run_tournament --model-path models/supervised/cnn_enc4_seed46_vol107969_none.pt \
      --encoding 4 --games-per-colour 25
"""

from __future__ import annotations

import argparse
import json
import os

from training.supervised.models.cnn_model import load_cnn

from .agents import GreedyAgent, MCTSAgent, UniformRandomAgent
from .elo import bootstrap_elo_ci, bradley_terry_strengths, to_elo
from .tournament import round_robin, win_rate


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="a .pt written by trainer.py --save-model-dir")
    ap.add_argument("--encoding", type=int, required=True, help="feature-plane count the model was trained with")
    ap.add_argument("--games-per-colour", type=int, default=25, help="per pair, per colour; total games = 6x this")
    ap.add_argument("--n-simulations", type=int, default=100)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--min-ply-before-pass", type=int, default=20)
    ap.add_argument(
        "--opening-plies", type=int, default=4,
        help="random legal plies played before the agents take over. Both model-backed agents "
        "are deterministic, so at 0 every repetition of a pairing replays the identical game "
        "and the effective sample size is 1, not --games-per-colour",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="results/tournament_track_a.json")
    return ap.parse_args()


def main():
    args = parse_args()
    model = load_cnn(args.model_path, device=args.device)

    agents = {
        "random": UniformRandomAgent(seed=args.seed),
        "greedy": GreedyAgent(
            model, encoding=args.encoding, min_ply_before_pass=args.min_ply_before_pass
        ),
        "mcts": MCTSAgent(
            model,
            encoding=args.encoding,
            n_simulations=args.n_simulations,
            c_puct=args.c_puct,
            min_ply_before_pass=args.min_ply_before_pass,
        ),
    }
    names = list(agents)

    print(
        f"Round-robin: {args.games_per_colour} games per colour per pair "
        f"({6 * args.games_per_colour} games total), MCTS at {args.n_simulations} sims/move, "
        f"{args.opening_plies} random opening plies"
    )
    results = round_robin(
        agents, games_per_colour=args.games_per_colour, seed=args.seed,
        opening_plies=args.opening_plies,
    )

    strength = bradley_terry_strengths(results, names)
    elo = to_elo(strength, anchor="random")
    ci = bootstrap_elo_ci(results, names, anchor="random", seed=args.seed)

    print(f"\n=== Tournament results ({len(results)} games) ===")
    print(f"{'agent':>8} {'win rate':>10} {'Elo':>9} {'95% CI':>20}")
    for n in sorted(names, key=lambda k: -elo[k]):
        lo, hi = ci[n]
        print(f"{n:>8} {win_rate(results, n):>10.3f} {elo[n]:>9.1f} {f'[{lo:.1f}, {hi:.1f}]':>20}")

    # Head-to-head is reported alongside Elo because Elo degenerates here: a strong
    # champion may never lose a single game to `random`, which drives random's
    # Bradley-Terry strength to zero and every Elo relative to it to +inf. The
    # mcts-vs-greedy record stays meaningful regardless, and it is the pair that
    # actually answers gate P2-G2 (same weights, search vs argmax).
    h2h = {}
    print("\n=== Head-to-head (row's record vs column) ===")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair = [r for r in results if {r.black, r.white} == {a, b}]
            a_w = sum(
                1.0 if (r.winner == "black" and r.black == a) or (r.winner == "white" and r.white == a)
                else 0.5 if r.winner == "tie" else 0.0
                for r in pair
            )
            h2h[f"{a}_vs_{b}"] = {"games": len(pair), f"{a}_score": a_w, f"{b}_score": len(pair) - a_w}
            print(f"  {a:>7} {a_w:5.1f} - {len(pair) - a_w:<5.1f} {b:<7}  ({len(pair)} games)")

    payload = {
        "model_path": args.model_path,
        "encoding": args.encoding,
        "n_simulations": args.n_simulations,
        "c_puct": args.c_puct,
        "min_ply_before_pass": args.min_ply_before_pass,
        "games_per_colour": args.games_per_colour,
        "opening_plies": args.opening_plies,
        "n_games": len(results),
        "seed": args.seed,
        "win_rate": {n: win_rate(results, n) for n in names},
        "elo": elo,
        "elo_ci95": {n: list(ci[n]) for n in names},
        "head_to_head": h2h,
        "games": [
            {
                "black": r.black, "white": r.white, "winner": r.winner,
                "black_score": r.black_score, "white_score": r.white_score, "plies": r.plies,
            }
            for r in results
        ],
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
