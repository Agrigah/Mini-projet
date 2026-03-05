"""
experiments.py - Expériences complètes du mini-projet
Exécuter : python experiments.py
Les figures sont sauvegardées dans un dossier "outputs/" local.
"""
import matplotlib
matplotlib.use('Agg')  # doit être avant tout import pyplot

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

from grid       import GRIDS, visualize_grid
from astar      import ucs, greedy, astar, h_manhattan, h_zero, path_to_policy, print_result
from markov     import (build_markov, evolve, prob_goal_over_time,
                        communication_classes, classify_states,
                        absorption_analysis)
from simulation import simulate_trajectories, print_simulation_report

# ── Dossier de sortie ────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    return os.path.join(OUTPUT_DIR, filename)

ALGO_COLORS = {"UCS": "#E53935", "Greedy": "#FB8C00", "A*": "#1E88E5"}

# ─────────────────────────────────────────────────────────────
def draw_multi_paths(grid, start, goal, paths_dict, title, ax):
    rows, cols = len(grid), len(grid[0])
    ax.set_facecolor("#F8F9FA")
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color="#1a1a2e", zorder=1))
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.6, zorder=0)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.tick_params(labelsize=7, colors="#555")
    offsets = {"UCS": (-0.13, 0), "Greedy": (0, 0.13), "A*": (0.13, 0)}
    for algo, path in paths_dict.items():
        color = ALGO_COLORS[algo]; ox, oy = offsets[algo]
        for (r, c) in path:
            if (r, c) not in (start, goal):
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color=color, alpha=0.10, zorder=2))
        for i in range(len(path)-1):
            r1, c1 = path[i]; r2, c2 = path[i+1]
            ax.annotate("", xy=(c2+ox, r2+oy), xytext=(c1+ox, r1+oy),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8), zorder=4)
    ax.add_patch(plt.Circle((start[1], start[0]), 0.35, color="#2E7D32", zorder=5))
    ax.add_patch(plt.Circle((goal[1],  goal[0]),  0.35, color="#C62828", zorder=5))
    ax.text(start[1], start[0], "S", ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=6)
    ax.text(goal[1],  goal[0],  "G", ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=6)
    ax.set_xlim(-0.5, cols-0.5); ax.set_ylim(rows-0.5, -0.5)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)


# ═══════════════════════════════════════════════════════════════
#  EXP 1 — UCS / Greedy / A* sur 3 grilles
# ═══════════════════════════════════════════════════════════════
def exp1_comparison():
    print("\n" + "="*60)
    print("  EXP 1 : Comparaison UCS / Greedy / A*")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("Expérience 1 — UCS / Greedy / A* sur les 3 grilles",
                 fontsize=14, fontweight="bold", y=1.01)

    summary = []
    for ax, name in zip(axes, ["facile", "moyenne", "difficile"]):
        cfg = GRIDS[name]; grid = cfg["grid"]; start = cfg["start"]; goal = cfg["goal"]
        r_ucs = ucs(grid, start, goal)
        r_grd = greedy(grid, start, goal)
        r_ast = astar(grid, start, goal)
        print(f"\n  Grille : {name.upper()}")
        for res in [r_ucs, r_grd, r_ast]: print_result(res)
        summary.append({"name": name, "ucs": r_ucs, "greedy": r_grd, "astar": r_ast})
        paths = {"UCS": r_ucs["path"], "Greedy": r_grd["path"], "A*": r_ast["path"]}
        title = (f"Grille {name}\n"
                 f"UCS={r_ucs['cost']:.0f}  Greedy={r_grd['cost']:.0f}  A*={r_ast['cost']:.0f}")
        draw_multi_paths(grid, start, goal, paths, title, ax)

    patches = [mpatches.Patch(color=ALGO_COLORS[a], label=a) for a in ["UCS","Greedy","A*"]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11,
               frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    fpath = out("exp1_grilles.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(); print(f"  ✓ Image générée : {fpath}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, label in zip(axes,
            ["cost","nodes_dev","time_s"],
            ["Coût du chemin","Nœuds développés","Temps (s)"]):
        x = np.arange(len(summary)); w = 0.25
        ax.bar(x-w, [s["ucs"][metric]   for s in summary], w, label="UCS",    color="#E53935")
        ax.bar(x,   [s["greedy"][metric] for s in summary], w, label="Greedy", color="#FB8C00")
        ax.bar(x+w, [s["astar"][metric]  for s in summary], w, label="A*",     color="#1E88E5")
        ax.set_xticks(x); ax.set_xticklabels([s["name"] for s in summary])
        ax.set_title(label); ax.legend(); ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(integer=(metric != "time_s")))
    fig.suptitle("Expérience 1 — Comparaison des métriques", fontweight="bold")
    plt.tight_layout()
    fpath = out("exp1_comparaison.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  ✓ Image générée : {fpath}")
    return summary


# ═══════════════════════════════════════════════════════════════
#  EXP 2 — Impact de ε
# ═══════════════════════════════════════════════════════════════
def exp2_epsilon():
    print("\n" + "="*60)
    print("  EXP 2 : Impact de ε (incertitude Markov)")
    print("="*60)
    cfg = GRIDS["moyenne"]; grid = cfg["grid"]; start = cfg["start"]; goal = cfg["goal"]
    res_astar = astar(grid, start, goal)
    policy = path_to_policy(res_astar["path"], grid)
    epsilons = [0.0, 0.1, 0.2, 0.3]; N_steps = 60
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Expérience 2 — P(GOAL) selon ε", fontsize=13, fontweight="bold")
    axes = axes.flatten()
    print(f"\n  {'ε':>6}  {'P_sim':>10}  {'P_mat':>10}  {'E[T]':>8}")
    for ax, eps in zip(axes, epsilons):
        states, idx, P = build_markov(grid, policy, goal, epsilon=eps)
        pi0 = np.zeros(len(states)); pi0[idx[start]] = 1.0
        dists = evolve(pi0, P, N_steps)
        pg_mat = prob_goal_over_time(dists, idx)
        sim_res = simulate_trajectories(states, idx, P, start, N_traj=3000, max_steps=N_steps)
        print(f"  {eps:>6.1f}  {sim_res['prob_goal']:>10.4f}  {pg_mat[-1]:>10.4f}  {sim_res['mean_time_goal']:>8.2f}")
        steps = np.arange(N_steps + 1)
        ax.plot(steps, pg_mat, lw=2, color="#2196F3", label="Calcul P^n")
        ax.axhline(sim_res["prob_goal"], color="#FF5722", ls="--", lw=1.5,
                   label=f"Sim. MC ({sim_res['prob_goal']:.3f})")
        ax.fill_between(steps, pg_mat, alpha=0.15, color="#2196F3")
        ax.set_xlabel("Étapes n"); ax.set_ylabel("P(GOAL)")
        ax.set_title(f"ε = {eps}"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fpath = out("exp2_epsilon.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  ✓ Image générée : {fpath}")


# ═══════════════════════════════════════════════════════════════
#  EXP 3 — Heuristiques
# ═══════════════════════════════════════════════════════════════
def exp3_heuristics():
    print("\n" + "="*60)
    print("  EXP 3 : Heuristiques admissibles")
    print("="*60)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Expérience 3 — Nœuds développés selon l'heuristique", fontsize=13, fontweight="bold")
    for ax, name in zip(axes, ["facile", "moyenne", "difficile"]):
        cfg = GRIDS[name]; grid = cfg["grid"]; start = cfg["start"]; goal = cfg["goal"]
        r_h0 = astar(grid, start, goal, heuristic=h_zero,      w=1.0)
        r_h1 = astar(grid, start, goal, heuristic=h_manhattan, w=1.0)
        r_w2 = astar(grid, start, goal, heuristic=h_manhattan, w=2.0)
        heuristics = ["A* (h=0\n=UCS)", "A* (Manhattan)", "WA* (w=2)"]
        nodes = [r_h0["nodes_dev"], r_h1["nodes_dev"], r_w2["nodes_dev"]]
        costs = [r_h0["cost"], r_h1["cost"], r_w2["cost"]]
        bars = ax.bar(heuristics, nodes, color=["#9C27B0","#4CAF50","#FF9800"], alpha=0.85)
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"coût={cost:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Grille {name}"); ax.set_ylabel("Nœuds développés"); ax.grid(axis="y", alpha=0.3)
        print(f"\n  {name.upper()}")
        for r, l in zip([r_h0,r_h1,r_w2], heuristics):
            print(f"    {l.replace(chr(10),' '):20s} nœuds={r['nodes_dev']:4d}  coût={r['cost']:.0f}")
    plt.tight_layout()
    fpath = out("exp3_heuristiques.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  ✓ Image générée : {fpath}")


# ═══════════════════════════════════════════════════════════════
#  EXP 4 — Analyse Markov
# ═══════════════════════════════════════════════════════════════
def exp4_markov_analysis():
    print("\n" + "="*60)
    print("  EXP 4 : Analyse Markov — classes, absorption")
    print("="*60)
    cfg = GRIDS["facile"]; grid = cfg["grid"]; start = cfg["start"]; goal = cfg["goal"]
    res = astar(grid, start, goal)
    pol = path_to_policy(res["path"], grid)
    eps = 0.15
    states, idx, P = build_markov(grid, pol, goal, epsilon=eps)
    n = len(states)
    print(f"\n  Grille FACILE | ε={eps} | {n} états")
    classes = communication_classes(P, states)
    class_info = classify_states(P, states, classes)
    print(f"  Classes de communication : {len(classes)}")
    for ci in class_info:
        ptype = "PERSISTANT (absorbant)" if ci["absorbing"] else ("PERSISTANT" if ci["persistent"] else "TRANSITOIRE")
        print(f"    Classe {ci['class_id']} ({ci['size']} états) → {ptype}")
        if ci["size"] <= 8: print(f"      {ci['states']}")
    ab = absorption_analysis(P, states, idx, absorbing_states=["GOAL"])
    if ab and ab["N"] is not None:
        print(f"  Absorption vers GOAL :")
        for s in res["path"][:6]:
            if s in idx and s in ab["trans_states"]:
                ti = ab["trans_states"].index(s)
                print(f"    {s} → P(GOAL)={ab['B'][ti,0]:.4f}  E[T]={ab['t'][ti]:.2f}")
    sim = simulate_trajectories(states, idx, P, start, N_traj=5000, max_steps=200)
    print(f"  Monte-Carlo N=5000 :"); print_simulation_report(sim)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Expérience 4 — Analyse Markov (grille facile, ε={eps})", fontweight="bold")
    axes[0].hist(sim["times_goal"], bins=25, color="#42A5F5", edgecolor="white", density=True)
    axes[0].axvline(sim["mean_time_goal"], color="red", lw=2, label=f"Moy={sim['mean_time_goal']:.1f}")
    axes[0].set_xlabel("Étapes"); axes[0].set_ylabel("Densité")
    axes[0].set_title("Temps d'atteinte GOAL"); axes[0].legend(); axes[0].grid(alpha=0.3)
    pi0 = np.zeros(n); pi0[idx[start]] = 1.0
    dists = evolve(pi0, P, 100); pg_mat = prob_goal_over_time(dists, idx)
    axes[1].plot(np.arange(101), pg_mat, lw=2.5, color="#1565C0", label="P^n")
    axes[1].set_xlabel("Étapes n"); axes[1].set_ylabel("P(GOAL)")
    axes[1].set_title("Convergence vers GOAL"); axes[1].grid(alpha=0.3)
    axes[1].legend(); axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    fpath = out("exp4_markov.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  ✓ Image générée : {fpath}")




# ═══════════════════════════════════════════════════════════════
#  AFFICHAGE TERMINAL — Matrice P(i,j)
# ═══════════════════════════════════════════════════════════════
def afficher_matrice_P(epsilon=0.1, grille="facile"):
    cfg = GRIDS[grille]; grid = cfg["grid"]; start = cfg["start"]; goal = cfg["goal"]
    res = astar(grid, start, goal)
    policy = path_to_policy(res["path"], grid)
    states, idx, P = build_markov(grid, policy, goal, epsilon=epsilon)
    n = len(states); labels = [str(s) for s in states]; W = 8
    print("\n" + "="*60)
    print(f"  MATRICE P(i,j)  |  ε = {epsilon}  |  grille = {grille}")
    print(f"  {n} états  —  Σ lignes = 1")
    print("="*60)
    header = f"{'':>12}" + "".join(f"{l:>{W}}" for l in labels) + "   | SOMME"
    print(header); print("-" * len(header))
    for i in range(n):
        row_str = f"{labels[i]:>12}"
        for v in P[i]:
            if v == 0.0:    row_str += f"{'·':>{W}}"
            elif v == 1.0:  row_str += f"{'1':>{W}}"
            else:            row_str += f"{v:{W}.4f}"
        total = P[i].sum()
        ok = "✓" if abs(total - 1.0) < 1e-9 else "✗"
        row_str += f"   | {total:.4f} {ok}"
        print(row_str)
    print("-" * len(header))
    print(f"  Toutes les sommes = 1 : {'✓' if np.allclose(P.sum(axis=1), 1) else '✗'}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  MINI-PROJET : A* + CHAÎNES DE MARKOV")
    print("  Planification robuste sur grille")
    print("█"*60)
    print(f"\n  Dossier de sortie : {OUTPUT_DIR}\n")

    exp1_comparison()
    exp2_epsilon()
    exp3_heuristics()
    exp4_markov_analysis()
    

    print("\n" + "-"*60)
    print("  MATRICES P(i,j) dans le terminal :")
    for eps in [0.0, 0.1, 0.2, 0.3]:
        afficher_matrice_P(epsilon=eps, grille="facile")

    print("\n" + "="*60)
    print("  Toutes les expériences terminées ✓")
    print(f"  Images PNG générées dans : {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            print(f"    • {f}")
    print("="*60 + "\n")