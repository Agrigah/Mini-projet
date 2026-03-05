"""
simulation.py - Simulation Monte-Carlo des trajectoires Markov
"""
import numpy as np
from collections import defaultdict

def simulate_trajectories(states, idx, P, start, goal_label="GOAL",
                           N_traj=5000, max_steps=500, seed=42):
    """
    Simule N_traj trajectoires à partir de 'start' selon la matrice P.

    Retourne un dict avec :
      - prob_goal        : P(atteindre GOAL)
      - mean_time_goal   : E[temps d'atteinte | atteint GOAL]
      - std_time_goal    : écart-type du temps d'atteinte
      - times_goal       : liste des temps (pour histogramme)
      - traj_lengths     : longueur de chaque trajectoire
      - prob_stuck       : fraction de trajectoires n'ayant pas atteint GOAL
    """
    rng       = np.random.default_rng(seed)
    n         = len(states)
    start_i   = idx[start]
    goal_i    = idx[goal_label]

    reached_goal  = 0
    times_goal    = []
    traj_lengths  = []

    # Pré-calcul des distributions cumulées pour échantillonnage rapide
    P_cumsum = np.cumsum(P, axis=1)

    for _ in range(N_traj):
        state = start_i
        for step in range(1, max_steps + 1):
            r      = rng.random()
            # Bisect pour trouver le prochain état
            next_s = int(np.searchsorted(P_cumsum[state], r))
            next_s = min(next_s, n - 1)  # sécurité
            state  = next_s

            if state == goal_i:
                reached_goal += 1
                times_goal.append(step)
                traj_lengths.append(step)
                break
        else:
            traj_lengths.append(max_steps)

    prob_goal       = reached_goal / N_traj
    mean_time_goal  = np.mean(times_goal) if times_goal else float('inf')
    std_time_goal   = np.std(times_goal)  if times_goal else 0.0

    return {
        "prob_goal"      : prob_goal,
        "mean_time_goal" : mean_time_goal,
        "std_time_goal"  : std_time_goal,
        "times_goal"     : times_goal,
        "traj_lengths"   : traj_lengths,
        "prob_stuck"     : 1.0 - prob_goal,
        "N_traj"         : N_traj,
    }

def print_simulation_report(sim_res):
    print(f"  Trajectoires simulées : {sim_res['N_traj']}")
    print(f"  P(atteindre GOAL)     : {sim_res['prob_goal']:.4f}")
    print(f"  Temps moyen (si GOAL) : {sim_res['mean_time_goal']:.2f} ± "
          f"{sim_res['std_time_goal']:.2f} étapes")
    print(f"  P(bloqué / timeout)   : {sim_res['prob_stuck']:.4f}")