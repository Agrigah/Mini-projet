"""
markov.py - Construction de la matrice de transition P,
            calculs probabilistes, analyse des classes,
            probabilités d'absorption.
"""
import numpy as np
from grid import DIRECTIONS_4, get_neighbors, is_free

# ─────────────────────────────────────────────────────────────
#  Construction de la matrice P
# ─────────────────────────────────────────────────────────────
GOAL_IDX = -2   # sentinelles (indices dans la liste d'états)
FAIL_IDX = -1   # (inutilisé ici, FAIL = toute collision)

def build_markov(grid, policy, goal, epsilon=0.1):
    """
    Construit la matrice de transition P à partir de la politique
    et du niveau d'incertitude ε.

    Modèle d'incertitude :
      - action voulue (selon policy) avec prob 1 - ε
      - déviation vers chaque voisin latéral avec prob ε/2 par direction
        (s'il y en a 2, sinon redistribué)
      - si la cible est un obstacle / hors grille → reste sur place

    États spéciaux : GOAL (absorbant), les états hors chemin sont aussi
    inclus si ε > 0.

    Retourne :
      states  : liste ordonnée des états (tuples)  + "GOAL"
      idx     : dict state -> index
      P       : matrice numpy (n_states × n_states)
    """
    # ── Collecte des états ──────────────────────────────────
    rows = len(grid)
    cols = len(grid[0])
    free_cells = [(r, c)
                  for r in range(rows)
                  for c in range(cols)
                  if grid[r][c] == 0]

    states = free_cells + ["GOAL"]
    idx    = {s: i for i, s in enumerate(states)}
    n      = len(states)
    P      = np.zeros((n, n))

    goal_i = idx["GOAL"]

    # ── Remplissage ─────────────────────────────────────────
    for s in free_cells:
        i = idx[s]

        if s == goal:
            # GOAL absorbant
            P[i, goal_i] = 1.0
            continue

        action = policy.get(s, (0, 0))   # (dr, dc)

        # Directions latérales = toutes directions sauf l'action voulue
        lat_dirs = [d for d in DIRECTIONS_4 if d != action and d != (0,0)]
        n_lat    = len(lat_dirs)

        # Probabilité par direction latérale
        p_lat = epsilon / n_lat if n_lat > 0 else 0.0
        p_fwd = 1.0 - epsilon

        all_moves = [(action, p_fwd)] + [(d, p_lat) for d in lat_dirs]

        for (dr, dc), prob in all_moves:
            if prob <= 0:
                continue
            nr, nc = s[0] + dr, s[1] + dc
            if is_free(grid, (nr, nc)):
                dest = (nr, nc)
                if dest == goal:
                    P[i, goal_i] += prob
                else:
                    P[i, idx[dest]] += prob
            else:
                # Collision → reste sur place
                P[i, i] += prob

    # GOAL absorbant
    P[goal_i, goal_i] = 1.0

    # ── Vérification stochasticité ──────────────────────────
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), \
        f"Matrice non stochastique ! Sommes lignes : {row_sums}"

    return states, idx, P

# ─────────────────────────────────────────────────────────────
#  Évolution de la distribution
# ─────────────────────────────────────────────────────────────
def evolve(pi0, P, n_steps):
    """
    Calcule π^(n) = π^(0) · P^n pour n = 0, 1, …, n_steps.
    Retourne un tableau (n_steps+1) × len(states).
    """
    distributions = [pi0.copy()]
    pi = pi0.copy()
    for _ in range(n_steps):
        pi = pi @ P
        distributions.append(pi.copy())
    return np.array(distributions)

def prob_goal_over_time(distributions, idx):
    """Extrait P(être dans GOAL) à chaque instant."""
    gi = idx["GOAL"]
    return distributions[:, gi]

# ─────────────────────────────────────────────────────────────
#  Analyse des classes de communication (graphe orienté)
# ─────────────────────────────────────────────────────────────
def communication_classes(P, states, threshold=1e-9):
    """
    Identifie les classes de communication par DFS sur le graphe orienté.
    Retourne une liste de sets (classes).
    """
    n = len(states)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if P[i, j] > threshold:
                adj[i].add(j)

    # Kosaraju : 2 DFS
    visited = [False] * n
    finish  = []

    def dfs1(v):
        stack = [(v, iter(adj[v]))]
        visited[v] = True
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if not visited[child]:
                    visited[child] = True
                    stack.append((child, iter(adj[child])))
            except StopIteration:
                finish.append(node)
                stack.pop()

    for i in range(n):
        if not visited[i]:
            dfs1(i)

    # Graphe transposé
    radj = {i: set() for i in range(n)}
    for i in range(n):
        for j in adj[i]:
            radj[j].add(i)

    visited2 = [False] * n
    classes  = []

    def dfs2(v, comp):
        stack = [v]
        visited2[v] = True
        comp.append(v)
        while stack:
            node = stack[-1]; stack.pop()
            for nb in radj[node]:
                if not visited2[nb]:
                    visited2[nb] = True
                    comp.append(nb)
                    stack.append(nb)

    for v in reversed(finish):
        if not visited2[v]:
            comp = []
            dfs2(v, comp)
            classes.append(set(comp))

    return classes

def classify_states(P, states, classes, threshold=1e-9):
    """
    Pour chaque classe, détermine si elle est persistante (récurrente)
    ou transitoire selon la définition : une classe est persistante si
    aucun arc ne sort vers une autre classe.
    """
    n = len(states)
    # Mapping état → classe
    state_to_class = {}
    for ci, cls in enumerate(classes):
        for s in cls:
            state_to_class[s] = ci

    results = []
    for ci, cls in enumerate(classes):
        is_persistent = True
        for i in cls:
            for j in range(n):
                if P[i, j] > threshold and state_to_class.get(j) != ci:
                    is_persistent = False
                    break
            if not is_persistent:
                break

        absorbing = (len(cls) == 1 and
                     P[list(cls)[0], list(cls)[0]] > 1 - threshold)

        results.append({
            "class_id"   : ci,
            "states"     : [states[s] for s in sorted(cls)],
            "size"       : len(cls),
            "persistent" : is_persistent,
            "absorbing"  : absorbing,
        })
    return results

# ─────────────────────────────────────────────────────────────
#  Absorption (états transitoires → états absorbants)
# ─────────────────────────────────────────────────────────────
def absorption_analysis(P, states, idx, absorbing_states):
    """
    Décompose P sous la forme canonique et calcule :
      - N = (I - Q)^{-1}   : matrice fondamentale
      - B = N · R           : probabilités d'absorption
      - t = N · 1           : temps moyen d'absorption

    absorbing_states : liste des noms d'états absorbants (ex: ["GOAL"])
    Retourne dict avec N, B, t, indices transitoires/absorbants.
    """
    abs_idx   = [idx[a] for a in absorbing_states if a in idx]
    trans_idx = [i for i in range(len(states))
                 if i not in abs_idx]

    if not trans_idx:
        return None

    Q = P[np.ix_(trans_idx, trans_idx)]
    R = P[np.ix_(trans_idx, abs_idx)]

    I = np.eye(len(trans_idx))
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = None
        B = None
        t = None
    else:
        B = N @ R
        t = N @ np.ones(len(trans_idx))

    return {
        "trans_states" : [states[i] for i in trans_idx],
        "abs_states"   : absorbing_states,
        "N"            : N,
        "B"            : B,
        "t"            : t,
        "trans_idx"    : trans_idx,
        "abs_idx"      : abs_idx,
    }