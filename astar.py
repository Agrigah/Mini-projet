"""
astar.py - Implémentation de UCS, Greedy Best-First et A*
"""
import heapq
import time
import tracemalloc
from grid import get_neighbors, DIRECTIONS_4

# ─────────────────────────────────────────────────────────────
#  Heuristiques
# ─────────────────────────────────────────────────────────────
def h_zero(pos, goal):
    """h=0  →  dégénère en UCS (utilisé pour test)."""
    return 0

def h_manhattan(pos, goal):
    """Distance Manhattan — admissible si coût unitaire = 1."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def h_chebyshev(pos, goal):
    """Distance de Chebyshev (admissible en 8-voisins, ici sur-estime
    légèrement en 4-voisins → sert d'exemple NON admissible)."""
    return max(abs(pos[0] - goal[0]), abs(pos[1] - goal[1]))

# ─────────────────────────────────────────────────────────────
#  Algorithme générique : UCS / Greedy / A*
# ─────────────────────────────────────────────────────────────
def search(grid, start, goal,
           heuristic=h_manhattan,
           weight_g=1.0,
           weight_h=1.0,
           algo_name="A*"):
    """
    Recherche sur grille.

    f(n) = weight_g * g(n) + weight_h * h(n)
      - UCS    : weight_g=1, weight_h=0
      - Greedy : weight_g=0, weight_h=1
      - A*     : weight_g=1, weight_h=1
      - WA*    : weight_g=1, weight_h=w>1

    Retourne un dict avec chemin, coût, métriques.
    """
    # ── Structures ──────────────────────────────────────────
    # heap : (f, tie-breaker, g, state)
    counter   = 0
    open_heap = []
    g_cost    = {start: 0.0}
    came_from = {start: None}

    h0 = heuristic(start, goal)
    f0 = weight_g * 0 + weight_h * h0
    heapq.heappush(open_heap, (f0, counter, 0.0, start))

    closed    = set()
    open_set  = {start}   # pour suivi taille OPEN
    nodes_dev = 0          # nœuds développés

    t_start = time.perf_counter()
    tracemalloc.start()

    found = False
    while open_heap:
        f, _, g, node = heapq.heappop(open_heap)
        open_set.discard(node)

        if node in closed:
            continue
        closed.add(node)
        nodes_dev += 1

        if node == goal:
            found = True
            break

        for nb in get_neighbors(grid, node):
            new_g = g + 1.0          # coût unitaire
            if new_g < g_cost.get(nb, float('inf')):
                g_cost[nb]    = new_g
                came_from[nb] = node
                h              = heuristic(nb, goal)
                f_nb           = weight_g * new_g + weight_h * h
                counter       += 1
                heapq.heappush(open_heap, (f_nb, counter, new_g, nb))
                open_set.add(nb)

    elapsed  = time.perf_counter() - t_start
    _, peak  = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── Reconstruction du chemin ─────────────────────────────
    path = []
    if found:
        node = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()

    return {
        "algo"       : algo_name,
        "found"      : found,
        "path"       : path,
        "cost"       : g_cost.get(goal, float('inf')),
        "nodes_dev"  : nodes_dev,
        "open_size"  : len(open_set),
        "time_s"     : elapsed,
        "mem_kb"     : peak / 1024,
    }

# ─────────────────────────────────────────────────────────────
#  Raccourcis nommés
# ─────────────────────────────────────────────────────────────
def ucs(grid, start, goal):
    return search(grid, start, goal,
                  heuristic=h_zero, weight_g=1.0, weight_h=0.0,
                  algo_name="UCS")

def greedy(grid, start, goal, heuristic=h_manhattan):
    return search(grid, start, goal,
                  heuristic=heuristic, weight_g=0.0, weight_h=1.0,
                  algo_name="Greedy")

def astar(grid, start, goal, heuristic=h_manhattan, w=1.0):
    name = f"A* (w={w})" if w != 1.0 else "A*"
    return search(grid, start, goal,
                  heuristic=heuristic, weight_g=1.0, weight_h=w,
                  algo_name=name)

# ─────────────────────────────────────────────────────────────
#  Politique induite par le chemin A*
# ─────────────────────────────────────────────────────────────
def path_to_policy(path, grid):
    """
    Transforme le chemin A* en politique déterministe :
      policy[state] = direction (dr, dc)
    Le dernier état (goal) est auto-absorbant.
    """
    from grid import DIRECTIONS_4
    policy = {}
    for i in range(len(path) - 1):
        s = path[i]
        s_next = path[i+1]
        dr = s_next[0] - s[0]
        dc = s_next[1] - s[1]
        policy[s] = (dr, dc)
    # goal : rester sur place
    policy[path[-1]] = (0, 0)
    return policy

# ─────────────────────────────────────────────────────────────
#  Affichage résumé
# ─────────────────────────────────────────────────────────────
def print_result(res):
    print(f"  [{res['algo']:12s}]  "
          f"Coût={res['cost']:.0f}  "
          f"Nœuds={res['nodes_dev']:4d}  "
          f"Temps={res['time_s']*1000:.2f}ms  "
          f"Mém={res['mem_kb']:.1f}KB  "
          f"{'✓' if res['found'] else '✗'}")