"""
grid.py - Définition de la grille, obstacles, visualisation
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ─────────────────────────────────────────────
#  Grilles prédéfinies  (0=libre, 1=obstacle)
# ─────────────────────────────────────────────
GRIDS = {
    "facile": {
        "grid": [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        "start": (0, 0),
        "goal":  (4, 4),
    },
    "moyenne": {
        "grid": [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        "start": (0, 0),
        "goal":  (6, 6),
    },
    "difficile": {
        "grid": [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        ],
        "start": (0, 0),
        "goal":  (9, 9),
    },
}

DIRECTIONS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # haut, bas, gauche, droite
DIR_NAMES    = {(-1,0):"H", (1,0):"B", (0,-1):"G", (0,1):"D"}

def get_neighbors(grid, pos):
    """Retourne les voisins libres (4-connexité) d'une cellule."""
    rows, cols = len(grid), len(grid[0])
    r, c = pos
    neighbors = []
    for dr, dc in DIRECTIONS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

def is_free(grid, pos):
    r, c = pos
    rows, cols = len(grid), len(grid[0])
    return 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0

def visualize_grid(grid, start, goal, path=None, title="Grille", ax=None):
    """Affiche la grille avec chemin optionnel."""
    g = np.array(grid, dtype=float)
    rows, cols = g.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, cols), max(5, rows)))

    cmap = ListedColormap(["white", "black"])
    ax.imshow(g, cmap=cmap, vmin=0, vmax=1, origin="upper")

    # Grille
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Chemin
    if path:
        for (r, c) in path:
            if (r, c) not in (start, goal):
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                           color="lightblue", alpha=0.7))
        # Flèches
        for i in range(len(path)-1):
            r1, c1 = path[i]; r2, c2 = path[i+1]
            ax.annotate("", xy=(c2, r2), xytext=(c1, r1),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

    # Start / Goal
    ax.add_patch(plt.Circle((start[1], start[0]), 0.35, color="green", zorder=3))
    ax.add_patch(plt.Circle((goal[1],  goal[0]),  0.35, color="red",   zorder=3))
    ax.text(start[1], start[0], "S", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold", zorder=4)
    ax.text(goal[1],  goal[0],  "G", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold", zorder=4)

    ax.set_title(title)
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    return ax