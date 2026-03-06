# Planification Robuste sur Grille — A* + Chaînes de Markov
**Mini-Projet Intelligence Artificielle — Mars 2026**  
*Planification heuristique + Modélisation stochastique sur grille 2D*

---

## Description

Ce projet implémente un système de planification robuste sur grille 2D en deux étapes :

1. **Recherche heuristique** (UCS, Greedy, A*, Weighted A*) pour trouver un chemin optimal
2. **Chaînes de Markov** pour modéliser l'incertitude d'exécution et évaluer la robustesse probabiliste du plan

---

## Structure du projet

```
├── grid.py                # Définition des grilles (facile, moyenne, difficile)
├── astar.py               # Algorithmes : UCS, Greedy, A*, Weighted A*
├── markov.py              # Modèle Markovien : P, π^(n), absorption, Monte-Carlo
├── simulation.py          # Simulation Monte-Carlo des trajectoires
├── experiments.py         # Expériences E1–E4 + figures automatiques
├── notebook.ipynb         # Notebook interactif (toutes les expériences + sélecteur ε)
├── outputs/               # Graphiques générés automatiquement
│   ├── exp1_grilles.png
│   ├── exp1_comparaison.png
│   ├── exp2_epsilon.png
│   ├── exp3_heuristiques.png
│   ├── exp4_markov.png
│   └── exp5_matrices_P.png
└── README.md
```

---

## Installation

```bash
pip install numpy matplotlib
jupyter notebook notebook.ipynb
```

---

## Modélisation mathématique

### Fonction d'évaluation A*

| Algorithme   | f(n)          |
|--------------|---------------|
| UCS          | g(n)          |
| Greedy       | h(n)          |
| A*           | g(n) + h(n)   |
| Weighted A*  | g(n) + w·h(n) |

### Matrice de transition Markov — Calcul analytique (ε = 0.1)

- `(1 − ε)` : action prévue exécutée correctement
- `(ε / k)` : déviation vers chaque voisin latéral valide (k voisins)
- `ε × (n_bloqués / 3)` : rebond sur obstacle → s'ajoute à P(i,i)
- `π^(n) = π^(0) · P^n` (Chapman-Kolmogorov)

### Analyse d'absorption

```
N = (I − Q)⁻¹    →    B = N·R  (proba absorption)    →    t = N·1  (temps moyen)
```

---

## Expériences

### E1 — UCS / Greedy / A* sur 3 grilles

| Grille          | Algo   | Coût | Nœuds | Temps (ms) |
|-----------------|--------|------|-------|------------|
| Facile (5×5)    | UCS    | 8    | 21    | 0.277      |
| Facile (5×5)    | Greedy | 8    | 9     | 0.068      |
| Facile (5×5)    | A*     | 8    | 21    | 0.185      |
| Moyenne (7×7)   | UCS    | 12   | 34    | 0.260      |
| Moyenne (7×7)   | Greedy | 12   | 13    | 0.070      |
| Moyenne (7×7)   | A*     | 12   | 19    | 0.100      |
| Difficile (10×10)| UCS   | 18   | 59    | 0.412      |
| Difficile (10×10)| Greedy| 22   | 24    | 0.119      |
| Difficile (10×10)| A*    | 18   | 35    | 0.173      |

Greedy explore **3–5× moins** de nœuds. A* garantit l'optimalité avec h admissible.

---

### E2 — Impact de ε sur P(GOAL)

Plan A* fixé, grille facile 5×5, N = 3 000 trajectoires Monte-Carlo.

| ε   | P(GOAL) | E[T] (étapes) | σ(T)  | Timeout |
|-----|---------|---------------|-------|---------|
| 0.0 | 1.0000  | 12.0          | 0.0   | 0.000   |
| 0.1 | 0.9113  | 16.23         | ~3.1  | 0.089   |
| 0.2 | 0.8487  | 20.80         | ~5.2  | 0.151   |
| 0.3 | 0.7833  | 26.03         | ~7.4  | 0.217   |

À ε = 0.3, seulement **78% des trajectoires** atteignent le but → plan déterministe fragile.

---

### E3 — Comparaison heuristiques admissibles

| Grille           | h = 0 (nœuds) | h = Manhattan | Remarque                     |
|------------------|---------------|---------------|------------------------------|
| Facile (5×5)     | 21            | 21            | Équivalent sur grille simple |
| Moyenne (7×7)    | 34            | 19            | −44% avec Manhattan          |
| Difficile (10×10)| 59            | 35            | −41% avec Manhattan          |

Manhattan réduit les expansions de **41%** sur labyrinthe. Les deux heuristiques sont admissibles → même coût optimal garanti.

---

### E4 — Analyse Markov : classes de communication et absorption

Grille facile (5×5), ε = 0.15, politique A* — 22 états (21 libres + GOAL).

| Classe | Taille   | Type                  | Description                       |
|--------|----------|-----------------------|-----------------------------------|
| C₀     | 1 état   | Transitoire           | État isolé (4,4)                  |
| C₁     | 20 états | Transitoire           | Tous les états libres accessibles |
| C₂     | 1 état   | Persistant absorbant  | GOAL — attracteur irréversible    |

**Simulation Monte-Carlo (N = 5 000) :**

| ε    | P(GOAL) | E[T] (étapes) | σ(T)  |
|------|---------|---------------|-------|
| 0.15 | 0.9748  | 16.11         | 23.07 |

Calcul matriciel Pⁿ et simulation Monte-Carlo sont **parfaitement concordants** (écart < 1%).

---

### E5 — Matrice de transition P(i,j) — Calcul analytique ε = 0.1

Grille facile 5×5 — 22 états — chemin A* : `(0,0)→(1,0)→(2,0)→(3,0)→(4,0)→(4,1)→(4,2)→(4,3)→(4,4)`

| État i    | Action π(i)  | Cible j*   | k lat. | P(i, j*)      | P(i, j_lat)         | P(i,i) rebond  | Σ      |
|-----------|--------------|------------|--------|---------------|---------------------|----------------|--------|
| ★ (0,0)   | ↓ → (1,0)    | (1,0)      | 1      | 9/10 = 0.9000 | ε/1=0.1 → (0,1)     | ε×2/3 ≈ 0.0667 | 1.00 ✓ |
| ★ (1,0)   | ↓ → (2,0)    | (2,0)      | 0      | 9/10 = 0.9000 | —                   | ε×2/3 ≈ 0.0667 | 1.00 ✓ |
| ★ (2,0)   | ↓ → (3,0)    | (3,0)      | 1      | 9/10 = 0.9000 | ε/1=0.1 → (2,1)     | ε×1/3 ≈ 0.0333 | 1.00 ✓ |
| ★ (3,0)   | ↓ → (4,0)    | (4,0)      | 0      | 9/10 = 0.9000 | —                   | ε×2/3 ≈ 0.0667 | 1.00 ✓ |
| ★ (4,0)   | → → (4,1)    | (4,1)      | 1      | 9/10 = 0.9000 | ε/1=0.1 → (3,0)     | ε×2/3 ≈ 0.0667 | 1.00 ✓ |
| ★ (4,1)   | → → (4,2)    | (4,2)      | 0      | 9/10 = 0.9000 | —                   | ε×2/3 ≈ 0.0667 | 1.00 ✓ |
| ★ (4,2)   | → → (4,3)    | (4,3)      | 1      | 9/10 = 0.9000 | ε/1=0.1 → (3,2)     | ε×1/3 ≈ 0.0333 | 1.00 ✓ |
| ★ (4,3)   | → → GOAL     | GOAL       | 0      | 9/10 = 0.9000 | —                   | ε×2/3 ≈ 0.0667 | 1.00 ✓ |
| GOAL      | absorbant    | GOAL       | —      | 1.0000        | —                   | —              | 1.00 ✓ |
| · autres  | sans policy  | reste      | —      | P(i,i) = 1   | —                   | —              | 1.00 ✓ |

> ★ = états sur le chemin A*  |  · = états hors chemin (P(i,i) = 1)

---

## 💻 Utilisation rapide

```python
from grid import GRIDS
from astar import astar, path_to_policy
from markov import build_markov, evolve, prob_goal_over_time
from simulation import simulate_trajectories

# 1. Planification A*
cfg = GRIDS["facile"]
grid, start, goal = cfg["grid"], cfg["start"], cfg["goal"]
result = astar(grid, start, goal)
print(f"Chemin : {result['path']}  |  Coût : {result['cost']}")

# 2. Modèle Markov (ε = 0.1)
policy = path_to_policy(result["path"], grid)
states, idx, P = build_markov(grid, policy, goal, epsilon=0.1)

# 3. Calcul analytique P^n
import numpy as np
pi0 = np.zeros(len(states)); pi0[idx[start]] = 1.0
dists = evolve(pi0, P, n_steps=60)
pg = prob_goal_over_time(dists, idx)
print(f"P(GOAL, n=60) = {pg[-1]:.4f}")

# 4. Simulation Monte-Carlo
sim = simulate_trajectories(states, idx, P, start, N_traj=5000)
print(f"P(GOAL) = {sim['prob_goal']:.3f}  |  E[T] = {sim['mean_time_goal']:.1f} étapes")
```

---

## 🔑 Conclusions

| # | Résultat |
|---|----------|
| 1 | A* avec h = Manhattan garantit l'optimalité avec exploration réduite (−41% vs UCS) |
| 2 | Greedy explore 3–5× moins de nœuds sans garantie d'optimalité |
| 3 | P(GOAL) chute de 1.0 → 0.78 quand ε passe de 0 à 0.3 |
| 4 | Calcul matriciel Pⁿ et simulation Monte-Carlo concordent à < 1% |
| 5 | La matrice P(i,j) est stochastique par ligne : ∑ⱼ P(i,j) = 1 ∀i ✓ |
| 6 | Chemin A* optimal ≠ robuste sous incertitude → nécessite modélisation Markov |

---

## 📚 Références

1. Russell, S. & Norvig, P. — *Artificial Intelligence: A Modern Approach*, 4th ed. Pearson, 2020.
2. Hart, P. E., Nilsson, N. J., & Raphael, B. — *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*. IEEE Trans. SSC, 4(2), 1968.
3. Puterman, M. L. — *Markov Decision Processes*. Wiley, 1994.
4. Kemeny, J. G. & Snell, J. L. — *Finite Markov Chains*. Springer, 1976.

---

*Mini-Projet — Master SDIA | ENSET Mohammedia — Université Hassan II de Casablanca — 2025–2026*  
**Étudiante :** Agrigah Aya | **Encadrant :** Mohamed MESTARI
