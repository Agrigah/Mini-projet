🗺️ Planification Robuste sur Grille — A* + Chaînes de Markov

Mini-Projet | Master SDIA — Systèmes Distribués et Intelligence Artificielle
Étudiante : Agrigah Aya | Encadrant : Mohamed MESTARI | Année : 2025–2026


📋 Description
Ce projet combine deux approches pour planifier un chemin optimal sur une grille 2D avec obstacles :

Recherche heuristique (UCS, Greedy, A*) — trouver le chemin le moins coûteux
Chaînes de Markov — évaluer la robustesse du plan face à l'incertitude stochastique

Le paramètre ε ∈ [0, 1] modélise l'incertitude : avec probabilité (1−ε) l'agent suit la direction prévue, et avec probabilité ε il dévie vers un voisin latéral.

📁 Structure du Projet
mini-projet/
│
├── grid.py           # Définition des 3 grilles et utilitaires
├── astar.py          # Algorithmes : UCS, Greedy, A*, heuristiques
├── markov.py         # Construction P(i,j), évolution Pⁿ, absorption
├── simulation.py     # Simulation Monte-Carlo des trajectoires
├── experiments.py    # Script principal — génère toutes les figures
├── notebook.ipynb    # Notebook Jupyter interactif
│
└── outputs/          # Figures générées automatiquement
    ├── exp1_grilles.png          # Chemins UCS/Greedy/A* superposés
    ├── exp1_comparaison.png      # Métriques comparatives
    ├── exp2_epsilon.png          # P(GOAL,n) pour 4 valeurs de ε
    ├── exp3_heuristiques.png     # Nœuds développés selon h
    ├── exp4_markov.png           # Distribution temps + convergence
    └── exp5_matrices_P.png       # Heatmap matrices P(i,j)

⚙️ Installation
Prérequis

Python 3.8+
pip

Dépendances
bashpip install numpy matplotlib
Aucune autre dépendance externe requise — tous les algorithmes sont implémentés from scratch.

🚀 Utilisation
Exécuter toutes les expériences
bashpython experiments.py
Génère 5 images PNG dans le dossier outputs/ et affiche les matrices P(i,j) dans le terminal pour ε ∈ {0.0, 0.1, 0.2, 0.3}.
Notebook interactif
bashjupyter notebook notebook.ipynb
Le notebook permet de choisir ε interactivement pour visualiser la matrice correspondante :
python# ╔══════════════════════════════════════╗
# ║  CHOISIR ε ICI  →  0.0 / 0.1 / 0.2 / 0.3  ║
EPSILON = 0.1
# ╚══════════════════════════════════════╝

📐 Modélisation Mathématique
Algorithme A*
f(n) = g(n) + h(n)
SymboleSignificationg(n)Coût réel depuis le départh(n)Heuristique (distance Manhattan)f(n)Fonction d'évaluation totale
Heuristique de Manhattan : h((x,y)) = |x − x_goal| + |y − y_goal|
→ Admissible et cohérente, garantit l'optimalité de A*.
Matrice de Transition P(i,j)
P(i, j) = (1 − ε)   si j = π(i)         [action voulue]
P(i, j) = ε / k     si j est un voisin latéral  [déviation]
Propriété stochastique : ∀i ∈ S : ∑ⱼ P(i,j) = 1
Évolution & Absorption
π⁽ⁿ⁾ = π⁽⁰⁾ · Pⁿ          [distribution à l'étape n]
N = (I − Q)⁻¹              [matrice fondamentale]
B = N · R                  [probabilités d'absorption vers GOAL]
t = N · 1                  [temps moyen d'absorption]

🗺️ Grilles Disponibles
GrilleTailleÉtats libresDépartArrivéefacile5×521(0,0)(4,4)moyenne7×735(0,0)(6,6)difficile10×1072(0,0)(9,9)

📊 Résultats Principaux
Expérience 1 — Comparaison des algorithmes
GrilleAlgoCoûtNœudsTemps (ms)Facile (5×5)UCS8210.277Facile (5×5)Greedy890.068Facile (5×5)A*8210.185Difficile (10×10)UCS18590.412Difficile (10×10)Greedy22240.119Difficile (10×10)A*18350.173

Conclusion : A* est optimal et développe 41% moins de nœuds que UCS sur la grille difficile.

Expérience 2 — Impact de ε
εP(GOAL)E[T | GOAL]0.01.000012.00 étapes0.10.911316.23 étapes0.20.848720.80 étapes0.30.783326.03 étapes

Conclusion : ε = 0.3 → seulement 78% de succès, E[T] doublé.

Matrice P(i,j) — ε = 0.1 (grille facile)
Légende :
  1 ou P(≥0.7)  →  Bleu foncé  — action principale voulue
  P(0.1–0.7)    →  Bleu moyen  — probabilité intermédiaire
  P(<0.1)       →  Bleu clair  — déviation latérale
  ·             →  Gris        — probabilité = 0
  Σ = 1.00 ✓    →  Vert        — stochasticité vérifiée

🧩 API des Modules
astar.py
pythonfrom astar import ucs, greedy, astar, h_manhattan, path_to_policy

# Chercher un chemin
result = astar(grid, start=(0,0), goal=(4,4), heuristic=h_manhattan, w=1.0)
print(result['path'])       # [(0,0), (1,0), ...]
print(result['cost'])       # 8
print(result['nodes_dev'])  # 21
print(result['time_s'])     # 0.000185

# Extraire la politique
policy = path_to_policy(result['path'], grid)
# {(0,0): (1,0), (1,0): (2,0), ...}
markov.py
pythonfrom markov import build_markov, evolve, prob_goal_over_time

# Construire la matrice de transition
states, idx, P = build_markov(grid, policy, goal, epsilon=0.1)
# P : matrice (n×n) stochastique par ligne

# Évolution de la distribution
pi0 = np.zeros(len(states)); pi0[idx[start]] = 1.0
distributions = evolve(pi0, P, n_steps=60)
prob_goal = prob_goal_over_time(distributions, idx)
simulation.py
pythonfrom simulation import simulate_trajectories

sim = simulate_trajectories(states, idx, P, start, N_traj=5000, max_steps=200)
print(sim['prob_goal'])       # 0.9748
print(sim['mean_time_goal'])  # 16.11
experiments.py
python# Fonctions disponibles
exp1_comparison()    # Comparaison UCS/Greedy/A* — chemins + métriques
exp2_epsilon()       # Impact de ε sur P(GOAL)
exp3_heuristics()    # Heuristiques : h=0, Manhattan, WA*(w=2)
exp4_markov_analysis() # Classes, absorption, Monte-Carlo
exp5_matrices_P()    # Heatmap P(i,j) pour 4 valeurs de ε

afficher_matrice_P(epsilon=0.1, grille="facile")  # Affichage terminal

🔬 Analyse Markov — Grille Facile (ε = 0.15)
ClasseTailleTypeDescriptionC₀1 étatTransitoireÉtat isolé (4,4)C₁20 étatsTransitoireTous les états libres accessiblesC₂1 étatPersistant absorbantGOAL — attracteur irréversible
Simulation Monte-Carlo (N = 5000) :
MétriqueValeurP(GOAL) en 200 étapes97.5%E[T | GOAL]16.11 étapesÉcart-type σ[T]23.07 étapesE[T] théorique23.82 étapes

🚧 Limites et Extensions
Limites actuelles :

Politique statique — pas de replanification si l'agent dévie du chemin A*
ε uniforme sur toute la grille
Inversion (I−Q)⁻¹ en O(n³) — coûteux pour grandes grilles

Extensions possibles :

MDP (Processus de Décision Markovien) — optimiser la politique directement sous incertitude
D*Lite / LPA* — replanification en ligne pour environnements dynamiques
WA* adaptatif — ajuster w selon les contraintes temps/qualité


📚 Références

Russell, S. & Norvig, P. — Artificial Intelligence: A Modern Approach, 4th ed. Pearson, 2020.
Hart, P. E., Nilsson, N. J., & Raphael, B. — A Formal Basis for the Heuristic Determination of Minimum Cost Paths. IEEE Trans. SSC, 4(2), 1968.
Puterman, M. L. — Markov Decision Processes. Wiley, 1994.
Kemeny, J. G. & Snell, J. L. — Finite Markov Chains. Springer, 1976.


Mini-Projet — Master SDIA | ENSET Mohammedia — Université Hassan II de Casablanca — 2025–2026
