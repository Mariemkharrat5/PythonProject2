import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import matplotlib.patches as patches


class TabouOrdonnancement:
    def __init__(self, n_taches, temps_taches, contraintes_precedence=None, machines=1):
        """
        n_taches: nombre total de tâches
        temps_taches: liste des durées de chaque tâche
        contraintes_precedence: liste de tuples (tache_avant, tache_apres)
        machines: nombre de machines disponibles
        """
        self.n_taches = n_taches
        self.temps_taches = temps_taches
        self.contraintes = contraintes_precedence if contraintes_precedence else []
        self.machines = machines

    def evaluer_solution(self, sequence):
        """Évalue la qualité d'une solution (makespan + pénalités pour contraintes)"""
        makespan = self.calculer_makespan(sequence)
        penalite = self.calculer_penalite_contraintes(sequence)
        return makespan + penalite

    def calculer_makespan(self, sequence):
        """Calcule le temps total d'exécution (makespan)"""
        if self.machines == 1:
            return sum(self.temps_taches[tache] for tache in sequence)
        else:
            return self.makespan_multimachine(sequence)

    def makespan_multimachine(self, sequence):
        """Calcule le makespan pour plusieurs machines"""
        temps_machines = [0] * self.machines

        for tache in sequence:
            # Assigner à la machine qui finit le plus tôt
            machine_min = min(range(self.machines), key=lambda m: temps_machines[m])
            temps_machines[machine_min] += self.temps_taches[tache]

        return max(temps_machines)

    def calculer_penalite_contraintes(self, sequence):
        """Calcule une pénalité pour les contraintes de précédence violées"""
        if not self.contraintes:
            return 0

        penalite = 0
        positions = {tache: idx for idx, tache in enumerate(sequence)}

        for avant, apres in self.contraintes:
            if positions[avant] > positions[apres]:
                # Pénalité proportionnelle à la violation
                penalite += 1000 * (positions[avant] - positions[apres])

        return penalite

    def verifier_contraintes(self, sequence):
        """Vérifie si les contraintes de précédence sont respectées"""
        positions = {tache: idx for idx, tache in enumerate(sequence)}

        for avant, apres in self.contraintes:
            if positions[avant] > positions[apres]:
                return False
        return True

    def generer_solution_initial(self):
        """Génère une solution initiale respectant les contraintes"""
        if not self.contraintes:
            return random.sample(range(self.n_taches), self.n_taches)

        # Algorithme basé sur le tri topologique pour respecter les contraintes
        return self.tri_topologique()

    def tri_topologique(self):
        """Tri topologique pour respecter les contraintes de précédence"""
        # Construction du graphe
        graphe = {i: [] for i in range(self.n_taches)}
        degre_entrant = [0] * self.n_taches

        for avant, apres in self.contraintes:
            graphe[avant].append(apres)
            degre_entrant[apres] += 1

        # File des nœuds sans contrainte entrante
        file = deque([i for i in range(self.n_taches) if degre_entrant[i] == 0])
        resultat = []

        while file:
            # Choisir aléatoirement parmi les tâches disponibles
            noeud = random.choice(list(file))
            file.remove(noeud)
            resultat.append(noeud)

            for voisin in graphe[noeud]:
                degre_entrant[voisin] -= 1
                if degre_entrant[voisin] == 0:
                    file.append(voisin)

        return resultat if len(resultat) == self.n_taches else list(range(self.n_taches))


def generer_voisins(solution, n_voisins=20):
    """Génère des voisins par échange de tâches"""
    voisins = []
    n = len(solution)

    for _ in range(n_voisins):
        i, j = random.sample(range(n), 2)
        voisin = solution.copy()
        voisin[i], voisin[j] = voisin[j], voisin[i]
        voisins.append(voisin)

    return voisins


def recherche_tabou_ordonnancement(probleme, iterations_max=500, taille_tabu=30,
                                   verbose=True, diversite=True):
    """
    Algorithme de recherche Tabou pour l'ordonnancement de tâches
    """
    if verbose:
        print("🔍 RECHERCHE TABOU POUR ORDONNANCEMENT DE TÂCHES")
        print(f"Itérations maximum: {iterations_max}")
        print(f"Taille liste Tabou: {taille_tabu}")
        print(f"Nombre de machines: {probleme.machines}")
        print("-" * 50)

    # Solution initiale
    solution_actuelle = probleme.generer_solution_initial()
    cout_actuel = probleme.evaluer_solution(solution_actuelle)

    meilleure_solution = solution_actuelle.copy()
    meilleur_cout = cout_actuel

    # Liste Tabou
    liste_tabu = deque(maxlen=taille_tabu)
    liste_tabu.append(tuple(solution_actuelle))

    # Statistiques pour les figures
    historique_couts = [cout_actuel]
    historique_meilleurs = [meilleur_cout]
    iterations_sans_amelioration = 0

    if verbose:
        print(f"Solution initiale: {solution_actuelle}")
        print(f"Coût initial: {cout_actuel:.2f}")

    for iteration in range(iterations_max):
        # Génération des voisins
        voisins = generer_voisins(solution_actuelle)

        # Filtrer les solutions tabou
        voisins_non_tabu = [v for v in voisins if tuple(v) not in liste_tabu]

        if not voisins_non_tabu:
            if diversite:
                # Diversification: réinitialisation partielle
                if verbose:
                    print(f"Itération {iteration}: Diversification activée")
                solution_actuelle = probleme.generer_solution_initial()
                cout_actuel = probleme.evaluer_solution(solution_actuelle)
                liste_tabu.clear()
                liste_tabu.append(tuple(solution_actuelle))
                continue
            else:
                break

        # Évaluation des voisins
        couts_voisins = [(v, probleme.evaluer_solution(v)) for v in voisins_non_tabu]

        # Sélection du meilleur voisin non tabou
        solution_voisine, cout_voisin = min(couts_voisins, key=lambda x: x[1])

        # Mise à jour de la solution courante
        solution_actuelle = solution_voisine
        cout_actuel = cout_voisin

        # Mise à jour liste tabou
        liste_tabu.append(tuple(solution_actuelle))

        # Mise à jour de la meilleure solution
        if cout_voisin < meilleur_cout:
            meilleure_solution = solution_voisine.copy()
            meilleur_cout = cout_voisin
            iterations_sans_amelioration = 0
            if verbose and iteration % 50 == 0:
                print(f"Itération {iteration}: Nouveau meilleur coût = {meilleur_cout:.2f}")
        else:
            iterations_sans_amelioration += 1

        # Sauvegarde des statistiques
        historique_couts.append(cout_actuel)
        historique_meilleurs.append(meilleur_cout)

        # Critère d'arrêt prématuré (stagnation)
        if iterations_sans_amelioration > 100:
            if verbose:
                print(f"Arrêt prématuré à l'itération {iteration} (stagnation)")
            break

    # Résultats finaux
    makespan_reel = probleme.calculer_makespan(meilleure_solution)
    penalite = probleme.calculer_penalite_contraintes(meilleure_solution)
    contraintes_ok = probleme.verifier_contraintes(meilleure_solution)

    if verbose:
        print("-" * 50)
        print("🎯 RÉSULTATS FINAUX")
        print(f"Meilleure séquence: {meilleure_solution}")
        print(f"Makespan: {makespan_reel}")
        print(f"Pénalité contraintes: {penalite}")
        print(f"Coût total: {meilleur_cout:.2f}")
        print(f"Contraintes respectées: {'✅ OUI' if contraintes_ok else '❌ NON'}")
        print(f"Itérations effectuées: {min(iteration + 1, iterations_max)}")

    return (meilleure_solution, meilleur_cout, historique_couts,
            historique_meilleurs, makespan_reel)


# ========== FONCTIONS DE VISUALISATION ==========

def afficher_diagramme_gantt(sequence, temps_taches, machines, title="Diagramme de Gantt"):
    """FIGURE 1: Diagramme de Gantt de l'ordonnancement"""
    fig, ax = plt.subplots(figsize=(12, 6))

    if machines == 1:
        # Cas mono-machine
        start_time = 0
        for i, tache in enumerate(sequence):
            duree = temps_taches[tache]
            ax.barh(0, duree, left=start_time, height=0.6,
                    color=plt.cm.Set3(tache / len(sequence)), alpha=0.7,
                    edgecolor='black', linewidth=1)

            # Texte au centre de la barre
            ax.text(start_time + duree / 2, 0, f'T{tache}\n({duree})',
                    ha='center', va='center', fontweight='bold', fontsize=10)

            start_time += duree

        ax.set_yticks([0])
        ax.set_yticklabels(['Machine 1'])
        ax.set_xlabel('Temps')
        ax.set_title(f'{title} - Mono-machine\nMakespan: {start_time}')

    else:
        # Cas multi-machines
        temps_machines = [0] * machines
        affectations = [[] for _ in range(machines)]

        colors = plt.cm.Set3(np.linspace(0, 1, len(sequence)))

        for tache in sequence:
            machine_min = min(range(machines), key=lambda m: temps_machines[m])
            start_time = temps_machines[machine_min]
            duree = temps_taches[tache]

            ax.barh(machine_min, duree, left=start_time, height=0.6,
                    color=colors[tache], alpha=0.7,
                    edgecolor='black', linewidth=1)

            ax.text(start_time + duree / 2, machine_min, f'T{tache}\n({duree})',
                    ha='center', va='center', fontweight='bold', fontsize=9)

            temps_machines[machine_min] += duree
            affectations[machine_min].append(tache)

        makespan = max(temps_machines)
        ax.set_yticks(range(machines))
        ax.set_yticklabels([f'Machine {i + 1}' for i in range(machines)])
        ax.set_xlabel('Temps')
        ax.set_title(f'{title} - {machines} machines\nMakespan: {makespan}')

    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def afficher_convergence_tabou(historique_couts, historique_meilleurs):
    """FIGURE 2: Courbe de convergence de l'algorithme Tabou"""
    plt.figure(figsize=(12, 6))

    iterations = range(len(historique_couts))

    plt.plot(iterations, historique_couts, 'b-', alpha=0.5, linewidth=1, label='Coût actuel')
    plt.plot(iterations, historique_meilleurs, 'r-', linewidth=2, label='Meilleur coût')

    # Marquer le point final
    plt.axhline(y=historique_meilleurs[-1], color='g', linestyle='--', alpha=0.7)
    plt.text(len(historique_couts) * 0.7, historique_meilleurs[-1] * 1.05,
             f'Solution optimale: {historique_meilleurs[-1]:.2f}',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

    plt.xlabel('Itérations')
    plt.ylabel('Coût (Makespan + Pénalités)')
    plt.title('Convergence de la Recherche Tabou')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def afficher_analyse_contraintes(solution, contraintes, temps_taches):
    """FIGURE 3: Analyse du respect des contraintes"""
    if not contraintes:
        print("Aucune contrainte à analyser")
        return

    positions = {tache: idx for idx, tache in enumerate(solution)}
    violations = []
    respectees = []

    for avant, apres in contraintes:
        if positions[avant] > positions[apres]:
            violations.append((avant, apres))
        else:
            respectees.append((avant, apres))

    # Création du graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Graphique 1: Statut des contraintes
    categories = ['Contraintes\nrespectées', 'Contraintes\nviolées']
    valeurs = [len(respectees), len(violations)]
    couleurs = ['lightgreen', 'lightcoral']

    bars1 = ax1.bar(categories, valeurs, color=couleurs, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Nombre de contraintes')
    ax1.set_title('Analyse du Respect des Contraintes de Précédence')

    # Ajouter les valeurs sur les barres
    for bar, valeur in zip(bars1, valeurs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{valeur}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Graphique 2: Détails des violations
    if violations:
        ax2.axis('off')
        ax2.text(0.1, 0.9, 'DÉTAILS DES VIOLATIONS:', fontweight='bold', fontsize=12,
                 transform=ax2.transAxes)

        for i, (avant, apres) in enumerate(violations[:8]):  # Limiter à 8 affichages
            ax2.text(0.1, 0.8 - i * 0.1, f'• Tâche {avant} → Tâche {apres}',
                     transform=ax2.transAxes, fontsize=10, color='red')

        if len(violations) > 8:
            ax2.text(0.1, 0.1, f'... et {len(violations) - 8} autres violations',
                     transform=ax2.transAxes, fontsize=10, color='red', style='italic')
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, '✅ TOUTES LES CONTRAINTES\nSONT RESPECTÉES!',
                 fontweight='bold', fontsize=14, color='green',
                 ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.show()

    # Afficher les détails dans la console
    if violations:
        print(f"\n❌ Contraintes violées ({len(violations)}):")
        for avant, apres in violations:
            print(
                f"  Tâche {avant} (position {positions[avant]}) devrait être avant Tâche {apres} (position {positions[apres]})")
    else:
        print("\n✅ Toutes les contraintes sont respectées!")


def afficher_comparaison_solutions(probleme, solutions_dict):
    """FIGURE 4: Comparaison de différentes solutions"""
    noms = list(solutions_dict.keys())
    makespans = [probleme.calculer_makespan(sol) for sol in solutions_dict.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(noms, makespans, color=['lightblue', 'lightgreen', 'lightcoral', 'gold'], alpha=0.7)

    plt.ylabel('Makespan')
    plt.title('Comparaison des Solutions d\'Ordonnancement')
    plt.xticks(rotation=45)

    # Ajouter les valeurs sur les barres
    for bar, makespan in zip(bars, makespans):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{makespan}', ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def afficher_analyse_temps_taches(solution, temps_taches):
    """FIGURE 5: Analyse des temps de traitement"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique 1: Temps par tâche dans l'ordre d'exécution
    temps_ordonnes = [temps_taches[tache] for tache in solution]
    positions = range(len(solution))

    bars = ax1.bar(positions, temps_ordonnes, color=plt.cm.viridis(np.linspace(0, 1, len(solution))))
    ax1.set_xlabel('Position dans la séquence')
    ax1.set_ylabel('Temps de la tâche')
    ax1.set_title('Temps des tâches dans l\'ordre d\'exécution')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'T{t}' for t in solution])

    # Ajouter les valeurs sur les barres
    for bar, temps in zip(bars, temps_ordonnes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{temps}', ha='center', va='bottom', fontsize=9)

    # Graphique 2: Diagramme en camembert des temps
    ax2.pie(temps_ordonnes, labels=[f'Tâche {t}' for t in solution],
            autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(solution))))
    ax2.set_title('Répartition des temps de traitement')

    plt.tight_layout()
    plt.show()


def afficher_graphe_contraintes(contraintes, n_taches):
    """FIGURE 6: Visualisation du graphe de contraintes"""
    if not contraintes:
        print("Aucune contrainte à visualiser")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Positionner les nœuds en cercle
    angles = np.linspace(0, 2 * np.pi, n_taches, endpoint=False)
    radius = 5
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Dessiner les nœuds (tâches)
    for i in range(n_taches):
        circle = plt.Circle((x[i], y[i]), 0.3, color='lightblue', alpha=0.7)
        ax.add_patch(circle)
        ax.text(x[i], y[i], f'T{i}', ha='center', va='center', fontweight='bold')

    # Dessiner les arêtes (contraintes)
    for avant, apres in contraintes:
        # Calculer la position de la flèche
        start_x, start_y = x[avant], y[avant]
        end_x, end_y = x[apres], y[apres]

        # Dessiner la flèche
        ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                 head_width=0.2, head_length=0.2,
                 fc='red', ec='red', alpha=0.7, length_includes_head=True)

    ax.set_xlim(-radius - 1, radius + 1)
    ax.set_ylim(-radius - 1, radius + 1)
    ax.set_aspect('equal')
    ax.set_title('Graphe des Contraintes de Précédence')
    ax.axis('off')

    # Légende
    ax.text(-radius - 0.5, -radius - 0.5, '→ : Tâche i doit précéder Tâche j',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()


# ========== EXEMPLE COMPLET D'UTILISATION ==========

def executer_exemple_complet():
    """Exécute un exemple complet avec toutes les visualisations"""

    print("🎯 RECHERCHE TABOU - ORDONNANCEMENT DE TÂCHES")
    print("=" * 60)

    # Paramètres du problème
    n_taches = 8
    temps_taches = [5, 3, 7, 2, 4, 6, 3, 5]
    contraintes = [(0, 2), (1, 3), (2, 4), (3, 5)]  # Tâche i avant tâche j
    machines = 2

    # Création du problème
    probleme = TabouOrdonnancement(n_taches, temps_taches, contraintes, machines)

    print(f"Problème d'ordonnancement:")
    print(f"- {n_taches} tâches")
    print(f"- {machines} machine(s)")
    print(f"- {len(contraintes)} contraintes de précédence")
    print(f"- Temps des tâches: {temps_taches}")
    print(f"- Somme des temps: {sum(temps_taches)}")
    print(f"- Borne inférieure théorique: {sum(temps_taches) / machines:.1f}")
    print("=" * 50)

    # Exécution de l'algorithme Tabou
    print("\n🔍 EXÉCUTION DE LA RECHERCHE TABOU...")
    solution, cout, historique_couts, historique_meilleurs, makespan = recherche_tabou_ordonnancement(
        probleme,
        iterations_max=300,
        taille_tabu=25,
        verbose=True
    )

    # ========== GÉNÉRATION DES FIGURES ==========

    print("\n📈 GÉNÉRATION DES FIGURES...")

    # Figure 1: Graphe des contraintes
    print("1. Graphe des contraintes...")
    afficher_graphe_contraintes(contraintes, n_taches)

    # Figure 2: Diagramme de Gantt
    print("2. Diagramme de Gantt...")
    afficher_diagramme_gantt(solution, temps_taches, machines, "Ordonnancement Optimal - Recherche Tabou")

    # Figure 3: Courbe de convergence
    print("3. Courbe de convergence...")
    afficher_convergence_tabou(historique_couts, historique_meilleurs)

    # Figure 4: Analyse des contraintes
    print("4. Analyse des contraintes...")
    afficher_analyse_contraintes(solution, contraintes, temps_taches)

    # Figure 5: Analyse des temps
    print("5. Analyse des temps de traitement...")
    afficher_analyse_temps_taches(solution, temps_taches)

    # Figure 6: Comparaison avec d'autres méthodes
    print("6. Comparaison des méthodes...")

    # Générer quelques solutions alternatives pour comparaison
    solution_aleatoire = random.sample(range(n_taches), n_taches)
    solution_plus_court = sorted(range(n_taches), key=lambda x: temps_taches[x])

    solutions_comparaison = {
        'Aléatoire': solution_aleatoire,
        'Plus court d\'abord': solution_plus_court,
        'Recherche Tabou': solution
    }

    afficher_comparaison_solutions(probleme, solutions_comparaison)

    print("\n✅ ANALYSE TERMINÉE!")
    return solution, makespan


def comparer_parametres_tabou():
    """Compare différents paramètres de l'algorithme Tabou"""

    n_taches = 6
    temps_taches = [4, 2, 5, 3, 6, 4]
    contraintes = [(0, 2), (1, 3), (4, 5)]
    machines = 2

    probleme = TabouOrdonnancement(n_taches, temps_taches, contraintes, machines)

    parametres = [
        {'iterations_max': 100, 'taille_tabu': 10, 'label': 'Petite recherche'},
        {'iterations_max': 200, 'taille_tabu': 20, 'label': 'Recherche moyenne'},
        {'iterations_max': 300, 'taille_tabu': 30, 'label': 'Recherche longue'},
    ]

    resultats = {}
    plt.figure(figsize=(12, 6))

    for params in parametres:
        print(f"Test des paramètres: {params['label']}")

        solution, cout, historique_couts, historique_meilleurs, makespan = recherche_tabou_ordonnancement(
            probleme,
            iterations_max=params['iterations_max'],
            taille_tabu=params['taille_tabu'],
            verbose=False
        )

        resultats[params['label']] = {
            'solution': solution,
            'makespan': makespan,
            'historique': historique_meilleurs
        }

        # Tracer la convergence
        plt.plot(historique_meilleurs,
                 label=f"{params['label']} (final: {makespan})",
                 linewidth=2)

    plt.xlabel('Itérations')
    plt.ylabel('Meilleur Makespan')
    plt.title('Comparaison des Paramètres de la Recherche Tabou')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Afficher le classement
    classement = sorted(resultats.items(), key=lambda x: x[1]['makespan'])
    print("\n🏆 CLASSEMENT DES PARAMÈTRES:")
    for i, (nom, res) in enumerate(classement, 1):
        print(f"{i}. {nom}: makespan = {res['makespan']}")

    return resultats


if __name__ == "__main__":
    print("=== RECHERCHE TABOU POUR ORDONNANCEMENT DE TÂCHES ===")
    print("Choisissez une option:")
    print("1. Exemple complet avec visualisations")
    print("2. Comparaison des paramètres")

    choix = input("Votre choix (1 ou 2): ").strip()

    if choix == "1":
        solution, makespan = executer_exemple_complet()
    elif choix == "2":
        resultats = comparer_parametres_tabou()
    else:
        print("Choix invalide, exécution de l'exemple complet")
        solution, makespan = executer_exemple_complet()