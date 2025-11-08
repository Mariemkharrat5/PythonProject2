import random
import matplotlib.pyplot as plt
import numpy as np


class GenetiqueOrdonnancement:
    def __init__(self, n_taches, temps_taches, contraintes_precedence=None, machines=1):
        self.n_taches = n_taches
        self.temps_taches = temps_taches
        self.contraintes = contraintes_precedence if contraintes_precedence else []
        self.machines = machines

    def evaluer_solution(self, sequence):
        """ADAPTATION: Calcule le coût d'une solution"""
        makespan = self.calculer_makespan(sequence)
        penalite = self.calculer_penalite_contraintes(sequence)
        return makespan + penalite

    def calculer_makespan(self, sequence):
        """Calcule le temps total d'exécution"""
        if self.machines == 1:
            return sum(self.temps_taches[tache] for tache in sequence)
        else:
            temps_machines = [0] * self.machines
            for tache in sequence:
                machine_min = min(range(self.machines), key=lambda m: temps_machines[m])
                temps_machines[machine_min] += self.temps_taches[tache]
            return max(temps_machines)

    def calculer_penalite_contraintes(self, sequence):
        """Pénalité pour contraintes de précédence violées"""
        if not self.contraintes:
            return 0

        penalite = 0
        positions = {tache: idx for idx, tache in enumerate(sequence)}

        for avant, apres in self.contraintes:
            if positions[avant] > positions[apres]:
                penalite += 1000
        return penalite

    def verifier_contraintes(self, sequence):
        """Vérifie si les contraintes sont respectées"""
        positions = {tache: idx for idx, tache in enumerate(sequence)}
        for avant, apres in self.contraintes:
            if positions[avant] > positions[apres]:
                return False
        return True


# ========== FONCTIONS GENETIQUES ==========
def initialiser_population(taille_pop, n_taches):
    """POPULATION: Initialise la population aléatoirement"""
    return [random.sample(range(n_taches), n_taches) for _ in range(taille_pop)]


def calculer_fitness(population, probleme):
    """ADAPTATION: Calcule la fitness (inverse du coût)"""
    couts = [probleme.evaluer_solution(ind) for ind in population]
    return [1 / (cout + 1) for cout in couts]


def selection_parents(population, fitness_scores):
    """ASSIMILATION: Sélection par tournoi"""
    parents = []
    for _ in range(2):
        participants = random.sample(range(len(population)), 3)
        meilleur = max(participants, key=lambda i: fitness_scores[i])
        parents.append(population[meilleur])
    return parents


def croisement(parent1, parent2):
    """CROISEMENT: Ordered Crossover"""
    taille = len(parent1)
    point1, point2 = sorted(random.sample(range(taille), 2))

    enfant = [-1] * taille
    enfant[point1:point2] = parent1[point1:point2]

    genes_restants = [gene for gene in parent2 if gene not in enfant]
    index = 0
    for i in range(taille):
        if enfant[i] == -1:
            enfant[i] = genes_restants[index]
            index += 1
    return enfant


def mutation(individu):
    """MUTATION: Échange de deux gènes"""
    i, j = random.sample(range(len(individu)), 2)
    individu[i], individu[j] = individu[j], individu[i]
    return individu


# ========== ALGORITHME PRINCIPAL ==========
def algorithme_genetique(probleme, taille_pop=50, generations=200,
                         taux_croisement=0.8, taux_mutation=0.1):
    """
    Algorithme génétique complet avec visualisation
    """
    # Initialisation
    population = initialiser_population(taille_pop, probleme.n_taches)

    meilleure_solution = None
    meilleur_cout = float('inf')

    # Statistiques pour les figures
    historique_meilleur = []
    historique_moyen = []
    historique_pire = []

    for generation in range(generations):
        # Évaluation
        fitness_scores = calculer_fitness(population, probleme)
        couts = [probleme.evaluer_solution(ind) for ind in population]

        # Mise à jour des statistiques
        cout_min = min(couts)
        cout_moyen = np.mean(couts)
        cout_max = max(couts)

        historique_meilleur.append(cout_min)
        historique_moyen.append(cout_moyen)
        historique_pire.append(cout_max)

        if cout_min < meilleur_cout:
            idx_meilleur = couts.index(cout_min)
            meilleure_solution = population[idx_meilleur][:]
            meilleur_cout = cout_min

        # Nouvelle population
        nouvelle_population = [meilleure_solution[:]]  # Élitisme

        while len(nouvelle_population) < taille_pop:
            parents = selection_parents(population, fitness_scores)

            if random.random() < taux_croisement:
                enfant = croisement(parents[0], parents[1])
            else:
                enfant = random.choice(parents)[:]

            if random.random() < taux_mutation:
                enfant = mutation(enfant)

            nouvelle_population.append(enfant)

        population = nouvelle_population

    return meilleure_solution, meilleur_cout, historique_meilleur, historique_moyen, historique_pire


# ========== FIGURES ET VISUALISATIONS ==========
def afficher_gantt(sequence, temps_taches, machines):
    """FIGURE 1: Diagramme de Gantt de l'ordonnancement"""
    if machines == 1:
        # Cas mono-machine
        fig, ax = plt.subplots(figsize=(12, 4))
        start_time = 0
        for i, tache in enumerate(sequence):
            duree = temps_taches[tache]
            ax.barh(0, duree, left=start_time, height=0.6,
                    label=f'Tâche {tache}', alpha=0.7)
            ax.text(start_time + duree / 2, 0, f'T{tache}',
                    ha='center', va='center', fontweight='bold')
            start_time += duree
        ax.set_yticks([0])
        ax.set_yticklabels(['Machine 1'])
        ax.set_xlabel('Temps')
        ax.set_title('Diagramme de Gantt - Ordonnancement Mono-machine')

    else:
        # Cas multi-machines
        fig, ax = plt.subplots(figsize=(12, 6))
        temps_machines = [0] * machines
        affectations = [[] for _ in range(machines)]

        for tache in sequence:
            machine_min = min(range(machines), key=lambda m: temps_machines[m])
            start_time = temps_machines[machine_min]
            duree = temps_taches[tache]

            ax.barh(machine_min, duree, left=start_time, height=0.6,
                    label=f'Tâche {tache}', alpha=0.7)
            ax.text(start_time + duree / 2, machine_min, f'T{tache}',
                    ha='center', va='center', fontweight='bold')

            temps_machines[machine_min] += duree
            affectations[machine_min].append(tache)

        ax.set_yticks(range(machines))
        ax.set_yticklabels([f'Machine {i + 1}' for i in range(machines)])
        ax.set_xlabel('Temps')
        ax.set_title('Diagramme de Gantt - Ordonnancement Multi-machines')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def afficher_convergence(historique_meilleur, historique_moyen, historique_pire):
    """FIGURE 2: Courbe de convergence de l'algorithme"""
    plt.figure(figsize=(12, 6))

    generations = range(len(historique_meilleur))

    plt.plot(generations, historique_meilleur, 'g-', linewidth=2, label='Meilleur')
    plt.plot(generations, historique_moyen, 'b-', linewidth=2, label='Moyen', alpha=0.7)
    plt.plot(generations, historique_pire, 'r-', linewidth=2, label='Pire', alpha=0.7)

    plt.fill_between(generations, historique_meilleur, historique_pire,
                     alpha=0.1, color='blue', label='Plage des solutions')

    plt.xlabel('Générations')
    plt.ylabel('Coût (Makespan + Pénalités)')
    plt.title('Convergence de l\'Algorithme Génétique')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def afficher_repartition_temps(solution, temps_taches):
    """FIGURE 3: Répartition des temps de tâches"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique 1: Temps par tâche dans l'ordre d'exécution
    temps_ordonnes = [temps_taches[tache] for tache in solution]
    positions = range(len(solution))

    bars = ax1.bar(positions, temps_ordonnes, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Position dans la séquence')
    ax1.set_ylabel('Temps de la tâche')
    ax1.set_title('Temps des tâches dans l\'ordre d\'exécution')

    # Ajouter les valeurs sur les barres
    for bar, temps in zip(bars, temps_ordonnes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{temps}', ha='center', va='bottom')

    # Graphique 2: Diagramme en camembert des temps
    ax2.pie(temps_ordonnes, labels=[f'Tâche {t}' for t in solution],
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Répartition des temps de traitement')

    plt.tight_layout()
    plt.show()


def afficher_analyse_contraintes(solution, contraintes):
    """FIGURE 4: Analyse du respect des contraintes"""
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
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Contraintes respectées', 'Contraintes violées']
    valeurs = [len(respectees), len(violations)]
    couleurs = ['lightgreen', 'lightcoral']

    bars = ax.bar(categories, valeurs, color=couleurs, alpha=0.7)
    ax.set_ylabel('Nombre de contraintes')
    ax.set_title('Analyse du Respect des Contraintes de Précédence')

    # Ajouter les valeurs sur les barres
    for bar, valeur in zip(bars, valeurs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{valeur}', ha='center', va='bottom', fontweight='bold')

    # Afficher les détails des violations
    if violations:
        print(f"\n❌ Contraintes violées ({len(violations)}):")
        for avant, apres in violations:
            print(f"  Tâche {avant} devrait être avant Tâche {apres}")
    else:
        print("\n✅ Toutes les contraintes sont respectées!")

    plt.tight_layout()
    plt.show()


def afficher_comparaison_methodes(resultats_methodes):
    """FIGURE 5: Comparaison des différentes méthodes"""
    noms = [res['nom'] for res in resultats_methodes]
    makespans = [res['makespan'] for res in resultats_methodes]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(noms, makespans, color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.7)

    plt.ylabel('Makespan')
    plt.title('Comparaison des Méthodes d\'Optimisation')
    plt.xticks(rotation=45)

    # Ajouter les valeurs sur les barres
    for bar, makespan in zip(bars, makespans):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{makespan}', ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# ========== EXEMPLE COMPLET AVEC FIGURES ==========
def executer_exemple_complet():
    """Exécute un exemple complet avec toutes les figures"""

    print("🎯 DÉMONSTRATION COMPLÈTE - ALGORITHME GÉNÉTIQUE")
    print("=" * 50)

    # Paramètres du problème
    n_taches = 8
    temps_taches = [5, 3, 7, 2, 4, 6, 3, 5]
    contraintes = [(0, 2), (1, 3), (2, 4), (3, 5)]  # Tâche i avant tâche j
    machines = 2

    # Création du problème
    probleme = GenetiqueOrdonnancement(n_taches, temps_taches, contraintes, machines)

    print(f"Problème d'ordonnancement:")
    print(f"- {n_taches} tâches")
    print(f"- {machines} machine(s)")
    print(f"- {len(contraintes)} contraintes de précédence")
    print(f"- Temps des tâches: {temps_taches}")
    print(f"- Somme des temps: {sum(temps_taches)}")
    print(f"- Borne inférieure théorique: {sum(temps_taches) / machines:.1f}")
    print("=" * 50)

    # Exécution de l'algorithme génétique
    print("\n🧬 EXÉCUTION DE L'ALGORITHME GÉNÉTIQUE...")
    solution, cout, hist_meilleur, hist_moyen, hist_pire = algorithme_genetique(
        probleme,
        taille_pop=40,
        generations=150,
        taux_croisement=0.8,
        taux_mutation=0.15
    )

    # Résultats
    makespan_reel = probleme.calculer_makespan(solution)
    penalite = probleme.calculer_penalite_contraintes(solution)
    contraintes_ok = probleme.verifier_contraintes(solution)

    print("\n📊 RÉSULTATS OBTENUS:")
    print(f"Meilleure séquence: {solution}")
    print(f"Makespan: {makespan_reel}")
    print(f"Pénalité contraintes: {penalite}")
    print(f"Coût total: {cout}")
    print(f"Contraintes respectées: {'✅ OUI' if contraintes_ok else '❌ NON'}")

    # ========== AFFICHAGE DES FIGURES ==========

    print("\n📈 GÉNÉRATION DES FIGURES...")

    # Figure 1: Diagramme de Gantt
    print("1. Génération du diagramme de Gantt...")
    afficher_gantt(solution, temps_taches, machines)

    # Figure 2: Courbe de convergence
    print("2. Génération de la courbe de convergence...")
    afficher_convergence(hist_meilleur, hist_moyen, hist_pire)

    # Figure 3: Répartition des temps
    print("3. Génération de la répartition des temps...")
    afficher_repartition_temps(solution, temps_taches)

    # Figure 4: Analyse des contraintes
    print("4. Génération de l'analyse des contraintes...")
    afficher_analyse_contraintes(solution, contraintes)

    # Figure 5: Comparaison avec d'autres méthodes (exemple)
    print("5. Génération de la comparaison des méthodes...")
    resultats_comparaison = [
        {'nom': 'Aléatoire', 'makespan': 28},
        {'nom': 'Plus court d\'abord', 'makespan': 22},
        {'nom': 'Algo Génétique', 'makespan': makespan_reel}
    ]
    afficher_comparaison_methodes(resultats_comparaison)

    print("\n🎉 ANALYSE TERMINÉE!")
    return solution, makespan_reel


def comparer_configurations():
    """Compare différentes configurations de l'algorithme"""

    n_taches = 6
    temps_taches = [4, 2, 5, 3, 6, 4]
    contraintes = [(0, 2), (1, 3)]
    machines = 2

    probleme = GenetiqueOrdonnancement(n_taches, temps_taches, contraintes, machines)

    configurations = [
        {'nom': 'Petite Pop', 'taille_pop': 20, 'generations': 100, 'taux_mutation': 0.1},
        {'nom': 'Grande Pop', 'taille_pop': 50, 'generations': 100, 'taux_mutation': 0.1},
        {'nom': 'Mutation Élevée', 'taille_pop': 30, 'generations': 100, 'taux_mutation': 0.3},
    ]

    resultats = []
    plt.figure(figsize=(12, 6))

    for config in configurations:
        print(f"Test configuration: {config['nom']}")

        solution, cout, hist_meilleur, _, _ = algorithme_genetique(
            probleme,
            taille_pop=config['taille_pop'],
            generations=config['generations'],
            taux_mutation=config['taux_mutation'],
            taux_croisement=0.8
        )

        makespan = probleme.calculer_makespan(solution)
        resultats.append({
            'nom': config['nom'],
            'makespan': makespan,
            'historique': hist_meilleur
        })

        # Tracer la convergence pour cette configuration
        plt.plot(hist_meilleur, label=config['nom'], linewidth=2)

    plt.xlabel('Générations')
    plt.ylabel('Meilleur Coût')
    plt.title('Comparaison des Configurations d\'Algorithme Génétique')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Afficher le classement
    resultats_tries = sorted(resultats, key=lambda x: x['makespan'])
    print("\n🏆 CLASSEMENT DES CONFIGURATIONS:")
    for i, res in enumerate(resultats_tries, 1):
        print(f"{i}. {res['nom']}: makespan = {res['makespan']}")

    return resultats


if __name__ == "__main__":
    print("=== ALGORITHME GÉNÉTIQUE AVEC FIGURES ===")
    print("Choisissez une option:")
    print("1. Exemple complet avec figures")
    print("2. Comparaison de configurations")

    choix = input("Votre choix (1 ou 2): ").strip()

    if choix == "1":
        solution, makespan = executer_exemple_complet()
    elif choix == "2":
        resultats = comparer_configurations()
    else:
        print("Choix invalide, exécution de l'exemple complet")
        solution, makespan = executer_exemple_complet()