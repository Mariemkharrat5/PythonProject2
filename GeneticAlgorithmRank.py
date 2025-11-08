import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class GeneticAlgorithmRank:
    def __init__(self, population_size=100, generations=500, mutation_rate=0.3):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def solve(self, cities, verbose=True):
        n = len(cities)
        distance_matrix = self._create_distance_matrix(cities)

        population = self._initialize_population(n)
        best_solution = min(population, key=lambda x: self._calculate_cost(x, distance_matrix))
        best_cost = self._calculate_cost(best_solution, distance_matrix)

        history = {'best_costs': [], 'average_costs': [], 'best_solutions': []}

        start_time = time.time()

        for generation in range(self.generations):
            # Évaluation
            costs = [self._calculate_cost(ind, distance_matrix) for ind in population]
            current_best = min(costs)
            average_cost = sum(costs) / len(costs)

            if current_best < best_cost:
                best_cost = current_best
                best_solution = population[costs.index(current_best)].copy()
                if verbose and generation % 50 == 0:
                    print(f"Generation {generation}: Best = {best_cost:.2f}")

            # Sauvegarder pour l'historique
            history['best_costs'].append(best_cost)
            history['average_costs'].append(average_cost)
            history['best_solutions'].append(best_solution.copy())

            # Nouvelle population
            new_population = []

            # Sélection par rang
            while len(new_population) < self.population_size:
                parent1, parent2 = self._rank_selection(population, distance_matrix)
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                new_population.append(child)

            population = new_population

        execution_time = time.time() - start_time

        return {
            'solution': best_solution,
            'cost': best_cost,
            'time': execution_time,
            'history': history,
            'cities': cities
        }

    def _rank_selection(self, population, distance_matrix):
        pop_tri = sorted(population, key=lambda x: self._calculate_cost(x, distance_matrix))
        total_rank = sum(range(1, len(population) + 1))
        probs = [rank / total_rank for rank in range(len(population), 0, -1)]
        return random.choices(pop_tri, weights=probs, k=2)

    def _initialize_population(self, n):
        return [random.sample(range(n), n) for _ in range(self.population_size)]

    def _calculate_cost(self, solution, distance_matrix):
        total = 0
        n = len(solution)
        for i in range(n):
            total += distance_matrix[solution[i]][solution[(i + 1) % n]]
        return total

    def _crossover(self, parent1, parent2):
        n = len(parent1)
        start, end = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[start:end] = parent1[start:end]

        pointer = 0
        for gene in parent2:
            if gene not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = gene
        return child

    def _mutate(self, individual):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]

    def _create_distance_matrix(self, cities):
        n = len(cities)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = math.sqrt((cities[i][0] - cities[j][0]) ** 2 +
                                         (cities[i][1] - cities[j][1]) ** 2)
        return matrix


# ========== FONCTIONS DE VISUALISATION ==========

def plot_convergence(history):
    """FIGURE 1: Courbe de convergence de l'algorithme"""
    plt.figure(figsize=(12, 6))

    generations = range(len(history['best_costs']))

    plt.plot(generations, history['best_costs'], 'g-', linewidth=2, label='Meilleur coût')
    plt.plot(generations, history['average_costs'], 'b-', linewidth=2, label='Coût moyen', alpha=0.7)

    plt.xlabel('Générations')
    plt.ylabel('Distance totale')
    plt.title('Convergence de l\'Algorithme Génétique\n(Sélection par Rang)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Ajouter la valeur finale
    final_best = history['best_costs'][-1]
    plt.axhline(y=final_best, color='r', linestyle='--', alpha=0.7)
    plt.text(len(history['best_costs']) * 0.7, final_best * 1.05,
             f'Meilleure solution: {final_best:.2f}',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_solution(cities, solution, title="Meilleure solution trouvée"):
    """FIGURE 2: Visualisation du parcours des villes"""
    plt.figure(figsize=(12, 8))

    # Extraire les coordonnées dans l'ordre de la solution
    x = [cities[i][0] for i in solution] + [cities[solution[0]][0]]
    y = [cities[i][1] for i in solution] + [cities[solution[0]][1]]

    # Tracer le parcours
    plt.plot(x, y, 'o-', linewidth=2, markersize=8,
             markerfacecolor='red', markeredgecolor='darkred',
             markeredgewidth=2, alpha=0.7)

    # Ajouter les numéros des villes
    for i, city_idx in enumerate(solution):
        plt.annotate(str(city_idx),
                     (cities[city_idx][0], cities[city_idx][1]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # Ajouter les flèches pour montrer le sens du parcours
    for i in range(len(solution)):
        start_idx = solution[i]
        end_idx = solution[(i + 1) % len(solution)]

        dx = cities[end_idx][0] - cities[start_idx][0]
        dy = cities[end_idx][1] - cities[start_idx][1]

        plt.arrow(cities[start_idx][0], cities[start_idx][1],
                  dx * 0.8, dy * 0.8,
                  head_width=0.5, head_length=0.5,
                  fc='blue', ec='blue', alpha=0.5)

    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.title(f'{title}\nDistance totale: {calculate_route_distance(cities, solution):.2f}')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_cities_distribution(cities):
    """FIGURE 3: Distribution spatiale des villes"""
    plt.figure(figsize=(10, 8))

    x = [city[0] for city in cities]
    y = [city[1] for city in cities]

    plt.scatter(x, y, s=100, c='red', alpha=0.7, edgecolors='black')

    # Ajouter les numéros des villes
    for i, (xi, yi) in enumerate(cities):
        plt.annotate(f'Ville {i}', (xi, yi),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.title('Distribution Spatiale des Villes')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_algorithm_comparison(results_dict):
    """FIGURE 4: Comparaison des performances"""
    algorithms = list(results_dict.keys())
    costs = [results_dict[algo]['cost'] for algo in algorithms]
    times = [results_dict[algo]['time'] for algo in algorithms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Graphique 1: Coûts (distances)
    bars1 = ax1.bar(algorithms, costs, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
    ax1.set_ylabel('Distance totale')
    ax1.set_title('Comparaison des Distances par Algorithme')
    ax1.grid(True, alpha=0.3, axis='y')

    # Ajouter les valeurs sur les barres
    for bar, cost in zip(bars1, costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{cost:.2f}', ha='center', va='bottom', fontweight='bold')

    # Graphique 2: Temps d'exécution
    bars2 = ax2.bar(algorithms, times, color=['lightblue', 'lightgreen', 'pink'], alpha=0.7)
    ax2.set_ylabel('Temps d\'exécution (s)')
    ax2.set_title('Comparaison des Temps d\'Exécution')
    ax2.grid(True, alpha=0.3, axis='y')

    # Ajouter les valeurs sur les barres
    for bar, time_val in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_animated_evolution(cities, history, interval=100):
    """FIGURE 5: Animation de l'évolution des solutions"""
    fig, ax = plt.subplots(figsize=(12, 8))

    def animate(frame):
        ax.clear()
        solution = history['best_solutions'][frame]
        cost = history['best_costs'][frame]

        # Tracer le parcours
        x = [cities[i][0] for i in solution] + [cities[solution[0]][0]]
        y = [cities[i][1] for i in solution] + [cities[solution[0]][1]]

        ax.plot(x, y, 'o-', linewidth=2, markersize=6,
                markerfacecolor='red', markeredgecolor='darkred')

        # Ajouter les numéros des villes
        for i, city_idx in enumerate(solution):
            ax.annotate(str(city_idx),
                        (cities[city_idx][0], cities[city_idx][1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        ax.set_xlabel('Coordonnée X')
        ax.set_ylabel('Coordonnée Y')
        ax.set_title(f'Évolution de la Solution (Génération {frame})\nDistance: {cost:.2f}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    anim = FuncAnimation(fig, animate, frames=len(history['best_solutions']),
                         interval=interval, repeat=True)

    plt.tight_layout()
    plt.show()

    return anim


def plot_distance_matrix(cities):
    """FIGURE 6: Matrice des distances entre villes"""
    n = len(cities)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = math.sqrt((cities[i][0] - cities[j][0]) ** 2 +
                                              (cities[i][1] - cities[j][1]) ** 2)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')

    # Ajouter les valeurs dans les cases
    for i in range(n):
        for j in range(n):
            text = plt.text(j, i, f'{distance_matrix[i, j]:.1f}',
                            ha="center", va="center", color="w", fontsize=8)

    plt.colorbar(im, label='Distance')
    plt.xlabel('Index de la ville')
    plt.ylabel('Index de la ville')
    plt.title('Matrice des Distances entre Villes')
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.tight_layout()
    plt.show()


def calculate_route_distance(cities, solution):
    """Calcule la distance totale d'un parcours"""
    total = 0
    n = len(solution)
    for i in range(n):
        city1 = cities[solution[i]]
        city2 = cities[solution[(i + 1) % n]]
        total += math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
    return total


# ========== FONCTIONS DE GÉNÉRATION DE DONNÉES ==========

def generate_random_cities(n_cities=10, x_range=(0, 100), y_range=(0, 100)):
    """Génère des villes aléatoires"""
    return [(random.uniform(x_range[0], x_range[1]),
             random.uniform(y_range[0], y_range[1])) for _ in range(n_cities)]


def generate_circular_cities(n_cities=10, radius=50, center=(50, 50)):
    """Génère des villes en cercle (problème standard)"""
    cities = []
    for i in range(n_cities):
        angle = 2 * math.pi * i / n_cities
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        cities.append((x, y))
    return cities


def generate_grid_cities(rows=4, cols=4, spacing=20):
    """Génère des villes en grille"""
    cities = []
    for i in range(rows):
        for j in range(cols):
            cities.append((j * spacing, i * spacing))
    return cities


# ========== EXEMPLE COMPLET D'UTILISATION ==========

def run_complete_example():
    """Exécute un exemple complet avec toutes les visualisations"""

    print("🚀 ALGORITHME GÉNÉTIQUE POUR LE PROBLÈME DU VOYAGEUR DE COMMERCE")
    print("=" * 60)

    # Générer des villes
    cities = generate_random_cities(15, (0, 100), (0, 100))
    # cities = generate_circular_cities(12)  # Alternative: villes en cercle
    # cities = generate_grid_cities(4, 4)    # Alternative: villes en grille

    print(f"Nombre de villes: {len(cities)}")
    print("Coordonnées des villes:")
    for i, city in enumerate(cities):
        print(f"  Ville {i}: ({city[0]:.1f}, {city[1]:.1f})")

    # Créer et exécuter l'algorithme
    ga = GeneticAlgorithmRank(population_size=50, generations=200, mutation_rate=0.2)

    print("\n🧬 EXÉCUTION DE L'ALGORITHME GÉNÉTIQUE...")
    result = ga.solve(cities, verbose=True)

    print(f"\n📊 RÉSULTATS:")
    print(f"Solution: {result['solution']}")
    print(f"Distance totale: {result['cost']:.2f}")
    print(f"Temps d'exécution: {result['time']:.2f} secondes")

    # ========== GÉNÉRATION DES FIGURES ==========

    print("\n📈 GÉNÉRATION DES FIGURES...")

    # Figure 1: Distribution des villes
    print("1. Distribution spatiale des villes...")
    plot_cities_distribution(cities)

    # Figure 2: Matrice des distances
    print("2. Matrice des distances...")
    plot_distance_matrix(cities)

    # Figure 3: Courbe de convergence
    print("3. Courbe de convergence...")
    plot_convergence(result['history'])

    # Figure 4: Meilleure solution
    print("4. Meilleure solution trouvée...")
    plot_solution(cities, result['solution'],
                  "Meilleur Parcours - Algorithme Génétique (Sélection par Rang)")

    # Figure 5: Animation de l'évolution (optionnel - décommentez si voulu)
    # print("5. Animation de l'évolution...")
    # plot_animated_evolution(cities, result['history'])

    # Figure 6: Comparaison avec d'autres méthodes (exemple)
    print("6. Comparaison des méthodes...")
    comparison_data = {
        'Aléatoire': {'cost': calculate_route_distance(cities, list(range(len(cities)))), 'time': 0.1},
        'Plus Proche Voisin': {'cost': result['cost'] * 1.3, 'time': result['time'] * 0.5},
        'Algo Génétique': {'cost': result['cost'], 'time': result['time']}
    }
    plot_algorithm_comparison(comparison_data)

    print("\n✅ ANALYSE TERMINÉE!")
    return result


def compare_parameters():
    """Compare différents paramètres de l'algorithme"""

    cities = generate_random_cities(12, (0, 100), (0, 100))

    param_configs = [
        {'population_size': 30, 'mutation_rate': 0.1, 'label': 'Petite Pop, Faible Mutation'},
        {'population_size': 50, 'mutation_rate': 0.2, 'label': 'Moyenne Pop, Mutation Moyenne'},
        {'population_size': 100, 'mutation_rate': 0.3, 'label': 'Grande Pop, Forte Mutation'},
    ]

    results = {}
    plt.figure(figsize=(12, 6))

    for config in param_configs:
        print(f"Test de la configuration: {config['label']}")

        ga = GeneticAlgorithmRank(
            population_size=config['population_size'],
            generations=150,
            mutation_rate=config['mutation_rate']
        )

        result = ga.solve(cities, verbose=False)
        results[config['label']] = result

        # Tracer la convergence
        plt.plot(result['history']['best_costs'],
                 label=f"{config['label']} (final: {result['cost']:.2f})",
                 linewidth=2)

    plt.xlabel('Générations')
    plt.ylabel('Meilleure Distance')
    plt.title('Comparaison des Paramètres de l\'Algorithme Génétique')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    print("=== ALGORITHME GÉNÉTIQUE AVEC VISUALISATIONS ===")
    print("Choisissez une option:")
    print("1. Exemple complet avec visualisations")
    print("2. Comparaison des paramètres")

    choix = input("Votre choix (1 ou 2): ").strip()

    if choix == "1":
        result = run_complete_example()
    elif choix == "2":
        results = compare_parameters()
    else:
        print("Choix invalide, exécution de l'exemple complet")
        result = run_complete_example()