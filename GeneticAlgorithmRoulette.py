import random
import math
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class GeneticAlgorithmRoulette:
    def __init__(self, population_size=100, generations=500, mutation_rate=0.3,
                 max_time=60, crossover_type='OX', mutation_type='swap',
                 elitism_count=2, tournament_size=5, selection_type='roulette'):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.max_time = max_time
        self.crossover_type = crossover_type  # 'OX', 'PMX', 'CX'
        self.mutation_type = mutation_type  # 'swap', 'inversion', 'scramble'
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.selection_type = selection_type  # 'roulette', 'tournament'

    def solve(self, cities: List[Tuple[float, float]], verbose=True) -> Dict[str, Any]:
        n = len(cities)
        distance_matrix = self._create_distance_matrix(cities)

        population = self._initialize_population(n)
        best_solution = min(population, key=lambda x: self._calculate_cost(x, distance_matrix))
        best_cost = self._calculate_cost(best_solution, distance_matrix)

        history = {'best_costs': [], 'average_costs': [], 'diversity': []}
        stagnation_count = 0
        previous_best = best_cost

        start_time = time.time()

        for generation in range(self.generations):
            # Vérification du temps écoulé
            if time.time() - start_time > self.max_time:
                if verbose:
                    print(f"Arrêt après {generation} générations - Temps maximum atteint")
                break

            # Évaluation
            costs = [self._calculate_cost(ind, distance_matrix) for ind in population]
            current_best = min(costs)
            average_cost = sum(costs) / len(costs)
            diversity = self._calculate_diversity(population)

            # Mise à jour de la meilleure solution
            if current_best < best_cost:
                best_cost = current_best
                best_solution = population[costs.index(current_best)].copy()
                stagnation_count = 0
                if verbose and generation % 50 == 0:
                    print(f"Generation {generation}: Best = {best_cost:.2f}")
            else:
                stagnation_count += 1

            # Diversification si stagnation
            if stagnation_count >= 100:
                if verbose:
                    print(f"Diversification à la génération {generation}")
                population = self._diversify_population(population, best_solution)
                stagnation_count = 0

            # Nouvelle population avec élitisme
            new_population = self._select_elites(population, costs)

            # Remplissage du reste de la population
            while len(new_population) < self.population_size:
                if self.selection_type == 'tournament':
                    parent1, parent2 = self._tournament_selection(population, costs)
                else:  # roulette par défaut
                    parent1, parent2 = self._roulette_selection(population, distance_matrix)

                child = self._crossover(parent1, parent2)
                self._mutate(child)
                new_population.append(child)

            population = new_population

            history['best_costs'].append(best_cost)
            history['average_costs'].append(average_cost)
            history['diversity'].append(diversity)

        execution_time = time.time() - start_time

        return {
            'solution': best_solution,
            'cost': best_cost,
            'time': execution_time,
            'history': history
        }

    def _roulette_selection(self, population: List[List[int]],
                            distance_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        fitness = [1 / (self._calculate_cost(ind, distance_matrix) + 1e-9) for ind in population]
        total = sum(fitness)
        prob = [f / total for f in fitness]
        return random.choices(population, weights=prob, k=2)

    def _tournament_selection(self, population: List[List[int]],
                              costs: List[float]) -> Tuple[List[int], List[int]]:
        def select_one():
            tournament_indices = random.sample(range(len(population)), self.tournament_size)
            tournament_costs = [costs[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_costs)]
            return population[winner_index]

        return select_one(), select_one()

    def _select_elites(self, population: List[List[int]], costs: List[float]) -> List[List[int]]:
        """Sélectionne les meilleurs individus pour l'élitisme"""
        elite_indices = np.argsort(costs)[:self.elitism_count]
        return [population[i].copy() for i in elite_indices]

    def _diversify_population(self, population: List[List[int]],
                              best_solution: List[int]) -> List[List[int]]:
        """Diversifie la population tout en conservant la meilleure solution"""
        n = len(best_solution)
        new_population = [best_solution.copy()]

        # Générer de nouvelles solutions diversifiées
        for _ in range(self.population_size - 1):
            if random.random() < 0.7:
                # Solution aléatoire
                individual = list(range(n))
                random.shuffle(individual)
            else:
                # Mutation forte de la meilleure solution
                individual = best_solution.copy()
                self._strong_mutate(individual)

            new_population.append(individual)

        return new_population

    def _strong_mutate(self, individual: List[int]):
        """Applique une mutation plus forte pour la diversification"""
        n = len(individual)
        # Inversion d'une grande séquence
        i, j = sorted(random.sample(range(n), 2))
        individual[i:j + 1] = reversed(individual[i:j + 1])

        # Échange supplémentaire
        i, j = random.sample(range(n), 2)
        individual[i], individual[j] = individual[j], individual[i]

    def _initialize_population(self, n: int) -> List[List[int]]:
        return [random.sample(range(n), n) for _ in range(self.population_size)]

    def _calculate_cost(self, solution: List[int], distance_matrix: np.ndarray) -> float:
        total = 0
        n = len(solution)
        for i in range(n):
            total += distance_matrix[solution[i]][solution[(i + 1) % n]]
        return total

    def _calculate_diversity(self, population: List[List[int]]) -> float:
        """Calcule la diversité de la population"""
        if len(population) <= 1:
            return 0.0

        n = len(population[0])
        diversity = 0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Distance de Hamming
                diff_positions = sum(1 for k in range(n) if population[i][k] != population[j][k])
                diversity += diff_positions / n
                count += 1

        return diversity / count if count > 0 else 0.0

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        if self.crossover_type == 'PMX':
            return self._pmx_crossover(parent1, parent2)
        elif self.crossover_type == 'CX':
            return self._cycle_crossover(parent1, parent2)
        else:  # OX par défaut
            return self._ox_crossover(parent1, parent2)

    def _ox_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order Crossover (OX)"""
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

    def _pmx_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Partially Mapped Crossover (PMX)"""
        n = len(parent1)
        start, end = sorted(random.sample(range(n), 2))
        child = [None] * n

        # Copier le segment du parent1
        child[start:end] = parent1[start:end]

        # Remplir le reste avec le parent2 en respectant les mappings
        for i in list(range(0, start)) + list(range(end, n)):
            gene = parent2[i]
            while gene in child[start:end]:
                idx = parent1[start:end].index(gene)
                gene = parent2[start + idx]
            child[i] = gene

        return child

    def _cycle_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Cycle Crossover (CX)"""
        n = len(parent1)
        child = [None] * n
        cycle = 0
        index = 0

        while None in child:
            if child[index] is None:
                if cycle % 2 == 0:
                    current = parent1[index]
                    while True:
                        child[index] = parent1[index]
                        index = parent1.index(parent2[index])
                        if parent1[index] == current:
                            break
                else:
                    current = parent2[index]
                    while True:
                        child[index] = parent2[index]
                        index = parent2.index(parent1[index])
                        if parent2[index] == current:
                            break
                cycle += 1
            index = (index + 1) % n

        return child

    def _mutate(self, individual: List[int]):
        if random.random() < self.mutation_rate:
            if self.mutation_type == 'inversion':
                self._inversion_mutation(individual)
            elif self.mutation_type == 'scramble':
                self._scramble_mutation(individual)
            else:  # swap par défaut
                self._swap_mutation(individual)

    def _swap_mutation(self, individual: List[int]):
        """Échange deux gènes"""
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

    def _inversion_mutation(self, individual: List[int]):
        """Inverse une séquence de gènes"""
        n = len(individual)
        i, j = sorted(random.sample(range(n), 2))
        individual[i:j + 1] = reversed(individual[i:j + 1])

    def _scramble_mutation(self, individual: List[int]):
        """Mélange une séquence de gènes"""
        n = len(individual)
        i, j = sorted(random.sample(range(n), 2))
        segment = individual[i:j + 1]
        random.shuffle(segment)
        individual[i:j + 1] = segment

    def _create_distance_matrix(self, cities: List[Tuple[float, float]]) -> np.ndarray:
        n = len(cities)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = math.sqrt((cities[i][0] - cities[j][0]) ** 2 +
                                         (cities[i][1] - cities[j][1]) ** 2)
        return matrix


# Exemple d'utilisation
if __name__ == "__main__":
    # Générer des villes aléatoires pour le test
    random.seed(42)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(20)]

    # Algorithme génétique avec les nouveaux paramètres
    ga = GeneticAlgorithmRoulette(
        population_size=100,
        generations=500,
        mutation_rate=0.2,
        max_time=30,
        crossover_type='PMX',
        mutation_type='inversion',
        elitism_count=2,
        tournament_size=5,
        selection_type='tournament'
    )

    result = ga.solve(cities, verbose=True)

    print(f"\nMeilleure solution trouvée: {result['cost']:.2f}")
    print(f"Temps d'exécution: {result['time']:.2f} secondes")
    print(f"Nombre de générations effectuées: {len(result['history']['best_costs'])}")