import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple


class JobSchedulingSA:
    def __init__(self, max_iterations=1000, initial_temperature=1000,
                 cooling_rate=0.95, max_time=60, neighbor_type='swap'):
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_time = max_time
        self.neighbor_type = neighbor_type

    def solve(self, jobs: List[Dict], machines: int, verbose=True) -> Dict[str, Any]:
        self.jobs = jobs
        self.machines = machines
        self.n_jobs = len(jobs)

        # Solution initiale aléatoire
        current_solution = self._initialize_solution()
        current_cost = self._calculate_cost(current_solution)

        best_solution = current_solution.copy()
        best_cost = current_cost

        temperature = self.initial_temperature

        history = {
            'costs': [],
            'best_costs': [],
            'temperatures': [],
            'makespan': [],
            'tardiness': []
        }

        start_time = time.time()

        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.max_time:
                if verbose:
                    print(f"Arrêt après {iteration} itérations - Temps maximum atteint")
                break

            # Génération d'un voisin
            neighbor = self._generate_neighbor(current_solution)
            neighbor_cost = self._calculate_cost(neighbor)

            delta_cost = neighbor_cost - current_cost

            # Critère d'acceptation
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                current_solution = neighbor
                current_cost = neighbor_cost

                if neighbor_cost < best_cost:
                    best_solution = neighbor.copy()
                    best_cost = neighbor_cost
                    if verbose and iteration % 100 == 0:
                        print(f"Iteration {iteration}: Best = {best_cost:.2f}, Temp = {temperature:.2f}")

            # Refroidissement
            temperature *= self.cooling_rate

            # Historique
            history['costs'].append(current_cost)
            history['best_costs'].append(best_cost)
            history['temperatures'].append(temperature)

            makespan = self._calculate_makespan(current_solution)
            tardiness = self._calculate_total_tardiness(current_solution)
            history['makespan'].append(makespan)
            history['tardiness'].append(tardiness)

        execution_time = time.time() - start_time

        if verbose:
            print(f"\nOptimisation terminée en {execution_time:.2f} secondes")
            print(f"Meilleur coût: {best_cost:.2f}")
            print(f"Makespan final: {self._calculate_makespan(best_solution):.2f}")
            print(f"Retard total: {self._calculate_total_tardiness(best_solution):.2f}")

        return {
            'solution': best_solution,
            'cost': best_cost,
            'time': execution_time,
            'history': history,
            'schedule': self._create_schedule(best_solution)
        }

    def _initialize_solution(self) -> List[int]:
        return random.sample(range(self.n_jobs), self.n_jobs)

    def _generate_neighbor(self, solution: List[int]) -> List[int]:
        neighbor = solution.copy()

        if self.neighbor_type == 'insert':
            i, j = random.sample(range(self.n_jobs), 2)
            job = neighbor.pop(i)
            neighbor.insert(j, job)

        elif self.neighbor_type == 'reverse':
            i, j = sorted(random.sample(range(self.n_jobs), 2))
            neighbor[i:j + 1] = reversed(neighbor[i:j + 1])

        else:
            i, j = random.sample(range(self.n_jobs), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        return neighbor

    def _calculate_cost(self, solution: List[int]) -> float:
        makespan = self._calculate_makespan(solution)
        tardiness = self._calculate_total_tardiness(solution)
        priority_cost = self._calculate_priority_cost(solution)

        return 0.5 * makespan + 0.3 * tardiness + 0.2 * priority_cost

    def _calculate_makespan(self, solution: List[int]) -> float:
        machine_times = [0] * self.machines
        job_assignment = self._assign_jobs_to_machines(solution)

        for machine in range(self.machines):
            for job_idx in job_assignment[machine]:
                processing_time = self.jobs[job_idx]['processing_time']
                machine_times[machine] += processing_time

        return max(machine_times)

    def _calculate_total_tardiness(self, solution: List[int]) -> float:
        completion_times = self._calculate_completion_times(solution)
        total_tardiness = 0

        for job_idx, completion_time in completion_times.items():
            due_date = self.jobs[job_idx]['due_date']
            tardiness = max(0, completion_time - due_date)
            total_tardiness += tardiness

        return total_tardiness

    def _calculate_priority_cost(self, solution: List[int]) -> float:
        completion_times = self._calculate_completion_times(solution)
        priority_cost = 0

        for job_idx, completion_time in completion_times.items():
            priority = self.jobs[job_idx].get('priority', 1)
            priority_cost += completion_time * priority

        return priority_cost

    def _assign_jobs_to_machines(self, solution: List[int]) -> List[List[int]]:
        machine_times = [0] * self.machines
        job_assignment = [[] for _ in range(self.machines)]

        for job_idx in solution:
            min_machine = np.argmin(machine_times)
            job_assignment[min_machine].append(job_idx)
            machine_times[min_machine] += self.jobs[job_idx]['processing_time']

        return job_assignment

    def _calculate_completion_times(self, solution: List[int]) -> Dict[int, float]:
        machine_times = [0] * self.machines
        completion_times = {}
        job_assignment = self._assign_jobs_to_machines(solution)

        for machine in range(self.machines):
            current_time = 0
            for job_idx in job_assignment[machine]:
                processing_time = self.jobs[job_idx]['processing_time']
                current_time += processing_time
                completion_times[job_idx] = current_time

        return completion_times

    def _create_schedule(self, solution: List[int]) -> Dict[str, Any]:
        machine_times = [0] * self.machines
        job_assignment = self._assign_jobs_to_machines(solution)
        schedule = {
            'machines': [[] for _ in range(self.machines)],
            'completion_times': {},
            'job_assignments': job_assignment
        }

        for machine in range(self.machines):
            current_time = 0
            for job_idx in job_assignment[machine]:
                job = self.jobs[job_idx]
                start_time = current_time
                end_time = current_time + job['processing_time']

                schedule['machines'][machine].append({
                    'job_id': job_idx,
                    'job_name': job.get('name', f'Job_{job_idx}'),
                    'start_time': start_time,
                    'end_time': end_time,
                    'processing_time': job['processing_time'],
                    'due_date': job['due_date']
                })

                schedule['completion_times'][job_idx] = end_time
                current_time = end_time

        return schedule

    def plot_results(self, result: Dict[str, Any], figsize=(15, 10)):
        history = result['history']

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Résultats de l\'Ordonnancement par Recuit Simulé', fontsize=16)

        # Courbe de convergence
        axes[0, 0].plot(history['costs'], 'b-', alpha=0.3, label='Coût actuel')
        axes[0, 0].plot(history['best_costs'], 'r-', label='Meilleur coût')
        axes[0, 0].set_xlabel('Itération')
        axes[0, 0].set_ylabel('Coût')
        axes[0, 0].set_title('Convergence de l\'algorithme')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Évolution de la température
        axes[0, 1].semilogy(history['temperatures'], 'g-')
        axes[0, 1].set_xlabel('Itération')
        axes[0, 1].set_ylabel('Température (échelle log)')
        axes[0, 1].set_title('Évolution de la température')
        axes[0, 1].grid(True, alpha=0.3)

        # Makespan
        axes[0, 2].plot(history['makespan'], 'orange')
        axes[0, 2].set_xlabel('Itération')
        axes[0, 2].set_ylabel('Makespan')
        axes[0, 2].set_title('Évolution du Makespan')
        axes[0, 2].grid(True, alpha=0.3)

        # Retard total
        axes[1, 0].plot(history['tardiness'], 'purple')
        axes[1, 0].set_xlabel('Itération')
        axes[1, 0].set_ylabel('Retard total')
        axes[1, 0].set_title('Évolution du retard total')
        axes[1, 0].grid(True, alpha=0.3)

        # Diagramme de Gantt
        self._plot_gantt(result, axes[1, 1])

        # Métriques des machines
        self._plot_machine_metrics(result, axes[1, 2])

        plt.tight_layout()
        plt.show()

    def _plot_gantt(self, result: Dict[str, Any], ax):
        schedule = result['schedule']
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_jobs))

        for machine_idx, machine_jobs in enumerate(schedule['machines']):
            for job_info in machine_jobs:
                start = job_info['start_time']
                duration = job_info['processing_time']
                job_id = job_info['job_id']

                ax.barh(machine_idx, duration, left=start,
                        color=colors[job_id % len(colors)],
                        edgecolor='black', alpha=0.7)

                ax.text(start + duration / 2, machine_idx,
                        f"J{job_id}", ha='center', va='center', fontsize=8)

        ax.set_xlabel('Temps')
        ax.set_ylabel('Machines')
        ax.set_title('Diagramme de Gantt - Planning final')
        ax.set_yticks(range(self.machines))
        ax.set_yticklabels([f'Machine {i}' for i in range(self.machines)])
        ax.grid(True, alpha=0.3)

    def _plot_machine_metrics(self, result: Dict[str, Any], ax):
        schedule = result['schedule']
        machine_utilization = []

        for machine_idx, machine_jobs in enumerate(schedule['machines']):
            if machine_jobs:
                total_time = machine_jobs[-1]['end_time']
                working_time = sum(job['processing_time'] for job in machine_jobs)
                utilization = (working_time / total_time) * 100 if total_time > 0 else 0
            else:
                utilization = 0
            machine_utilization.append(utilization)

        bars = ax.bar(range(self.machines), machine_utilization,
                      color='skyblue', edgecolor='black')
        ax.set_xlabel('Machines')
        ax.set_ylabel('Taux d\'utilisation (%)')
        ax.set_title('Utilisation des machines')
        ax.set_xticks(range(self.machines))
        ax.set_ylim(0, 100)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}%', ha='center', va='bottom')

        ax.grid(True, alpha=0.3)

    def print_schedule(self, result: Dict[str, Any]):
        schedule = result['schedule']

        print("\n" + "=" * 60)
        print("PLANNING FINAL DÉTAILLÉ")
        print("=" * 60)

        for machine_idx, machine_jobs in enumerate(schedule['machines']):
            print(f"\nMachine {machine_idx}:")
            print("-" * 40)

            for job_info in machine_jobs:
                tardiness = max(0, job_info['end_time'] - job_info['due_date'])
                status = "EN RETARD" if tardiness > 0 else "À TEMPS"

                print(f"  Job {job_info['job_name']}: "
                      f"Start={job_info['start_time']:.1f}, "
                      f"End={job_info['end_time']:.1f}, "
                      f"Due={job_info['due_date']:.1f}, "
                      f"Tardiness={tardiness:.1f} - {status}")


# MAIN EXECUTION
if __name__ == "__main__":
    # Génération de données de test
    random.seed(42)
    n_jobs = 15
    n_machines = 3

    jobs = []
    for i in range(n_jobs):
        jobs.append({
            'id': i,
            'name': f'Job_{i}',
            'processing_time': random.uniform(1, 10),
            'due_date': random.uniform(5, 30),
            'priority': random.randint(1, 3)
        })

    # Création et résolution du problème - MAINTENANT LA CLASSE EST DÉFINIE
    sa = JobSchedulingSA(
        max_iterations=1000,
        initial_temperature=1000,
        cooling_rate=0.95,
        max_time=30,
        neighbor_type='swap'
    )

    # Résolution
    result = sa.solve(jobs, n_machines, verbose=True)

    # Affichage des résultats
    sa.print_schedule(result)

    # Génération des graphiques
    sa.plot_results(result)

    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"Coût final: {result['cost']:.2f}")
    print(f"Makespan: {sa._calculate_makespan(result['solution']):.2f}")
    print(f"Retard total: {sa._calculate_total_tardiness(result['solution']):.2f}")
    print(f"Temps d'exécution: {result['time']:.2f} secondes")