import copy
import random
import numpy as np
import math
from warehouse import build_sample_warehouse

class GeneticAlgorithm:
    def __init__(self, warehouse, population_size=500, generations=100, crossover_rate=0.8, mutation_rate=0.3):
        self.initial_warehouse = warehouse
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []

    def objective_function(self, warehouse):
        """Evaluate the candidate warehouse state."""
        total_distance = 0
        for zone in warehouse.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                item = rack.storage[i, j, k]
                                if item:
                                    total_distance += item.retrieval_urgency * np.linalg.norm(rack.coordinates)
        return total_distance

    def generate_initial_population(self):
        """Generate an initial population of candidate warehouse states."""
        self.population = []
        for _ in range(self.population_size):
            candidate = copy.deepcopy(self.initial_warehouse)
            # Apply a random number of mutations to diversify the candidate.
            num_mutations = random.randint(1, 10)
            for _ in range(num_mutations):
                candidate = self.mutate(candidate)
            self.population.append(candidate)

    def mutate(self, warehouse):
        """Mutate the warehouse state by swapping two items in a randomly selected rack."""
        new_warehouse = copy.deepcopy(warehouse)
        # Choose a random zone, aisle, and rack
        zone = random.choice(new_warehouse.zones)
        aisle = random.choice(zone.aisles)
        rack = random.choice(aisle.racks)
        positions = rack.get_supported_positions()
        if len(positions) < 2:
            return new_warehouse
        pos1, pos2 = random.sample(positions, 2)
        rack.storage[pos1], rack.storage[pos2] = rack.storage[pos2], rack.storage[pos1]
        return new_warehouse

    def crossover(self, parent1, parent2):
        child = copy.deepcopy(parent1)
        for z_idx, zone in enumerate(child.zones):
            for a_idx, aisle in enumerate(zone.aisles):
                for r_idx, rack in enumerate(aisle.racks):
                    rack_p2 = parent2.zones[z_idx].aisles[a_idx].racks[r_idx]
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                if random.random() < 0.5:
                                # Copy this item slot from parent2
                                   child.zones[z_idx].aisles[a_idx].racks[r_idx].storage[i,j,k] = rack_p2.storage[i,j,k]
        return child
    
    
    
    def select_parent(self):
        """Tournament selection (tournament size of 3)."""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        tournament.sort(key=lambda wh: self.objective_function(wh))
        return tournament[0]

    def run(self):
        """Run the genetic algorithm and return the best warehouse state found."""
        self.generate_initial_population()
        best_candidate = None
        best_cost = float('inf')
        for gen in range(self.generations):
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                # Apply crossover
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                # Apply mutation
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                new_population.append(child)
            self.population = new_population
                # Evaluate the new generation
            for candidate in self.population:
                cost = self.objective_function(candidate)
                if cost < best_cost:
                    best_cost = cost
                    best_candidate = candidate
            print(f"Generation {gen+1}, best cost: {best_cost}")
        return best_candidate

# Example usage:
# First, build and populate your warehouse using your existing functions.
warehouse = build_sample_warehouse(
    num_zones=3,
    num_aisles=4,
    num_racks=3,
    rack_dimensions=(6, 5, 6),
    rack_spacing=(2, 2, 0.5),
    show_vis=True  # Set to True if you want visualization during population
)
populate_warehouse(warehouse, 1000)

# Optionally visualize the initial state.
warehouse.visualize()

# Run the genetic algorithm.
ga = GeneticAlgorithm(warehouse, population_size=500, generations=100, crossover_rate=0.8, mutation_rate=0.3)
optimized_warehouse = ga.run()

# Visualize the optimized warehouse state.
optimized_warehouse.visualize()
