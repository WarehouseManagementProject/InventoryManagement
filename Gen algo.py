import copy
import random
import numpy as np
import math
from warehouse import build_sample_warehouse, populate_warehouse

"""
Genetic Algorithm that performs cross-rack swaps:
- 'item-to-item' swap: picks two racks that each have at least one item,
  then randomly swaps one occupied slot from rack1 with one occupied slot from rack2.
- 'item-to-empty' swap: picks one rack that has at least one item,
  plus one rack that has at least one supported empty slot,
  then moves an item from the first to a supported empty position in the second.
"""

class GeneticAlgorithm:
    def __init__(
        self,
        warehouse,
        population_size=300,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.3,
        fallback_prob=0.05,
        elitism_rate=0.05
    ):
        self.initial_warehouse = warehouse
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.fallback_prob = fallback_prob
        self.elitism_rate = elitism_rate
        self.population = []

    def objective_function(self, warehouse):
        """
        Revised objective function that:
        - Removes zone-level penalty for the dock zone.
        - Applies lower aisle and rack level diversity penalties in the dock zone vs. higher penalties elsewhere.
        """
        total_cost = 0.0
        dock = np.array([0, 0, 0])
        dock_zone = warehouse.zones[0]

        # Penalty weights :
        w_zone          = 1.0   # zone-level penalty (for zones other than dock_zone)
        w_aisle_dock    = 5.0   # aisle-level penalty inside dock zone
        w_aisle_other   = 10.0  # aisle-level penalty for other zones
        w_rack_dock     = 2.0   # rack-level penalty inside dock zone
        w_rack_other    = 3.0   # rack-level penalty for other zones

        # 1. RETRIEVAL COST
        for zone in warehouse.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    rack_base = np.array(rack.coordinates, dtype=float)
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                item = rack.storage[i, j, k]
                                if item is not None:
                                    item_pos = rack_base + np.array([i, j, k], dtype=float)
                                    distance = np.sum(np.abs(item_pos - dock))
                                    total_cost += item.retrieval_urgency * distance
                                    if item.retrieval_urgency >= 4 and zone != dock_zone:
                                        total_cost += 20.0

        # 2. ZONE-LEVEL PENALTY (skip for the dock zone)
        for z_idx, zone in enumerate(warehouse.zones):
            if zone == dock_zone:
                continue  # No zone-level penalty in the dock zone
            category_counts = {}
            total_items = 0
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                item = rack.storage[i, j, k]
                                if item is not None:
                                    total_items += 1
                                    cat = item.category
                                    category_counts[cat] = category_counts.get(cat, 0) + 1
            if total_items > 0:
                max_count = max(category_counts.values())
                diversity = total_items - max_count
                total_cost += w_zone * diversity

        # 3. AISLE-LEVEL PENALTY
        for zone in warehouse.zones:
            # Decide aisle penalty weight based on whether we're in the dock zone or not
            aisle_weight = w_aisle_dock if zone == dock_zone else w_aisle_other

            for aisle in zone.aisles:
                category_counts = {}
                total_items = 0
                for rack in aisle.racks:
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                item = rack.storage[i, j, k]
                                if item is not None:
                                    total_items += 1
                                    cat = item.category
                                    category_counts[cat] = category_counts.get(cat, 0) + 1
                if total_items > 0:
                    max_count = max(category_counts.values())
                    diversity = total_items - max_count
                    total_cost += aisle_weight * diversity

        # 4. RACK-LEVEL PENALTY
        for zone in warehouse.zones:
            # Decide rack penalty weight based on whether we're in the dock zone or not
            rack_weight = w_rack_dock if zone == dock_zone else w_rack_other

            for aisle in zone.aisles:
                for rack in aisle.racks:
                    subcat_counts = {}
                    total_items = 0
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                item = rack.storage[i, j, k]
                                if item is not None:
                                    total_items += 1
                                    subcat = item.sub_category
                                    subcat_counts[subcat] = subcat_counts.get(subcat, 0) + 1
                    if total_items > 0:
                        max_count = max(subcat_counts.values())
                        diversity = total_items - max_count
                        total_cost += rack_weight * diversity

        return total_cost



    def generate_initial_population(self):
        """Generate an initial population of candidate warehouse states by random mutations."""
        self.population = []
        for _ in range(self.population_size):
            candidate = copy.deepcopy(self.initial_warehouse)
            num_mutations = random.randint(1, 10)
            for _ in range(num_mutations):
                candidate = self.mutate(candidate)
            self.population.append(candidate)

    def mutate(self, warehouse):
        """
        Mutate the warehouse state by doing a cross-rack swap:
        1) item-to-item:
           - pick 2 distinct racks that each have at least 1 item
           - swap a random occupied cell in rack1 with a random occupied cell in rack2
        2) item-to-empty:
           - pick 1 rack that has at least 1 occupied cell
           - pick 1 rack that has at least 1 supported empty cell
        """
        original_cost = self.objective_function(warehouse)
        new_warehouse = copy.deepcopy(warehouse)

        # Step 1: Build lists of racks that have items, and racks that have
        # supported empty slots.

        racks_with_items = []
        racks_with_empty = []
        for zone in new_warehouse.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    # Checking if rack has at least one item
                    has_item = any(item is not None for item in rack.storage.flatten())
                    # Checking if rack has at least one supported empty slot
                    supported_empty_positions = [
                        p for p in rack.get_supported_positions()
                        if rack.storage[p] is None
                    ]
                    has_supported_empty = (len(supported_empty_positions) > 0)

                    if has_item:
                        racks_with_items.append(rack)
                    if has_supported_empty:
                        racks_with_empty.append(rack)

        item_to_item_possible = (len(racks_with_items) >= 2)
        item_to_empty_possible = (len(racks_with_items) >= 1 and len(racks_with_empty) >= 1)

        if not item_to_item_possible and not item_to_empty_possible:
            print("No cross-rack swap possible, skipping mutation.")
            return warehouse

        if item_to_item_possible and item_to_empty_possible:
            do_item_to_item = (random.random() < 0.5)
        elif item_to_item_possible:
            do_item_to_item = True
        else:
            do_item_to_item = False

        # Step 2: Perform the chosen cross-rack operation
        if do_item_to_item:
            rack1, rack2 = random.sample(racks_with_items, 2)

            all_positions_1 = [
                (i, j, k)
                for i in range(rack1.dimensions[0])
                for j in range(rack1.dimensions[1])
                for k in range(rack1.dimensions[2])
                if rack1.storage[i, j, k] is not None
            ]
            all_positions_2 = [
                (i, j, k)
                for i in range(rack2.dimensions[0])
                for j in range(rack2.dimensions[1])
                for k in range(rack2.dimensions[2])
                if rack2.storage[i, j, k] is not None
            ]
            if not all_positions_1 or not all_positions_2:
                #print("Cannot do item-to-item swap: one rack is unexpectedly empty.")
                return warehouse

            pos1 = random.choice(all_positions_1)
            pos2 = random.choice(all_positions_2)

            item1 = rack1.storage[pos1]
            item2 = rack2.storage[pos2]

            rack1.storage[pos1], rack2.storage[pos2] = item2, item1

        else:
            rack_with_items = random.choice(racks_with_items)
            rack_with_empty = random.choice(racks_with_empty)

            occupied_positions = [
                (i, j, k)
                for i in range(rack_with_items.dimensions[0])
                for j in range(rack_with_items.dimensions[1])
                for k in range(rack_with_items.dimensions[2])
                if rack_with_items.storage[i, j, k] is not None
            ]
            supported_empty_positions = [
                (i, j, k)
                for i, j, k in rack_with_empty.get_supported_positions()
                if rack_with_empty.storage[i, j, k] is None
            ]

            if not occupied_positions or not supported_empty_positions:
                #print("Cannot do item-to-empty swap: missing occupied or empty spots.")
                return warehouse

            pos1 = random.choice(occupied_positions)
            pos2 = random.choice(supported_empty_positions)

            item_to_move = rack_with_items.storage[pos1]
            rack_with_items.storage[pos1] = None
            rack_with_empty.storage[pos2] = item_to_move

        # Step 3: Accept/Reject the mutation
        mutated_cost = self.objective_function(new_warehouse)
        original_cost_epsilon = 1e-6

        if mutated_cost < original_cost - original_cost_epsilon:
            #print(f"Mutation accepted: cost improved from {original_cost:.6f} to {mutated_cost:.6f}")
            return new_warehouse
        else:
            # Accept with fallback probability (allowing some worse solutions)
            if random.random() < self.fallback_prob:
                #print(f"Mutation accepted (random chance) despite cost {mutated_cost:.6f} vs {original_cost:.6f}")
                return new_warehouse
            else:
                #print(f"Mutation rejected: cost {mutated_cost:.6f} did not improve from {original_cost:.6f}")
                return warehouse

    def crossover(self, parent1, parent2):
        """
        Perform uniform crossover between two parent warehouse states.
        For each position in the warehouse storage, with 50% probability, copy the item from parent2.
        """
        child = copy.deepcopy(parent1)
        for z_idx, zone in enumerate(child.zones):
            for a_idx, aisle in enumerate(zone.aisles):
                for r_idx, rack in enumerate(aisle.racks):
                    rack_p2 = parent2.zones[z_idx].aisles[a_idx].racks[r_idx]
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                if random.random() < 0.5:
                                    child.zones[z_idx].aisles[a_idx].racks[r_idx].storage[i, j, k] = rack_p2.storage[i, j, k]
        return child

    def select_parent(self):
        """Tournament selection with tournament size = 3."""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        # Sort by ascending cost, so that the lowest cost is index 0
        tournament.sort(key=lambda wh: self.objective_function(wh))
        return tournament[0]

    def run(self):
        """Run the genetic algorithm and return the best warehouse state found."""
        print("Run start: ")
        self.generate_initial_population()
        best_candidate = None
        best_cost = float('inf')

        for gen in range(self.generations):
            new_population = []

            # 1) Elitism:
            sorted_population = sorted(self.population, key=lambda wh: self.objective_function(wh))
            elitism_count = max(1, int(self.elitism_rate * self.population_size))
            elite_individuals = sorted_population[:elitism_count]
            new_population.extend(elite_individuals)

            # 2) Adaptive mutation rate:
            effective_mutation_rate = self.mutation_rate * (1 - (gen / self.generations) * 0.5)

            # 3) Fill up the new population
            while len(new_population) < self.population_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()

                # Possibly do crossover
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                # Possibly mutate
                if random.random() < effective_mutation_rate:
                    child = self.mutate(child)

                new_population.append(child)

            self.population = new_population

            # Check the best solution this generation
            for candidate in self.population:
                cost = self.objective_function(candidate)
                if cost < best_cost:
                    best_cost = cost
                    best_candidate = candidate

            print(f"Generation {gen+1}, best cost: {best_cost:.6f}", flush=True)

        print("End of run")
        return best_candidate

# Example usage:

if __name__ == "__main__":
    warehouse = build_sample_warehouse(
        num_zones=7,
        num_aisles=4,
        num_racks=2,
        rack_dimensions=(5, 4, 6),
        rack_spacing=(2, 2, 0.5),
        show_vis=False
    )

    populate_warehouse(warehouse, 300)

    ga = GeneticAlgorithm(
        warehouse,
        population_size=300,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.3,
        fallback_prob=0.05,
        elitism_rate=0.05
    )

    optimized_warehouse = ga.run()
    # optimized_warehouse.visualize()
