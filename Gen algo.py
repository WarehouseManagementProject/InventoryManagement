import copy
import random
import numpy as np
import math
from warehouse import build_sample_warehouse, populate_warehouse
from warehouse import item_directory

"""
Genetic Algorithm that performs cross-rack swaps:
- 'item-to-item' swap: picks two racks that each have at least one item,
  then randomly swaps one occupied slot from rack1 with one occupied slot from rack2.
- 'item-to-empty' swap: picks one rack that has at least one item,
  plus one rack that has at least one supported empty slot,
  then moves an item from the first to a supported empty position in the second.
"""
def count_unique_items(warehouse):
    seen_items = set()
    for zone in warehouse.zones:
        for aisle in zone.aisles:
            for rack in aisle.racks:
                for cell_item in rack.storage.flatten():
                    if cell_item is not None:
                        seen_items.add(cell_item.id)
    return len(seen_items)

class GeneticAlgorithm:
    def __init__(
        self,
        warehouse,
        ga_config
    ):
        self.initial_warehouse = warehouse
        self.__dict__.update(ga_config)

    def objective_function(self, warehouse):
        total_cost = 0.0
        w_zone = 200.0      # Highest penalty for zone category mismatch
        w_aisle = 50.0     # Aisle subcategory mismatch
        w_rack = 150.0      # Rack subcategory mismatch
        w_high_urgency = 150.0  # High urgency item not in bottom rack
        high_urgency_threshold = 4
      
        # 1. Zone Category penalty
        for zone in warehouse.zones:
            expected_category = zone.category
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                item = rack.storage[i, j, k]
                                if item and item.category != expected_category:
                                    total_cost += w_zone 

        # 2. Aisle Subcategory Penalty
        for zone in warehouse.zones:
            for aisle in zone.aisles:
                subcat_counts = {}
                total_items = 0
                for rack in aisle.racks:
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                item = rack.storage[i, j, k]
                                if item:
                                    total_items += 1
                                    subcat = item.sub_category
                                    subcat_counts[subcat] = subcat_counts.get(subcat, 0) + 1
                if total_items > 0:
                    dominant = max(subcat_counts.values(), default=0)
                    penalty = w_aisle * (total_items - dominant)
                    total_cost += penalty

        # 3. Rack Subcategory Penalty
        for zone in warehouse.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    subcat_counts = {}
                    total_items = 0
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                item = rack.storage[i, j, k]
                                if item:
                                    total_items += 1
                                    subcat = item.sub_category
                                    subcat_counts[subcat] = subcat_counts.get(subcat, 0) + 1
                    if total_items > 0:
                        dominant = max(subcat_counts.values(), default=0)
                        penalty = w_rack * (total_items - dominant)
                        total_cost += penalty

        # 4. High Urgency Item(must be in bottom rack)
        for zone in warehouse.zones:
            for aisle in zone.aisles:
                if not aisle.racks:
                    continue
                bottom_rack = aisle.racks[0]
                for rack in aisle.racks:
                    is_bottom_rack = (rack == bottom_rack)
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                item = rack.storage[i, j, k]
                                if item and item.retrieval_urgency >= high_urgency_threshold:
                                    if not is_bottom_rack:
                                        total_cost += w_high_urgency
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
        1) item-to-item swap:
            - pick 2 racks that each have at least 1 occupied cell
            - pick one occupied cell in rack1, one in rack2
            - swap them (only those single cells)
        2) item-to-empty:
            - pick 1 rack that has at least 1 occupied cell
            - pick 1 rack that has at least 1 empty cell
            - move one single cell from the first rack to one empty cell in the second rack
        """

        original_cost = self.objective_function(warehouse)

        # 2) Gather racks_with_items and racks_with_empty
        racks_with_items = []
        racks_with_empty = []
        for zone in warehouse.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    # Check if there's at least one occupied cell
                    occupied_positions = [(i, j, k)
                                        for i in range(rack.dimensions[0])
                                        for j in range(rack.dimensions[1])
                                        for k in range(rack.dimensions[2])
                                        if rack.storage[i, j, k] is not None]
                    if occupied_positions:
                        racks_with_items.append(rack)

                    empty_positions = [(i, j, k)
                                    for i in range(rack.dimensions[0])
                                    for j in range(rack.dimensions[1])
                                    for k in range(rack.dimensions[2])
                                    if rack.storage[i, j, k] is None]
                    if empty_positions:
                        racks_with_empty.append(rack)

        # If there aren't enough racks to do item-to-item or item-to-empty, skip mutation
        item_to_item_possible = (len(racks_with_items) >= 2)
        item_to_empty_possible = (len(racks_with_items) >= 1 and len(racks_with_empty) >= 1)
        if not item_to_item_possible and not item_to_empty_possible:
            return warehouse

        # Decide which operation
        if item_to_item_possible and item_to_empty_possible:
            do_item_to_item = (random.random() < 0.5)
        elif item_to_item_possible:
            do_item_to_item = True
        else:
            do_item_to_item = False

        # Store the changed cells so as to revert if cost doesn't improve
        changes = [] 

        def revert():
            for rack, (pos, old_val) in changes:
                rack.storage[pos] = old_val

        # 3) Perform the chosen operation
        if do_item_to_item:
            # ITEM-to-ITEM SWAP
            rack1, rack2 = random.sample(racks_with_items, 2)

            # Pick a random occupied cell from each
            occupied1 = [(i, j, k)
                        for i in range(rack1.dimensions[0])
                        for j in range(rack1.dimensions[1])
                        for k in range(rack1.dimensions[2])
                        if rack1.storage[i, j, k] is not None]
            occupied2 = [(i, j, k)
                        for i in range(rack2.dimensions[0])
                        for j in range(rack2.dimensions[1])
                        for k in range(rack2.dimensions[2])
                        if rack2.storage[i, j, k] is not None]

            if not occupied1 or not occupied2:
                return warehouse

            pos1 = random.choice(occupied1)
            pos2 = random.choice(occupied2)

            item1 = rack1.storage[pos1]
            item2 = rack2.storage[pos2]

            changes.append((rack1, (pos1, item1)))
            changes.append((rack2, (pos2, item2)))

            # Perform the swap
            rack1.storage[pos1] = item2
            rack2.storage[pos2] = item1

        else:
            # ITEM-to-EMPTY MOVE
            rack_with_items = random.choice(racks_with_items)
            rack_with_empty = random.choice(racks_with_empty)

            occupied_positions = [(i, j, k)
                                for i in range(rack_with_items.dimensions[0])
                                for j in range(rack_with_items.dimensions[1])
                                for k in range(rack_with_items.dimensions[2])
                                if rack_with_items.storage[i, j, k] is not None]
            empty_positions = [(i, j, k)
                            for i in range(rack_with_empty.dimensions[0])
                            for j in range(rack_with_empty.dimensions[1])
                            for k in range(rack_with_empty.dimensions[2])
                            if rack_with_empty.storage[i, j, k] is None]

            if not occupied_positions or not empty_positions:
                return warehouse

            pos1 = random.choice(occupied_positions)
            pos2 = random.choice(empty_positions)

            item_to_move = rack_with_items.storage[pos1]

            # Record old states for revert
            changes.append((rack_with_items, (pos1, item_to_move)))
            changes.append((rack_with_empty, (pos2, rack_with_empty.storage[pos2])))

            rack_with_items.storage[pos1] = None
            rack_with_empty.storage[pos2] = item_to_move

        # 4) Cost-based acceptance
        mutated_cost = self.objective_function(warehouse)
        if mutated_cost < (original_cost - 1e-6) or random.random() < self.fallback_prob:
            return warehouse
        else:
            revert()
            return warehouse


    def crossover(self, parent1, parent2):
        """
        All-or-nothing item-level crossover:
        1) Make a child with the same rack geometry as parent1 but empty.
        2) Collect all items from both parents (union of their IDs).
        3) Try to place each item into the child's racks.
        4) If even one item can't be placed, revert by returning a deep copy of parent1.
        
        """

        # 1) Copy parent's geometry
        child = copy.deepcopy(parent1)
        for zone in child.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    rack.storage.fill(None)

        # Helper to record item positions
        def record_positions(warehouse):
            positions = {}
            for zone in warehouse.zones:
                for aisle in zone.aisles:
                    for rack in aisle.racks:
                        for i in range(rack.dimensions[0]):
                            for j in range(rack.dimensions[1]):
                                for k in range(rack.dimensions[2]):
                                    item = rack.storage[i, j, k]
                                    if item is not None:
                                        positions[item.id] = (rack, (i, j, k))
            return positions

        parent1_positions = record_positions(parent1)
        parent2_positions = record_positions(parent2)

        all_item_ids = set(parent1_positions.keys()).union(parent2_positions.keys())

        def find_child_rack(child_warehouse, rack_id):
            for z in child_warehouse.zones:
                for a in z.aisles:
                    for r in a.racks:
                        if r.id == rack_id:
                            return r
            return None

        def attempt_place_at_parent_coords(child_warehouse, item_id, parent_positions):
            if item_id not in parent_positions:
                return False
            parent_rack, (i, j, k) = parent_positions[item_id]
            parent_rack_id = parent_rack.id

            item_obj = parent_rack.storage[i, j, k]
            if not item_obj:
                return False

            child_rack = find_child_rack(child_warehouse, parent_rack_id)
            if not child_rack:
                return False

            return child_rack.place_item(item_obj, (i, j, k))

        # Fallback:
        def attempt_place_anywhere(child_warehouse, item_obj):
            for z in child_warehouse.zones:
                for a in z.aisles:
                    for r in a.racks:
                        positions = r.get_supported_positions()
                        random.shuffle(positions)
                        for pos in positions:
                            if r.place_item(item_obj, pos):
                                return True
            return False

        # 2) Place each item. If any fail, revert to parent1 entirely.
        for item_id in all_item_ids:
            if item_id in parent1_positions and item_id in parent2_positions:
                if random.random() < 0.5:
                    first_positions = parent1_positions
                    second_positions = parent2_positions
                else:
                    first_positions = parent2_positions
                    second_positions = parent1_positions

                if not attempt_place_at_parent_coords(child, item_id, first_positions):
                    if not attempt_place_at_parent_coords(child, item_id, second_positions):
                        item_obj = (first_positions[item_id][0].storage[first_positions[item_id][1]] 
                                    if item_id in first_positions 
                                    else second_positions[item_id][0].storage[second_positions[item_id][1]])
                        if not attempt_place_anywhere(child, item_obj):
                            return copy.deepcopy(parent1) 
            elif item_id in parent1_positions:
                if not attempt_place_at_parent_coords(child, item_id, parent1_positions):
                    item_obj = parent1_positions[item_id][0].storage[parent1_positions[item_id][1]]
                    if not attempt_place_anywhere(child, item_obj):
                        return copy.deepcopy(parent1)
            elif item_id in parent2_positions:
                if not attempt_place_at_parent_coords(child, item_id, parent2_positions):
                    item_obj = parent2_positions[item_id][0].storage[parent2_positions[item_id][1]]
                    if not attempt_place_anywhere(child, item_obj):
                        return copy.deepcopy(parent1)
            else:
                return copy.deepcopy(parent1)

        return child


    def select_parent(self):
        """Tournament selection with tournament size = 3."""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        # Sort by ascending cost
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
        num_zones=4,
        num_aisles=4,
        num_racks=3,
        rack_dimensions=(5, 4, 6),
        rack_spacing=(2, 2, 0.5),
        show_vis=True
    )

    populate_warehouse(warehouse, 200)

    before_count = count_unique_items(warehouse)
    print(f"Items before GA: {before_count}")

    initial_state = copy.deepcopy(warehouse)

    ga_config = {
    "population_size": 400,
    "generations": 300,
    "crossover_rate": 0.8,
    "mutation_rate": 0.25,
    "fallback_prob": 0.075,
    "elitism_rate": 0.08,
    }

    ga = GeneticAlgorithm(warehouse, ga_config)

    optimized_warehouse = ga.run()

    after_count = count_unique_items(optimized_warehouse)
    print(f"Items after GA: {after_count}")

    initial_state.show_vis = True
    optimized_warehouse.show_vis = True

    print("Initial warehouse state:")
    initial_state.show_final_state()

    print("Optimized warehouse state:")
    optimized_warehouse.show_final_state()
