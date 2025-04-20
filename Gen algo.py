import copy
import random
import numpy as np
import math
from warehouse import build_sample_warehouse, populate_warehouse, item_directory

"""
Genetic Algorithm that performs cross-rack swaps:
- 'item-to-item' swap: picks two racks that each have at least one item,
  then randomly swaps one occupied slot from rack1 with one occupied slot from rack2.
- 'item-to-empty' swap: picks one rack that has at least one item,
  plus one rack that has at least one supported empty slot,
  then moves an item from the first to a supported empty position in the second.
"""

class GeneticAlgorithm:
    def __init__(self, warehouse, ga_config):
        self.initial_warehouse = warehouse
        self.__dict__.update(ga_config)

    def objective_function(self, warehouse):
        total_cost = 0.0
        w_zone = 100.0
        w_aisle = 30.0
        w_rack = 70.0
        w_high_urgency = 100.0
        high_urgency_threshold = 4

        zone_categories = list(item_directory.keys())

        # 1. Zone Category Enforcement
        for zone_idx, zone in enumerate(warehouse.zones):
            expected = zone_categories[zone_idx % len(zone_categories)]
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                itm = rack.storage[i, j, k]
                                if itm and itm.category != expected:
                                    total_cost += w_zone

        # 2. Aisle Subcategory Purity
        for zone in warehouse.zones:
            for aisle in zone.aisles:
                counts = {}
                n_items = 0
                for rack in aisle.racks:
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                itm = rack.storage[i, j, k]
                                if itm:
                                    n_items += 1
                                    counts[itm.sub_category] = counts.get(itm.sub_category, 0) + 1
                if n_items > 0:
                    dominant = max(counts.values(), default=0)
                    total_cost += w_aisle * (n_items - dominant)

        # 3. Rack Subcategory Purity
        for zone in warehouse.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    counts = {}
                    n_items = 0
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                itm = rack.storage[i, j, k]
                                if itm:
                                    n_items += 1
                                    counts[itm.sub_category] = counts.get(itm.sub_category, 0) + 1
                    if n_items > 0:
                        dominant = max(counts.values(), default=0)
                        total_cost += w_rack * (n_items - dominant)

        # 4. High-Urgency Item Placement
        for zone in warehouse.zones:
            for aisle in zone.aisles:
                if not aisle.racks:
                    continue
                bottom = aisle.racks[0]
                for rack in aisle.racks:
                    is_bottom = (rack is bottom)
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                itm = rack.storage[i, j, k]
                                if itm and itm.retrieval_urgency >= high_urgency_threshold:
                                    if not is_bottom:
                                        total_cost += w_high_urgency

        return total_cost

    def generate_initial_population(self):
        self.population = []
        for _ in range(self.population_size):
            candidate = copy.deepcopy(self.initial_warehouse)
            for _ in range(random.randint(1, 10)):
                candidate = self.mutate(candidate)
            self.population.append(candidate)

    def mutate(self, warehouse):
        orig_cost = self.objective_function(warehouse)
        new_wh = copy.deepcopy(warehouse)

        racks_with_items = []
        racks_with_empty = []
        for zone in new_wh.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    has_item = any(rack.storage.flatten())
                    empty_pos = [p for p in rack.get_supported_positions() if rack.storage[p] is None]
                    if has_item:
                        racks_with_items.append(rack)
                    if empty_pos:
                        racks_with_empty.append(rack)

        item2item = len(racks_with_items) >= 2
        item2empty = len(racks_with_items) >= 1 and len(racks_with_empty) >= 1

        if not item2item and not item2empty:
            return warehouse

        if item2item and item2empty:
            do_i2i = random.random() < 0.5
        else:
            do_i2i = item2item

        if do_i2i:
            r1, r2 = random.sample(racks_with_items, 2)
            pos1 = random.choice([(i,j,k)
                                  for i in range(r1.dimensions[0])
                                  for j in range(r1.dimensions[1])
                                  for k in range(r1.dimensions[2])
                                  if r1.storage[i,j,k] is not None])
            pos2 = random.choice([(i,j,k)
                                  for i in range(r2.dimensions[0])
                                  for j in range(r2.dimensions[1])
                                  for k in range(r2.dimensions[2])
                                  if r2.storage[i,j,k] is not None])
            r1.storage[pos1], r2.storage[pos2] = r2.storage[pos2], r1.storage[pos1]

        else:
            src = random.choice(racks_with_items)
            dst = random.choice(racks_with_empty)
            pos1 = random.choice([(i,j,k)
                                  for i in range(src.dimensions[0])
                                  for j in range(src.dimensions[1])
                                  for k in range(src.dimensions[2])
                                  if src.storage[i,j,k] is not None])
            pos2 = random.choice([p for p in dst.get_supported_positions() if dst.storage[p] is None])
            dst.storage[pos2] = src.storage[pos1]
            src.storage[pos1] = None

        mutated_cost = self.objective_function(new_wh)
        if mutated_cost < orig_cost - 1e-6:
            return new_wh
        elif random.random() < self.fallback_prob:
            return new_wh
        else:
            return warehouse

    def crossover(self, parent1, parent2):
        child = copy.deepcopy(parent1)
        for zone in child.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    rack.storage.fill(None)

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

        def attempt_place_at_parent_coords(child_w, item_id, parent_positions):
            if item_id not in parent_positions:
                return False
            rack_ref, pos = parent_positions[item_id]
            child_rack = find_child_rack(child_w, rack_ref.id)
            if not child_rack:
                return False
            return child_rack.place_item(rack_ref.storage[pos], pos)

        def attempt_place_anywhere(child_w, item_obj):
            for z in child_w.zones:
                for a in z.aisles:
                    for r in a.racks:
                        pos_list = r.get_supported_positions()
                        random.shuffle(pos_list)
                        for p in pos_list:
                            if r.place_item(item_obj, p):
                                return True
            return False

        for item_id in all_item_ids:
            if item_id in parent1_positions and item_id in parent2_positions:
                first, second = (parent1_positions, parent2_positions) if random.random() < 0.5 else (parent2_positions, parent1_positions)
                if not attempt_place_at_parent_coords(child, item_id, first):
                    if not attempt_place_at_parent_coords(child, item_id, second):
                        item_obj = first[item_id][0].storage[first[item_id][1]]
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
        tour = random.sample(self.population, 3)
        tour.sort(key=lambda wh: self.objective_function(wh))
        return tour[0]

    def run(self):
        print("Run start:")
        self.generate_initial_population()
        best_wh = None
        best_cost = float('inf')

        for gen in range(self.generations):
            new_pop = []

            # Elitism
            sorted_pop = sorted(self.population, key=lambda wh: self.objective_function(wh))
            elite_n = max(1, int(self.elitism_rate * self.population_size))
            new_pop.extend(sorted_pop[:elite_n])

            # Adaptive mutation rate
            mut_rate = self.mutation_rate * (1 - (gen / self.generations) * 0.5)

            while len(new_pop) < self.population_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                if random.random() < mut_rate:
                    child = self.mutate(child)
                new_pop.append(child)

            self.population = new_pop

            for wh in self.population:
                cost = self.objective_function(wh)
                if cost < best_cost:
                    best_cost, best_wh = cost, wh

            print(f"Generation {gen+1}/{self.generations}, best cost: {best_cost:.6f}", flush=True)

        print("End of run")
        return best_wh


if __name__ == "__main__":
    wh = build_sample_warehouse(
        num_zones=4,
        num_aisles=3,
        num_racks=3,
        rack_dimensions=(5, 4, 6),
        rack_spacing=(2, 2, 0.5),
        show_vis=True
    )
    populate_warehouse(wh, 200)
    initial_state = copy.deepcopy(wh)

    # GA configuration
    ga_config = {
        "population_size": 400,
        "generations": 800,
        "crossover_rate": 0.8,
        "mutation_rate": 0.3,
        "fallback_prob": 0.05,
        "elitism_rate": 0.05,
        "dock_zone_index": 0
    }

    ga = GeneticAlgorithm(wh, ga_config)
    optimized = ga.run()

    initial_state.show_vis = True
    optimized.show_vis = True

    print("Initial warehouse state:")
    initial_state.show_final_state()

    print("Optimized warehouse state:")
    optimized.show_final_state()
