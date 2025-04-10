import numpy as np
from warehouse import Warehouse, Item, Rack, build_sample_warehouse, item_directory
import random
random.seed(905248)
class SimulatedAnn:
    def __init__(self, warehouse: Warehouse, items: list[Item]):
        self.warehouse = warehouse
        self.listOfItems = items
        self.unPlacedItems = len(self.listOfItems)
        self.currEntropy = self.unPlacedItems * 35

    def calculateChangeInEntropy(self, item, rack, position, zone, aisle):
        category_score = 0
        if item.category == zone.category:
            category_score -= 100  # Strong penalty reduction for correct category
            if item.sub_category == aisle.sub_category:
                category_score -= 200  # Additional reduction for correct sub-category
        else:
            category_score += 500  # Large penalty for incorrect category

        # Calculate global position for accurate distance measurement
        rack_x, rack_y, rack_z = rack.coordinates
        global_x = rack_x + position[0]
        global_y = rack_y + position[1]
        global_z = rack_z + position[2]
        dist_from_origin = abs(global_x) + abs(global_y) + abs(global_z)
        distance_score = dist_from_origin * item.retrieval_urgency

        # Weight penalty based on vertical position in rack (z-axis)
        weight_score = position[2] * item.weight

        # Combine factors with category/sub-category dominating
        entropy = (category_score) + (0.01 * distance_score) + (0.01 * weight_score) - 35
        return entropy

    def train(self, initialTemperature=1000, decay=0.9):
        for item in self.listOfItems:
            isItemPlaced = False
            tries = 0
            while not isItemPlaced and tries < 10000:
                rack, position, zone, aisle = self.warehouse.add_item(item)
                if not rack:
                    tries += 1
                    continue

                entropy_change = self.calculateChangeInEntropy(item, rack, position, zone, aisle)
                new_entropy = self.currEntropy + entropy_change

                if new_entropy < self.currEntropy:
                    self.currEntropy = new_entropy
                    isItemPlaced = True
                else:
                    delta = new_entropy - self.currEntropy
                    probability = np.exp(-delta / initialTemperature)
                    if random.uniform(0, 1) <= probability:
                        self.currEntropy = new_entropy
                        isItemPlaced = True
                    else:
                        self.warehouse.undo_item()

                tries += 1

            self.unPlacedItems -= 1
            initialTemperature *= decay
        return self.warehouse

def generateItems(num_items):
    generatedItems = []
    for i in range(num_items):
        category = random.choice(list(item_directory.keys()))
        sub_category = random.choice(list(item_directory[category].keys()))
        weight = random.uniform(0.1, 5.0)
        dimensions = (random.randint(1,3), random.randint(1,2), random.randint(1,2))
        product_name = f"Item_{i}"
        retrieval_urgency = random.randint(1, 5)
        item = Item(category, sub_category, weight, dimensions, product_name, retrieval_urgency)
        generatedItems.append(item)
    return generatedItems

# Warehouse setup and execution
warehouse = build_sample_warehouse(
    num_zones=6,
    num_aisles=3,
    num_racks=3,
    rack_dimensions=(5, 4, 6),
    rack_spacing=(2, 2, 0.5),
    show_vis=True
)

warehouse.start_visualization()
items = generateItems(200)
simulated_annealing = SimulatedAnn(warehouse, items)
optimized_warehouse = simulated_annealing.train()
warehouse.show_final_state()