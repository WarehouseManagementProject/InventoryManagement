import numpy as np
import random
import math
from warehouse import build_sample_warehouse, populate_warehouse


class SimulatedAnnealing:
    def __init__(self, warehouse, initial_temp=1000, cooling_rate=0.99, min_temp=1):
        self.warehouse = warehouse
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.best_state = None
        self.best_cost = float('inf')


    '''
    this is currently only a dummy objective function. Barely accounts for urgency and distance.
    I will need more clarity on properties of a warehouse item. Goal is to have everything in place so that
    the algorithm works irrespective of how bad it is.
    
    '''
    
    def objective_function(self):
        total_distance = 0
        for zone in self.warehouse.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                item = rack.storage[i, j, k]
                                if item:
                                    total_distance += item.retrieval_urgency * np.linalg.norm(rack.coordinates)
        return total_distance
    
    '''
    Also temporary. Currently it generates a lot of mutations. Env could have a rollback option.
    '''
    def generate_neighbor(self):
        new_warehouse = self.warehouse  # Copy the warehouse state
        zone = random.choice(new_warehouse.zones)
        aisle = random.choice(zone.aisles)
        rack = random.choice(aisle.racks)
        positions = rack.get_supported_positions()
        if not positions:
            return new_warehouse
        pos1, pos2 = random.sample(positions, 2) if len(positions) > 1 else (positions[0], positions[0])
        rack.storage[pos1], rack.storage[pos2] = rack.storage[pos2], rack.storage[pos1]
        return new_warehouse
        

    def acceptance_probability(self, old_cost, new_cost):
        if new_cost < old_cost:
            return 1.0
        return math.exp((old_cost - new_cost) / self.temperature)

    def optimize(self, iterations=1000):
        pass