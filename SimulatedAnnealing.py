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

    def objective_function(self):
        pass
    def generate_neighbor(self):
        pass
        

    def acceptance_probability(self, old_cost, new_cost):
        pass

    def optimize(self, iterations=1000):
        pass