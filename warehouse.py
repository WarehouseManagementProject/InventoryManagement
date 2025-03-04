import numpy as np
import pyvista as pv
import random

item_directory = {
    "Electronics": {
        "Laptops": (0.2, 0.4, 0.8),
        "Phones": (0.4, 0.6, 0.9),
        "Accessories": (0.1, 0.3, 0.7)
    },
    "Books": {
        "Fiction": (0.8, 0.2, 0.2),
        "Non-Fiction": (0.9, 0.4, 0.4),
        "Magazines": (0.7, 0.1, 0.1)
    },
    "Clothing": {
        "Shirts": (0.2, 0.8, 0.2),
        "Pants": (0.4, 0.9, 0.4),
        "Shoes": (0.1, 0.7, 0.1)
    },
    "Food": {
        "Snacks": (0.8, 0.8, 0.2),
        "Drinks": (0.9, 0.9, 0.4),
        "Ingredients": (0.7, 0.7, 0.1)
    }
}

class Item:
    def __init__(self, category, sub_category, weight, dimensions, product_name, retrieval_urgency):
        self.category = category
        self.sub_category = sub_category
        self.weight = weight
        self.dimensions = dimensions
        self.product_name = product_name
        self.retrieval_urgency = retrieval_urgency
        self.id = f"{product_name}_{category}_{sub_category}"

class Rack:
    def __init__(self, dimensions, coordinates):
        self.dimensions = dimensions
        self.coordinates = coordinates
        self.id = f"Rack_{coordinates[0]}_{coordinates[1]}_{coordinates[2]}"
        self.description = f"Rack at {coordinates}"
        self.storage = np.empty(dimensions, dtype=object)

    def place_item(self, item, position):
        if 0 <= position[0] < self.dimensions[0] and 0 <= position[1] < self.dimensions[1] and 0 <= position[2] < self.dimensions[2]:
            if self.can_place_item(item, position):
                self.storage[position] = item
                return True
        return False

    def can_place_item(self, item, position):
        for x in range(position[0], position[0] + item.dimensions[0]):
            for y in range(position[1], position[1] + item.dimensions[1]):
                for z in range(position[2], position[2] + item.dimensions[2]):
                    if (x >= self.dimensions[0] or y >= self.dimensions[1] or z >= self.dimensions[2] or
                            self.storage[x, y, z] is not None):
                        return False
        if position[2] == 0:
            return True
        else:
            bottom_center_x = position[0] + item.dimensions[0] / 2.0
            bottom_center_y = position[1] + item.dimensions[1] / 2.0
            cell_x = int(bottom_center_x)
            cell_y = int(bottom_center_y)
            if self.storage[cell_x, cell_y, position[2] - 1] is not None:
                return True
            else:
                return False

    def get_supported_positions(self):
        supported_positions = []
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                for k in range(self.dimensions[2]):
                    if self.storage[i, j, k] is None:
                        if k == 0 or (k > 0 and self.storage[i,j,k-1] is not None):
                            supported_positions.append((i,j,k))
        return supported_positions

class Aisle:
    def __init__(self, id, description):
        self.racks = []
        self.id = id
        self.description = description

    def add_rack(self, rack):
        self.racks.append(rack)

class Zone:
    def __init__(self, id, description):
        self.aisles = []
        self.id = id
        self.description = description

    def add_aisle(self, aisle):
        self.aisles.append(aisle)

class Warehouse:
    def __init__(self):
        self.zones = []

    def add_zone(self, zone):
        self.zones.append(zone)
        
    def find_space(self, item):
        random.shuffle(self.zones)
        for zone in self.zones:
            random.shuffle(zone.aisles)
            for aisle in zone.aisles:
                random.shuffle(aisle.racks)
                for rack in aisle.racks:
                    supported_positions = rack.get_supported_positions()
                    random.shuffle(supported_positions)
                    for position in supported_positions:
                        if rack.can_place_item(item, position):
                            if rack.place_item(item,position):
                                return rack, position
        return None, None

    def visualize(self):
        plotter = pv.Plotter()
        plotter.set_background('white')
        visualized_items = set()

        for zone in self.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    x, y, z = rack.coordinates
                    dx, dy, dz = rack.dimensions
                    
                    rack_mesh = pv.Cube(
                        center=(x + dx/2, y + dy/2, z + dz/2),
                        x_length=dx, y_length=dy, z_length=dz
                    )
                    plotter.add_mesh(rack_mesh, color="lightgray", opacity=0.5, show_edges=True)

                    for i in range(dx):
                        for j in range(dy):
                            for k in range(dz):
                                item = rack.storage[i, j, k]
                                if item and item.id not in visualized_items:
                                    visualized_items.add(item.id)
                                    item_color = item_directory[item.category][item.sub_category]
                                    
                                    item_dx, item_dy, item_dz = item.dimensions
                                    item_mesh = pv.Cube(
                                        center=(
                                            x + i + item_dx/2, 
                                            y + j + item_dy/2, 
                                            z + k + item_dz/2
                                        ),
                                        x_length=item_dx, 
                                        y_length=item_dy, 
                                        z_length=item_dz
                                    )
                                    
                                    plotter.add_mesh(
                                        item_mesh, 
                                        color=item_color, 
                                        opacity=1.0
                                    )

        plotter.enable_trackball_style()
        def set_zoom(value):
            plotter.camera.zoom(value)
            plotter.render()

        plotter.add_slider_widget(
            callback=set_zoom,
            rng=[0.1, 2.0],
            value=1.0,
            title='Zoom Level',
            pointa=(0.1, 0.1),
            pointb=(0.4, 0.1)
        )
        plotter.show()
        plotter.close()

def build_sample_warehouse(num_zones, num_aisles, num_racks, rack_dimensions, rack_spacing):
    warehouse = Warehouse()
    for z_idx in range(num_zones):
        zone = Zone(id=f"Z{z_idx}", description=f"Zone {z_idx}")
        for a_idx in range(num_aisles):
            aisle = Aisle(id=f"A{a_idx}", description=f"Aisle {a_idx} in Zone {z_idx}")
            for r_idx in range(num_racks):
                rack_coords = (a_idx * (rack_dimensions[0] + rack_spacing[0]),
                               z_idx * (rack_dimensions[1] + rack_spacing[1]),
                               r_idx * (rack_dimensions[2] + rack_spacing[2]))
                rack = Rack(dimensions=rack_dimensions, coordinates=rack_coords)
                aisle.add_rack(rack)
            zone.add_aisle(aisle)
        warehouse.add_zone(zone)
    return warehouse

# This should be updated to make it more maintainable. Instead of using populate warehouse function, 
# We should pass a list of items directly to the warehouse object and warehouse should take care of placing those.
# So basically move this logic into warehouse class.
def populate_warehouse(warehouse, num_items):
    for i in range(num_items):
        category = random.choice(list(item_directory.keys()))
        sub_category = random.choice(list(item_directory[category].keys()))
        weight = random.uniform(0.1, 5.0)
        dimensions = (random.randint(1,3), random.randint(1,2), random.randint(1,2))
        product_name = f"Item_{i}"
        retrieval_urgency = random.randint(1, 5)
        item = Item(category, sub_category, weight, dimensions, product_name, retrieval_urgency)
        rack, position = warehouse.find_space(item)
        if rack is None:
            print(f"No space found for {product_name}")

warehouse = build_sample_warehouse(num_zones=2, num_aisles=3, num_racks=2,
                                  rack_dimensions=(5, 4, 6), rack_spacing=(2, 2, 0.5))
populate_warehouse(warehouse, 30)
warehouse.visualize()
