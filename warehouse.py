import numpy as np
import pyvista as pv
import random
import time
import platform

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
    def __init__(self, dimensions, coordinates, category = None, subCategory = None):
        self.dimensions = dimensions
        self.coordinates = coordinates
        self.id = f"Rack_{coordinates[0]}_{coordinates[1]}_{coordinates[2]}"
        self.description = f"Rack at {coordinates}"
        self.storage = np.empty(dimensions, dtype=object)
        self.category = category
        self.subCategory = subCategory

    def place_item(self, item, position):
        if 0 <= position[0] < self.dimensions[0] and 0 <= position[1] < self.dimensions[1] and 0 <= position[2] < self.dimensions[2]:
            if self.can_place_item(item, position):
                self.storage[position] = item
                return True
        return False

    def remove_item(self, position):
        if 0 <= position[0] < self.dimensions[0] and 0 <= position[1] < self.dimensions[1] and 0 <= position[2] < self.dimensions[2]:
            item = self.storage[position]
            if item is not None:
                for x in range(position[0], min(position[0] + item.dimensions[0], self.dimensions[0])):
                    for y in range(position[1], min(position[1] + item.dimensions[1], self.dimensions[1])):
                        for z in range(position[2], min(position[2] + item.dimensions[2], self.dimensions[2])):
                            self.storage[x, y, z] = None
                return item
        return None
    def can_place_item(self, item, position):
        for x in range(position[0], position[0] + item.dimensions[0]):
            for y in range(position[1], position[1] + item.dimensions[1]):
                for z in range(position[2], position[2] + item.dimensions[2]):
                    if not (0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1] and 0 <= z < self.dimensions[2]):
                        return False
                    if self.storage[x, y, z] is not None:
                        return False

        if position[2] == 0:
            return True
        else:
             for x in range(position[0], position[0] + item.dimensions[0]):
                for y in range(position[1], position[1] + item.dimensions[1]):
                    if not (0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1]):
                        continue
                    if self.storage[x,y,position[2]-1] is None:
                        return False
             return True

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
    def __init__(self, id, description, category = None):
        self.racks = []
        self.id = id
        self.description = description
        self.category = category
        self.sub_category = random.choice(list(item_directory[category].keys()))

    def add_rack(self, rack):
        self.racks.append(rack)

class Zone:
    def __init__(self, id, description):
        self.aisles = []
        self.id = id
        self.description = description
        self.category = random.choice(list(item_directory.keys()))

    def add_aisle(self, aisle):
        self.aisles.append(aisle)

class Warehouse:
    def __init__(self, show_vis=False):
        self.zones = []
        self.placement_history = []
        self.show_vis = show_vis
        self.plotter = None
        self.item_meshes = {}


    def add_zone(self, zone):
        self.zones.append(zone)
        if self.show_vis:
            self._add_racks_to_visualization(zone)


    def _add_racks_to_visualization(self, zone):
        if self.plotter is None:
            return

        for aisle in zone.aisles:
            for rack in aisle.racks:
                x, y, z = rack.coordinates
                dx, dy, dz = rack.dimensions

                rack_mesh = pv.Cube(
                    center=(x + dx/2, y + dy/2, z + dz/2),
                    x_length=dx, y_length=dy, z_length=dz
                )
                self.plotter.add_mesh(rack_mesh, color="lightgray", opacity=0.3, show_edges=True)
        self.plotter.render()


    def add_item(self, item):
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
                            if rack.place_item(item, position):
                                self.placement_history.append((item, rack, position))

                                if self.show_vis:
                                    self.add_item_to_visualization(item, rack, position)

                                return rack, position, zone, aisle
        return None, None, None, None

    def undo_item(self):
        if not self.placement_history:
            print("No items to undo.")
            return None

        item, rack, position = self.placement_history.pop()
        removed_item = rack.remove_item(position)

        if self.show_vis and removed_item is not None:
            self.remove_item_from_visualization(item.id)

        return removed_item


    def add_item_to_visualization(self, item, rack, position):
        if self.plotter is None:
            return

        x, y, z = rack.coordinates
        i, j, k = position
        item_dx, item_dy, item_dz = item.dimensions
        item_color = item_directory[item.category][item.sub_category]

        item_mesh = pv.Cube(
            center=(
                x + i + item_dx / 2,
                y + j + item_dy / 2,
                z + k + item_dz / 2
            ),
            x_length=item_dx,
            y_length=item_dy,
            z_length=item_dz
        )

        if item.id in self.item_meshes:
                self.plotter.remove_actor(self.item_meshes[item.id], render=False)

        self.item_meshes[item.id] = self.plotter.add_mesh(
            item_mesh,
            color=item_color,
            opacity=1.0
        )
        self.plotter.render()

    def remove_item_from_visualization(self, item_id):
        if self.plotter is None:
           return
        if item_id in self.item_meshes:
            self.plotter.remove_actor(self.item_meshes[item_id])
            del self.item_meshes[item_id]
            self.plotter.render()

    def visualize(self):
        pass

    def start_visualization(self):
        if self.show_vis:
            if self.plotter is not None:
                self.plotter.close()
                self.plotter.deep_clean()
                del self.plotter

            self.plotter = pv.Plotter()
            self.plotter.set_background('white')
            for zone in self.zones:
                self._add_racks_to_visualization(zone)
            self.plotter.show(auto_close=False, interactive_update=True)


    def show_final_state(self):
        if self.plotter:
            self.plotter.close()
            self.plotter.deep_clean()
            del self.plotter
            self.plotter = None
            time.sleep(0.1)


        final_plotter = pv.Plotter()
        final_plotter.set_background((220/255, 220/255, 220/255))

        for zone in self.zones:
            for aisle in zone.aisles:
                for rack in aisle.racks:
                    x, y, z = rack.coordinates
                    dx, dy, dz = rack.dimensions
                    rack_mesh = pv.Cube(
                        center=(x + dx / 2, y + dy / 2, z + dz / 2),
                        x_length=dx, y_length=dy, z_length=dz
                    )
                    final_plotter.add_mesh(rack_mesh, color="lightgray", opacity=0.3, show_edges=True)

                    for i in range(rack.dimensions[0]):
                        for j in range(rack.dimensions[1]):
                            for k in range(rack.dimensions[2]):
                                item = rack.storage[i, j, k]
                                if item is not None:
                                    item_dx, item_dy, item_dz = item.dimensions
                                    item_color = item_directory[item.category][item.sub_category]
                                    item_mesh = pv.Cube(
                                        center=(
                                            x + i + item_dx / 2,
                                            y + j + item_dy / 2,
                                            z + k + item_dz / 2
                                        ),
                                        x_length=item_dx,
                                        y_length=item_dy,
                                        z_length=item_dz
                                    )
                                    final_plotter.add_mesh(item_mesh, color=item_color, opacity=1.0)
        final_plotter.show()

def build_sample_warehouse(num_zones, num_aisles, num_racks, rack_dimensions, rack_spacing, show_vis=False):
    warehouse = Warehouse(show_vis=show_vis)
    for z_idx in range(num_zones):
        zone = Zone(id=f"Z{z_idx}", description=f"Zone {z_idx}")
        zoneCategory = zone.category
        for a_idx in range(num_aisles):
            aisle = Aisle(id=f"A{a_idx}", description=f"Aisle {a_idx} in Zone {z_idx}",category=zoneCategory)
            for r_idx in range(num_racks):
                rack_coords = (a_idx * (rack_dimensions[0] + rack_spacing[0]),
                               z_idx * (rack_dimensions[1] + rack_spacing[1]),
                               r_idx * (rack_dimensions[2] + rack_spacing[2]))
                rack = Rack(dimensions=rack_dimensions, coordinates=rack_coords, category=zoneCategory, subCategory=aisle.sub_category)
                aisle.add_rack(rack)
            zone.add_aisle(aisle)
        warehouse.add_zone(zone)
    return warehouse


def populate_warehouse(warehouse, num_items):
    added_items = []
    for i in range(num_items):
        category = random.choice(list(item_directory.keys()))
        sub_category = random.choice(list(item_directory[category].keys()))
        weight = random.uniform(0.1, 5.0)
        dimensions = (random.randint(1,3), random.randint(1,2), random.randint(1,2))
        product_name = f"Item_{i}"
        retrieval_urgency = random.randint(1, 5)
        item = Item(category, sub_category, weight, dimensions, product_name, retrieval_urgency)
        rack, position, zone, aisle = warehouse.add_item(item)
        if rack is None:
            print(f"No space found for {product_name}")
        else:
            added_items.append(item)
            if warehouse.show_vis:
                time.sleep(0.2)
    return added_items

warehouse = build_sample_warehouse(
    num_zones=2,
    num_aisles=3,
    num_racks=2,
    rack_dimensions=(5, 4, 6),
    rack_spacing=(2, 2, 0.5),
    show_vis=True
)

warehouse.start_visualization()

populate_warehouse(warehouse, 20)

removed_item = warehouse.undo_item()
if removed_item:
    print(f"Removed {removed_item.product_name}")

populate_warehouse(warehouse, 60)

warehouse.show_final_state()
