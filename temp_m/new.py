import numpy as np
import pyvista as pv

class Warehouse3D:
    def __init__(self, width, height, depth, num_zones, aisles_per_zone, racks_per_aisle, slots_per_rack):
        self.width = width
        self.height = height
        self.depth = depth
        self.num_zones = num_zones
        self.aisles_per_zone = aisles_per_zone
        self.racks_per_aisle = racks_per_aisle
        self.slots_per_rack = slots_per_rack
        self.matrix = np.full((width, height, depth), None)
        self._construct_warehouse()

    def _construct_warehouse(self):
        z_start = 0
        for zone_id in range(1, self.num_zones + 1):
            z_start += 5 if zone_id > 1 else 0 
            a_start = z_start
            for aisle_id in range(1, self.aisles_per_zone + 1):
                a_start += 2 if aisle_id > 1 else 0 

                for rack_id in range(1, self.racks_per_aisle + 1):
                    for level in range(self.height):
                        for slot_id in range(1, self.slots_per_rack + 1):
                            self.matrix[rack_id, level, a_start + slot_id] = {
                                "Zone": zone_id,
                                "Aisle": aisle_id,
                                "Rack": rack_id,
                                "Slot": slot_id
                            }

    def get_cube_metadata(self, x, y, z):
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            return self.matrix[x, y, z]
        return None 

    def visualize_warehouse(self):
        occupied_cells = []
        colors = []

       
        color_map = {
            "Zone": [255, 0, 0],   
            "Aisle": [0, 255, 0],  
            "Rack": [0, 0, 255],   
            "Slot": [255, 255, 0]  
        }

       
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    cube = self.matrix[x, y, z]
                    if cube:
                        occupied_cells.append([x, y, z])
                        colors.append(color_map["Slot"]) 

       
        occupied_cells = np.array(occupied_cells)
        colors = np.array(colors) / 255.0 
        grid = pv.PolyData(occupied_cells)
        grid["colors"] = colors
        voxels = grid.delaunay_3d()
        voxels["colors"] = grid["colors"]
        plotter = pv.Plotter()
        plotter.add_mesh(voxels, scalars="colors", rgb=True, opacity=0.8, show_edges=True)
        plotter.show()

warehouse = Warehouse3D(
    width=20, height=5, depth=30, 
    num_zones=3, aisles_per_zone=4, racks_per_aisle=5, slots_per_rack=3 
)

warehouse.visualize_warehouse()
