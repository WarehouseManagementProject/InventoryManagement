# Warehouse SLIM Optimization Project

## Summary of the Project Idea

The Warehouse SLIM Optimization project aims to determine the most suitable configuration for arranging items within warehouses, optimizing space, layout, inventory, and movement.

## Data Structure

- **Warehouse Class**
  - A Python class named `Warehouse` containing an array of Zone objects.

- **Zone**
  - Represents a dedicated category of items.
  - Implemented as an array of Aisle objects.

- **Aisle**
  - Represents a subcategory of items within a category.
  - Composed of an array of Rack objects.

- **Rack**
  - Represents the granular storage level.
  - Modeled as a 3D matrix representing available space.
  - Each random index in the 3D matrix may hold an Item object.

- **Item Object**
  - Contains attributes such as:
    - Category and sub-category
    - Weight and dimensions
    - Product name
    - Retrieval urgency score

- **Metadata Captured at Each Hierarchy Level**
  - Unique identifiers
  - Coordinates (x, y, z)
  - Dimensions (height, width, depth)
  - Descriptions

## Optimization

- **Virtual Entities**
  - The Warehouse, Zone, and Aisle are virtual entities and do not hold storage information like a Rack.
  
- **Storage Constraints**
  - Items are stored at the Rack level.
  - An item cannot occupy two racks simultaneously in a static layout.

- **Randomized Placement**
  - A randomization function will select a random Zone, Aisle, and Rack to place an item if space is available.
  - This setup allows for various local search techniques to be applied in order to find the optimal configuration.

## Simulating the Environment

- **Visualization**
  - The current idea is to use Python package such as [PyVista](https://www.pyvista.org/) to render 3D numpy matrices.
  
- **3D Representation**
  - The 3D coordinates and dimensions of entities are used to generate cuboids.
  - When an item is placed in a Rack, the corresponding cell in the 3D matrix is updated to display a color based on the item’s sub-category.

- **Animation**
  - The animation refresh rate can be set between 30 to 60 fps.
  - By skipping multiple frames, the simulation loop (which may run up to 10^6 iterations) remains efficient.

- **Simulation Metadata**
  - For simulation, only the metadata (specifically 3D coordinates, dimensions, and color) of the entities is required.
  - The color is determined based on the entity type and the item sub-category.


## Key Objectives

- Build Python classes for the warehouse hierarchy (Zones, Aisles, Racks, Items) with essential metadata.
- Organize the warehouse as Zones → Aisles → Racks, with only Racks storing items.
- Develop a randomized function to validate and assign items to available racks.
- Apply local search techniques and cost functions to optimize SLIM.
- Use a 3D visualization library (e.g., PyVista) to render entities and animate changes in real time.
- Measure performance based on space utilization, retrival efficiency, and layout effectiveness. One Advantage with PyVista is it's interactive. So a high level visual evaluation can be performed.
- Ensure scalability and adaptability to handle growing warehouses and new constraints.


## Algorithms & Models (In-Progress)
- Genetic Algorithms (GA) with Multi-Parent Crossover Operators
- Iterated Local Search & Simulated Annealing
- _If time permits, we plan on exploring Reinforcement Learning_

## Timeline & Roadmap

## Timeline & Roadmap (March 1st - April 5th)

| **Dates** | **Tasks** |
|-----------|-----------|
| **March 1 - 7** | Finalize project requirements and objectives. <br> Define data structure and metadata for warehouse hierarchy. <br> Research and finalize algorithms (GA, Local Search, Simulated Annealing). |
| **March 8 - 14** | Implement Python classes for the warehouse hierarchy. <br> Develop random placement algorithm for item storage validation. <br> Set up basic simulation environment using PyVista or equivalent. |
| **March 15 - 21** | Code Genetic Algorithm with multi-parent crossover. <br> Implement Iterated Local Search and Simulated Annealing. <br> Integrate cost/score functions for evaluating configurations. |
| **March 22 - 28** | Test algorithms on small-scale warehouse simulations. <br> Debug and refine optimization logic for accuracy and performance. <br> Validate random placement and storage constraints. |
| **March 29 - April 4** | Run full-scale simulations with varying configurations. <br> Analyze results to identify the best-performing layouts. <br> Fine-tune algorithms based on simulation feedback. |
| **April 5** | Prepare a summary of findings and performance metrics. <br> Document the project workflow, results, and potential improvements. |
