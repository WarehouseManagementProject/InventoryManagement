# Warehouse SLIM Optimization

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

### Diagramatic Representation:
![wso-mermaid-diagram](https://github.com/user-attachments/assets/b48a37c8-047e-457b-b3eb-adce784fcdf4)


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


https://github.com/user-attachments/assets/2c660ddd-98c4-4489-ad46-bdf0d5b822dc




## Key Objectives

- Build Python classes for the warehouse hierarchy (Zones, Aisles, Racks, Items) with essential metadata.
- Organize the warehouse as Zones → Aisles → Racks, with only Racks storing items.
- Develop a randomized function to validate and assign items to available racks.
- Apply local search techniques and cost functions to optimize SLIM.
- Use a 3D visualization library (e.g., PyVista) to render entities and animate changes in real time.
- Measure performance based on space utilization, retrival efficiency, and layout effectiveness. One Advantage with PyVista is it's interactive. So a high level visual evaluation can be performed.
- Ensure scalability and adaptability to handle growing warehouses and new constraints.


## Algorithms & Models (In-Progress)
- Genetic Algorithm
- Local Search (Simulated Annealing)
- * _If time permits, we plan on exploring Reinforcement Learning_


## Timeline & Roadmap (March 1st - April 5th)

| **Dates**         | **Tasks**                                                                 |
|--------------------|--------------------------------------------------------------------------|
| **March 1 - 3**    | Finalize requirements, objectives, and algorithms (GA, Local Search).    |
| **March 4 - 7**    | Define data structure and implement Python classes for hierarchy.        |
| **March 8 - 12**   | Develop random placement algorithm and set up simulation environment.    |
| **March 13 - 22**  | Implement Genetic Algorithm / Iterated Local Search & Simulated Annealing.|
| **March 23 - 27**  | Test algorithms.             |
| **March 28 - April 2** | Run full-scale simulations, analyze results, and fine-tune algorithms. |
| **April 3 - April 5** | Prepare final report, document findings, and outline improvements.     |



<img width="786" alt="image" src="https://github.com/user-attachments/assets/fe6a5b9c-9cd8-40f9-bbf0-d40fa47ac37b" />
