Speedrunning Optimization Algorithm
Overview
This Python script implements a speedrunning optimization algorithm based on Dijkstra's algorithm. The algorithm finds the shortest path through a directed graph representing a game level, considering various factors such as the player's skill level, edge weights, and weight decay.

Features
Dijkstra's Algorithm Implementation: Utilizes Dijkstra's algorithm to find the shortest path from a specified starting node to an ending node in the directed graph.
Skill-based Path Optimization: Adjusts the weights of edges in the graph based on the user's skill level (beginner, intermediate, or expert) to simulate different player capabilities.
Edge Weight Decay: Applies weight decay to the edges of the graph to simulate time-based factors affecting gameplay, such as deteriorating platform conditions or temporary power-ups.
Visualization: Visualizes the directed graph with optional highlighting of the optimal path.

Dependencies
Python 3.x
Matplotlib: For graph visualization.
NumPy: For numerical computations.
NetworkX: For graph manipulation and algorithms.
Tkinter: For the GUI interface.
