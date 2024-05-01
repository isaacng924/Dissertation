import heapq
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize global variables
G = None
pos = None
fig = None
ax = None

def dijkstra(graph, start):
    """
    Implementation of Dijkstra's algorithm to find the shortest paths from a single source to all other nodes in a graph.

    Parameters:
    - graph (dict): The graph represented as a dictionary of dictionaries.
    - start (str): The starting node from which to find the shortest paths.

    Returns:
    - distances (dict): A dictionary containing the shortest distances from the start node to all other nodes.
    - predecessors (dict): A dictionary containing the predecessors of each node in the shortest paths.
    """
    distances = {node: float('infinity') for node in graph}
    predecessors = {node: None for node in graph}
    distances[start] = 0

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors

def apply_weight_decay(graph, decay_factor):
    """
    Applies weight decay to the edges of the graph based on the specified decay factor.

    Parameters:
    - graph (dict): The graph represented as a dictionary of dictionaries.
    - decay_factor (float): The decay factor to apply to the edge weights.

    Returns:
    None
    """
    item_list = [] 
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            item_list.append(graph[node][neighbor])
    average_time = sum(item_list)/len(item_list)
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            if graph[node][neighbor] > average_time:
                graph[node][neighbor] *= (1 - (decay_factor * 5))
            else:
                graph[node][neighbor] *= (1 - (decay_factor))
            graph[node][neighbor] -= (graph[node][neighbor])*(decay_factor)
    print("average time = ",average_time, "s")

def shortest_path(graph, start, end):
    """
    Finds the shortest path from the start node to the end node in the given graph.

    Parameters:
    - graph (dict): The graph represented as a dictionary of dictionaries.
    - start (str): The starting node of the path.
    - end (str): The ending node of the path.

    Returns:
    - path (list): A list representing the shortest path from start to end.
    - shortest_distance (float): The length of the shortest path.
    """
    distances, predecessors = dijkstra(graph, start)

    path = []
    current_node = end
    while current_node is not None:
        path.insert(0, current_node)
        current_node = predecessors[current_node]

    return path, distances[end]

def adjust_graph_for_beginner(graph, scaling_factor):
    """
    Adjusts the weights of edges in the graph to accommodate beginner skill level.

    Parameters:
    - graph (dict): The graph represented as a dictionary of dictionaries.
    - scaling_factor (float): The scaling factor to adjust edge weights.

    Returns:
    - adjusted_graph (dict): The adjusted graph with modified edge weights.
    """
    adjusted_graph = graph.copy()
    item_list = []  

    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            item_list.append(adjusted_graph[node][neighbor])

    average_time = sum(item_list)/len(item_list)
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            if adjusted_graph[node][neighbor] > average_time:
                adjusted_graph[node][neighbor] *= (scaling_factor * 1.25)
            else:
                adjusted_graph[node][neighbor] *= scaling_factor
    return adjusted_graph

def adjust_graph_for_intermediate(graph, scaling_factor):
    """
    Adjusts the weights of edges in the graph to accommodate intermediate skill level.

    Parameters:
    - graph (dict): The graph represented as a dictionary of dictionaries.
    - scaling_factor (float): The scaling factor to adjust edge weights.

    Returns:
    - adjusted_graph (dict): The adjusted graph with modified edge weights.
    """
    adjusted_graph = graph.copy()
    item_list = []  

    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            item_list.append(adjusted_graph[node][neighbor])

    average_time = sum(item_list)/len(item_list)
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            if adjusted_graph[node][neighbor] > average_time:
                adjusted_graph[node][neighbor] *= (scaling_factor * 1.1)
            else:
                adjusted_graph[node][neighbor] *= scaling_factor
    return adjusted_graph

def adjust_graph_for_expert(graph, scaling_factor):
    """
    Adjusts the weights of edges in the graph to accommodate expert skill level.

    Parameters:
    - graph (dict): The graph represented as a dictionary of dictionaries.
    - scaling_factor (float): The scaling factor to adjust edge weights.

    Returns:
    - adjusted_graph (dict): The adjusted graph with modified edge weights.
    """
    adjusted_graph = graph.copy()
    item_list = []  

    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            item_list.append(adjusted_graph[node][neighbor])

    average_time = sum(item_list)/len(item_list)
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            if adjusted_graph[node][neighbor] > average_time:
                adjusted_graph[node][neighbor] *= (scaling_factor * 1.05)
            else:
                adjusted_graph[node][neighbor] *= scaling_factor
    return adjusted_graph

def plot_path_changes(graph, start_node, end_node, iteration, scaling_factor, decay_factor_range):
    """
    Plots the effect of decay factor on the length of the shortest path over multiple iterations.

    Parameters:
    - graph (dict): The graph represented as a dictionary of dictionaries.
    - start_node (str): The starting node of the path.
    - end_node (str): The ending node of the path.
    - iteration (int): The number of iterations to perform.
    - scaling_factor (float): The scaling factor for adjusting edge weights.
    - decay_factor_range (list): The range of decay factors to test.

    Returns:
    None
    """
    paths = {decay_factor: [] for decay_factor in decay_factor_range}

    for decay_factor in decay_factor_range:
        graph_copy = adjust_graph_for_beginner(graph, scaling_factor)
        for i in range(iteration):
            path, _ = shortest_path(graph_copy, start_node, end_node)
            paths[decay_factor].append(len(path))
            apply_weight_decay(graph_copy, decay_factor)

    # Plotting
    plt.figure(figsize=(10, 6))
    for decay_factor, path_lengths in paths.items():
        plt.plot(range(1, iteration + 1), path_lengths, label=f'Decay Factor: {decay_factor}')
    plt.xlabel('Iteration')
    plt.ylabel('Path Length')
    plt.title('Effect of Decay Factor on Path Length')
    plt.legend()
    plt.grid(True)
    plt.yticks(np.arange(3, max(max(lengths) for lengths in paths.values()) + 1, 1))
    plt.show()

def plot_time_scaling_changes(graph, start_node, end_node, iteration, time_scaling_range, decay_factor):
    """
    Plots the effect of time scaling factor on the length of the shortest path over multiple iterations.

    Parameters:
    - graph (dict): The graph represented as a dictionary of dictionaries.
    - start_node (str): The starting node of the path.
    - end_node (str): The ending node of the path.
    - iteration (int): The number of iterations to perform.
    - time_scaling_range (list): The range of time scaling factors to test.
    - decay_factor (float): The decay factor for applying weight decay to edge weights.

    Returns:
    None
    """
    paths = {time_scaling: [] for time_scaling in time_scaling_range}

    for time_scaling in time_scaling_range:
        graph_copy = adjust_graph_for_beginner(graph, time_scaling)
        for i in range(iteration):
            path, _ = shortest_path(graph_copy, start_node, end_node)
            paths[time_scaling].append(len(path))
            apply_weight_decay(graph_copy, decay_factor)

    # Plotting
    plt.figure(figsize=(10, 6))
    for time_scaling, path_lengths in paths.items():
        plt.plot(range(1, iteration + 1), path_lengths, label=f'Time Scaling: {time_scaling}')
    plt.xlabel('Iteration')
    plt.ylabel('Path Length')
    plt.title('Effect of Time Scaling Factor on Path Length')
    plt.legend()
    plt.grid(True)
    plt.yticks(np.arange(3, max(max(lengths) for lengths in paths.values()) + 1, 1))
    plt.show()

mario_directed_graph = {
    '1-0': {'1-1': 0}, '1-1': {'1-2': 38}, '1-2': {'1-3': 42}, '1-3': {'1-T': 39},
    '1-T': {'1-4': 57}, '1-4': {'1-A': 71, '1-5': 71}, '1-5': {'1-Cas': 57},
    '1-Cas': {'2-0': 55}, '1-A': {'1-Cas': 158},
    '2-0': {'2-1': 0}, '2-1': {'2-2': 35}, '2-2': {'2-3': 57}, '2-3': {'2-A': 59},
    '2-A': {'2-Cas': 36}, '2-4': {'2-T': 61, '2-6': 19}, '2-5': {'2-T': 55, '2-6': 65},
    '2-6': {'2-Cas': 158}, '2-Cas': {'3-0': 105},
    '3-0': {'3-1': 0, '3-A': 0}, '3-1': {'3-2': 58}, '3-A': {'3-2': 65}, 
    '3-2': {'3-T': 47}, '3-T': {'3-3': 60}, '3-3': {'3-GH': 76}, '3-GH': {'3-Cas': 52},
    '3-Cas': {'4-0': 118},
    '4-0': {'4-1': 0}, '4-1': {'4-2': 30}, '4-2': {'4-3': 52}, '4-3': {'4-T': 66}, 
    '4-T': {'4-A': 55, '4-4': 55}, '4-4': {'4-GH': 43}, '4-A': {'4-GH': 44},
    '4-GH': {'4-5': 48}, '4-5': {'4-6': 76}, '4-6': {'4-Cas': 93},
    '4-Cas': {'5-0': 88},
    '5-0': {'5-1': 0, '5-A': 0}, '5-1': {'5-2': 41}, '5-2': {'5-3': 41, '5-T': 39}, '5-A': {'5-T': 42},
    '5-3': {'5-GH': 38}, '5-C': {'5-4': 50}, '5-GH': {'5-4': 70},
    '5-4': {'5-Cas': 88}, '5-T': {'5-3': 94, '5-4': 70}, '5-Cas': {'6-0': 132},
    '6-0': {'6-1': 0, '6-A': 0}, '6-1': {'6-2': 39}, '6-A': {'6-2': 68}, 
    '6-2': {'6-T1': 92}, '6-T1': {'6-3': 68}, '6-3': {'6-4': 39}, '6-4': {'6-T2': 63}, 
    '6-T2': {'6-5': 62, '6-B': 62}, '6-5': {'6-6': 83}, '6-B': {'6-6': 86}, 
    '6-6': {'6-Cas': 90}, '6-Cas': {'7-0': 79},
    '7-0': {'7-1': 0}, '7-1': {'7-GH': 47}, '7-GH': {'7-2': 24}, '7-2': {'7-3': 14}, 
    '7-3': {'7-T': 142}, '7-T': {'7-4': 85}, '7-4': {'7-5': 35}, '7-5': {'7-A': 46, '7-6': 70}, 
    '7-A': {'7-Cas': 52}, '7-6': {'7-7': 36}, '7-7': {'7-Cas': 80}, '7-Cas': {'8-0': 153},
    '8-0': {'8-1': 0}, '8-1': {'8-2': 33}, '8-2': {'8-T1': 62}, '8-T1': {'8-3': 61}, 
    '8-3': {'8-4': 99}, '8-4': {'8-Cas': 53}, '8-Cas': {'8-5': 52}, '8-5': {'8-6': 33}, 
    '8-6': {'8-7': 40}, '8-7': {'8-8': 46}, '8-8': {'8-T2': 84}, '8-T2': {'8-BC': 38}, '8-BC': {},
}

megaman_directed_graph = {
    'A': {'B': 420, 'C': 725},
    'B': {'C': 300, 'D': 1300},
    'C': {'D': 900, 'E': 1300},
    'D': {'E': 360},
    'E': {'F': 360},
    'F': {'A': 480, 'C': 600, 'D': 480}
    }



plot_path_changes(megaman_directed_graph, 'A', 'F', 10, 1.2, [0.01, 0.005, 0.002])
plot_time_scaling_changes(megaman_directed_graph, 'A', 'F', 10, [1.2, 1.0, 0.8], 0.01)






