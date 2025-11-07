import os, torch

def save_graph(graph, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(graph, path)

def load_graph(path, map_location=None):
    return torch.load(path, map_location=map_location)

def graph_exists(path):
    return os.path.exists(path)
