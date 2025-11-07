import os
import torch

def save_graph(graph, path):
    torch.save(graph, path)

def load_graph(path):
    return torch.load(path)

def graph_exists(path):
    return os.path.exists(path)
