import networkx as nx
import numpy as np


class TangledVariant:
    def __init__(self, v, edges, adj_matrix):
        self.v = v
        self.edges = edges
        self.e = len(edges)
        self.adj_matrix = adj_matrix


def create_k3_graph():
    G = nx.complete_graph(3)
    v = int(nx.number_of_nodes(G))
    edges = np.array(G.edges, dtype=np.int32)
    adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
    return v, edges, adj_matrix


def create_k4_graph():
    G = nx.complete_graph(4)
    v = int(nx.number_of_nodes(G))
    edges = np.array(G.edges, dtype=np.int32)
    adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
    return v, edges, adj_matrix


def create_c4_graph():
    G = nx.cycle_graph(4)
    v = int(nx.number_of_nodes(G))
    edges = np.array(G.edges, dtype=np.int32)
    adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
    return v, edges, adj_matrix


def create_petersen_graph():
    G = nx.petersen_graph()
    v = int(nx.number_of_nodes(G))
    edges = np.array(G.edges, dtype=np.int32)
    adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
    return v, edges, adj_matrix


def create_q3_graph():
    G = nx.hypercube_graph(3)
    G = nx.convert_node_labels_to_integers(G)
    v = int(nx.number_of_nodes(G))
    edges = np.array(G.edges, dtype=np.int32)
    adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
    return v, edges, adj_matrix


def create_q4_graph():
    G = nx.hypercube_graph(4)
    G = nx.convert_node_labels_to_integers(G)
    v = int(nx.number_of_nodes(G))
    edges = np.array(G.edges, dtype=np.int32)
    adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
    return v, edges, adj_matrix


def create_q5_graph():
    G = nx.hypercube_graph(5)
    G = nx.convert_node_labels_to_integers(G)
    v = int(nx.number_of_nodes(G))
    edges = np.array(G.edges, dtype=np.int32)
    adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
    return v, edges, adj_matrix
