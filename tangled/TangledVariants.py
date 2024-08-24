import networkx as nx
import numpy as np
import pickle
import os


class TangledVariant:
    def __init__(self, v, edges, adj_matrix, aut):
        self.v = v
        self.edges = edges
        self.e = len(edges)
        self.adj_matrix = adj_matrix
        self.aut = aut


def create_k3_graph():
    filename = "tangled/graphdata/K3.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if the pickle file exists
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            v, edges, adj_matrix, aut = pickle.load(f)
    else:
        # Create the graph and its attributes
        G = nx.complete_graph(3)
        G = nx.convert_node_labels_to_integers(G)
        v = int(nx.number_of_nodes(G))
        edges = np.array(G.edges, dtype=np.int32)
        adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
        aut = list(nx.algorithms.isomorphism.GraphMatcher(G, G).isomorphisms_iter())

        # Save the values to a pickle file
        with open(filename, 'wb') as f:
            pickle.dump((v, edges, adj_matrix, aut), f)

    return v, edges, adj_matrix, aut


def create_k4_graph():
    filename = "tangled/graphdata/K4.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if the pickle file exists
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            v, edges, adj_matrix, aut = pickle.load(f)
    else:
        # Create the graph and its attributes
        G = nx.complete_graph(4)
        G = nx.convert_node_labels_to_integers(G)
        v = int(nx.number_of_nodes(G))
        edges = np.array(G.edges, dtype=np.int32)
        adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
        aut = list(nx.algorithms.isomorphism.GraphMatcher(G, G).isomorphisms_iter())

        # Save the values to a pickle file
        with open(filename, 'wb') as f:
            pickle.dump((v, edges, adj_matrix, aut), f)

    return v, edges, adj_matrix, aut


def create_petersen_graph():
    filename = "tangled/graphdata/P.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if the pickle file exists
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            v, edges, adj_matrix, aut = pickle.load(f)
    else:
        # Create the graph and its attributes
        G = nx.petersen_graph()
        G = nx.convert_node_labels_to_integers(G)
        v = int(nx.number_of_nodes(G))
        edges = np.array(G.edges, dtype=np.int32)
        adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
        aut = list(nx.algorithms.isomorphism.GraphMatcher(G, G).isomorphisms_iter())

        # Save the values to a pickle file
        with open(filename, 'wb') as f:
            pickle.dump((v, edges, adj_matrix, aut), f)

    return v, edges, adj_matrix, aut


def create_q3_graph():
    filename = "tangled/graphdata/Q3.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if the pickle file exists
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            v, edges, adj_matrix, aut = pickle.load(f)
    else:
        # Create the graph and its attributes
        G = nx.hypercube_graph(3)
        G = nx.convert_node_labels_to_integers(G)
        v = int(nx.number_of_nodes(G))
        edges = np.array(G.edges, dtype=np.int32)
        adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
        aut = list(nx.algorithms.isomorphism.GraphMatcher(G, G).isomorphisms_iter())

        # Save the values to a pickle file
        with open(filename, 'wb') as f:
            pickle.dump((v, edges, adj_matrix, aut), f)

    return v, edges, adj_matrix, aut


def create_q4_graph():
    filename = "tangled/graphdata/Q4.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if the pickle file exists
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            v, edges, adj_matrix, aut = pickle.load(f)
    else:
        # Create the graph and its attributes
        G = nx.hypercube_graph(4)
        G = nx.convert_node_labels_to_integers(G)
        v = int(nx.number_of_nodes(G))
        edges = np.array(G.edges, dtype=np.int32)
        adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
        aut = list(nx.algorithms.isomorphism.GraphMatcher(G, G).isomorphisms_iter())

        # Save the values to a pickle file
        with open(filename, 'wb') as f:
            pickle.dump((v, edges, adj_matrix, aut), f)

    return v, edges, adj_matrix, aut


def create_q5_graph():
    filename = "tangled/graphdata/Q5.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if the pickle file exists
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            v, edges, adj_matrix, aut = pickle.load(f)
    else:
        # Create the graph and its attributes
        G = nx.hypercube_graph(5)
        G = nx.convert_node_labels_to_integers(G)
        v = int(nx.number_of_nodes(G))
        edges = np.array(G.edges, dtype=np.int32)
        adj_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=np.int32)
        aut = list(nx.algorithms.isomorphism.GraphMatcher(G, G).isomorphisms_iter())

        # Save the values to a pickle file
        with open(filename, 'wb') as f:
            pickle.dump((v, edges, adj_matrix, aut), f)

    return v, edges, adj_matrix, aut
