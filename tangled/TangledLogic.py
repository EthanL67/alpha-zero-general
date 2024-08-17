import numpy as np
from .qubobrute.core import *
from .qubobrute.simulated_annealing import simulate_annealing_gpu
from pyqubo import Spin


'''
pieces array format:
e00 e01 e02 v0
e10 e11 e12 v1
e20 e21 e22 v2

Layer 2 is identical to the above, but tracks whether the space is occupied.
'''


class Board():
    def __init__(self, v, edges, adj_matrix, aut):
        """
        Set up initial board configuration.
        """
        self.v = v
        self.edges = edges
        self.e = len(edges)
        self.adj_matrix = adj_matrix
        self.aut = aut

        self.pieces = np.zeros((2 * self.v, self.v), dtype="int32")
        self.pieces[self.v:, :] = self.adj_matrix
        for i in range(self.v):
            self.pieces[self.v + i, i] = 1

    def get_legal_moves(self, player):
        """
        Returns all the legal moves
        @param color not used and came from previous version.
        """
        v_pieces = np.diag(self.pieces[:self.v, :])
        v_spaces = np.diag(self.pieces[self.v:, :])
        e_spaces = np.copy(self.pieces[self.v:, :])
        np.fill_diagonal(e_spaces, 0)

        legal_moves = np.zeros(3 * self.e + self.v, dtype="bool")

        # If all edges except one have been filled, and the player has not selected a vertex, we must select a vertex
        if np.sum(e_spaces) > 2 or player in v_pieces:
            # Check open edges
            for i in range(self.e):
                edge_index = self.edges[i]
                if e_spaces[edge_index[0], edge_index[1]] == 1:
                    legal_moves[3*i:3*i+3] = True

        # If the player has already selected a vertex, they may not select another
        if player not in v_pieces:
            # Check open vertices
            for i in range(self.v):
                if v_spaces[i] == 1:
                    legal_moves[3 * self.e + i] = True

        return legal_moves

    def has_legal_moves(self):
        return np.sum(self.pieces[self.v:, :]) != self.v - 2

    def execute_move(self, action, player=1):
        """
        Perform the given move on the board
        e0(-1) e0(0) e0(+1) e1(-1) ... en(+1) v0 ... vm
        """
        # print("\n")
        # print("Player: ", player)
        # print("Board: ")
        # print(self.pieces[:self.v, :])
        # print("Spaces: ")
        # print(self.pieces[self.v:, :])
        # print("Legal moves: ", self.get_legal_moves(player))
        # print("Action: ", action)

        # assert self.get_legal_moves(player)[action] == True

        # edge is played
        if action < self.e * 3:
            idx = int(action / 3)
            color = (action % 3) - 1
            # print("edge action idx: ", idx, ", color: ", color)self.board_data.

            edge_index = self.edges[idx]

            self.pieces[edge_index[0], edge_index[1]] = color
            self.pieces[edge_index[1], edge_index[0]] = color
            self.pieces[self.v + edge_index[0], edge_index[1]] = 0
            self.pieces[self.v + edge_index[1], edge_index[0]] = 0

        # vertex is played
        else:
            node = action - 3 * self.e
            # print("vertex action: ", action)
            self.pieces[node, node] = player
            self.pieces[self.v + node, node] = 0

def calculateScore(pieces, v):

    def qubo_energy(qubo: np.ndarray, offset: np.number, sample: np.ndarray) -> np.number:
        """Calculate the energy of a sample."""
        return np.dot(sample, np.dot(qubo, sample)) + offset

    J = np.copy(pieces[:v, :])

    # Fill the diagonal with 0
    np.fill_diagonal(J, 0)
    vertices = np.diag(pieces[:v, :])

    if np.all(J == 0):
        return 0

    # Define binary variables
    spins = [Spin(f'spin_{i}') for i in range(v)]

    # Construct the Hamiltonian
    H = 0.5 * np.sum(J * np.outer(spins, spins))

    # Compile the model to a binary quadratic model (BQM)
    model = H.compile()
    qubo, offset = model.to_qubo(index_label=True)

    if len(qubo) == 0:
        return 0

    # Initialize the 2D NumPy array with zeros
    q = np.zeros((v, v))

    # Fill the array with the values from the dictionary
    for index, value in qubo.items():
        q[index] = value

    if v < 24:
        # brute-force
        energies = solve_gpu(q, offset)

        # Find the minimum energy
        min_energy = energies.min()

        # Find all indices with the minimum energy
        min_indices = np.where(energies == min_energy)[0]

        # Create a set to store unique solutions
        unique_solutions = set()

        for index in min_indices:
            # Get the solution bits for the current index
            solution = bits(index, nbits=v)

            # Convert the solution to a tuple to make it hashable
            solution_tuple = tuple(solution)

            # Check if the solution is unique
            if solution_tuple not in unique_solutions:
                unique_solutions.add(solution_tuple)

    else:
        energies, solutions = simulate_annealing_gpu(q, offset, n_iter=1000, n_samples=10000, temperature=1.0,
                                                     cooling_rate=0.99)

        # Find the minimum energy
        min_energy = energies.min()

        # Find all indices with the minimum energy
        min_indices = np.where(energies == min_energy)[0]

        # Create a set to store unique solutions
        unique_solutions = set()

        for index in min_indices:
            # Convert the solution to a tuple to make it hashable
            solution_tuple = tuple(solutions[index])
            if solution_tuple not in unique_solutions:
                unique_solutions.add(solution_tuple)



    # assign an equal probability of finding each of the ground states
    prob = 1 / len(unique_solutions)

    # Convert the list of lists to a 2D NumPy array
    unique_solutions_np = np.array([list(tup) for tup in unique_solutions])

    C = np.corrcoef(unique_solutions_np, rowvar=False)

    scores = np.sum(C, axis=1) - 1
    score = np.dot(scores, vertices)

    return score
