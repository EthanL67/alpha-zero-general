from __future__ import print_function
import sys

import numpy as np
import networkx as nx

from Game import Game
from .TangledLogic import Board, calculateScore
from .TangledVariants import *

from numba import njit, prange

sys.path.append('..')


class TangledGame(Game):
    def __init__(self, name):
        self.name = name
        self.getInitBoard()

    def getInitBoard(self):
        # return initial board (numpy board)
        if self.name == "K4":
            v, edges, adj_matrix, aut = create_k4_graph()
        elif self.name == "P":
            v, edges, adj_matrix, aut = create_petersen_graph()
        elif self.name == "Q3":
            v, edges, adj_matrix, aut = create_q3_graph()
        elif self.name == "Q4":
            v, edges, adj_matrix, aut = create_q4_graph()
        elif self.name == "Q5":
            v, edges, adj_matrix, aut = create_q5_graph()
        else:
            v, edges, adj_matrix, aut = create_k3_graph()

        self.board = Board(v, edges, adj_matrix, aut)

        return self.board.pieces

    def getBoardSize(self):
        # (a,b) tuple
        return (2 * self.board.v, self.board.v)

    def getActionSize(self):
        # return number of actions: three colors per edge, or select one of the vertices
        return 3 * self.board.e + self.board.v

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.board.v, self.board.edges, self.board.adj_matrix, self.board.aut)
        b.pieces = np.copy(board)

        b.execute_move(action, player)

        return b.pieces, -player

    def getValidMoves(self, board, player):
        b = Board(self.board.v, self.board.edges, self.board.adj_matrix, self.board.aut)
        b.pieces = np.copy(board)

        return b.get_legal_moves(player)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.board.v, self.board.edges, self.board.adj_matrix, self.board.aut)
        b.pieces = np.copy(board)

        if b.has_legal_moves():
            return 0
        else:
            score = calculateScore(b.pieces, b.v) * player
            if score < 0:
                return 1  # player 1 won
            elif score > 0:
                return -1  # player 1 lost
            else:
                return 1e-4  # draw

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        b = Board(self.board.v, self.board.edges, self.board.adj_matrix, self.board.aut)
        b.pieces = np.copy(board)

        if player == -1:
            for v in range(self.board.v):
                b.pieces[v, v] *= -1

        return b.pieces

    def getSymmetries(self, board, pi):
        v = self.board.v
        e = self.board.e
        edges = self.board.edges

        # Split pi into edge and vertex probabilities
        edge_pi = pi[:3 * e]
        vertex_pi = np.array(pi[3 * e:])

        # Initialize a 3D matrix for edge probabilities
        pi_matrix = np.zeros((3, v, v))

        # Fill the edge probabilities into pi_matrix
        for idx, (x, y) in enumerate(edges):
            pi_matrix[0, x, y] = edge_pi[3 * idx]
            pi_matrix[0, y, x] = edge_pi[3 * idx]
            pi_matrix[1, x, y] = edge_pi[3 * idx + 1]
            pi_matrix[1, y, x] = edge_pi[3 * idx + 1]
            pi_matrix[2, x, y] = edge_pi[3 * idx + 2]
            pi_matrix[2, y, x] = edge_pi[3 * idx + 2]

        # Create permutation matrices and parallelize the process
        perm_matrices = []
        for aut in self.board.aut:
            perm_matrix = np.zeros((v, v), dtype=np.float32)
            for i in range(v):
                perm_matrix[i, aut[i]] = 1
            perm_matrices.append(perm_matrix)

        syms = [None] * len(self.board.aut)

        # Apply the permutation in parallel
        syms = calculate_syms(perm_matrices, board, pi_matrix, edges, vertex_pi, v, e)

        return syms

    def stringRepresentation(self, board):
        return board.tobytes()

    @staticmethod
    def display(board):
        print("Board: ")
        print(board)


@njit()
def calculate_syms(perm_matrices, board, pi_matrix, edges, vertex_pi, v, e):

    syms = []
    board = np.asarray(board, dtype=np.float32)

    for i in prange(len(perm_matrices)):
        # Initialize matrices for symmetric probabilities
        sym_edgen1_pi_matrix = np.zeros((v, v), dtype=np.float32)
        sym_edge0_pi_matrix = np.zeros((v, v), dtype=np.float32)
        sym_edgep1_pi_matrix = np.zeros((v, v), dtype=np.float32)

        for idx, (x, y) in enumerate(edges):
            sym_edgen1_pi_matrix[x, y] = pi_matrix[0, x, y]
            sym_edge0_pi_matrix[x, y] = pi_matrix[1, x, y]
            sym_edgep1_pi_matrix[x, y] = pi_matrix[2, x, y]

        # Permute the board
        sym_board1 = perm_matrices[i] @ board[:v, :] @ perm_matrices[i].T
        sym_board2 = perm_matrices[i] @ board[v:, :] @ perm_matrices[i].T
        sym_board = np.vstack((sym_board1, sym_board2))

        # Permute edge probabilities
        sym_edgen1_pi_matrix = perm_matrices[i] @ sym_edgen1_pi_matrix @ perm_matrices[i].T
        sym_edge0_pi_matrix = perm_matrices[i] @ sym_edge0_pi_matrix @ perm_matrices[i].T
        sym_edgep1_pi_matrix = perm_matrices[i] @ sym_edgep1_pi_matrix @ perm_matrices[i].T

        # Permute vertex probabilities
        sym_vertex_pi = vertex_pi[np.arange(v)]

        # Gather symmetric edge probabilities
        sym_edgen1_pi = np.zeros(e, dtype=np.float32)
        sym_edge0_pi = np.zeros(e, dtype=np.float32)
        sym_edgep1_pi = np.zeros(e, dtype=np.float32)

        for idx, (x, y) in enumerate(edges):
            sym_edgen1_pi[idx] = sym_edgen1_pi_matrix[x, y]
            sym_edge0_pi[idx] = sym_edge0_pi_matrix[x, y]
            sym_edgep1_pi[idx] = sym_edgep1_pi_matrix[x, y]

        # Combine symmetric edge and vertex probabilities
        sym_edge_pi = np.zeros(3 * e, dtype=np.float32)
        sym_edge_pi[0::3] = sym_edgen1_pi
        sym_edge_pi[1::3] = sym_edge0_pi
        sym_edge_pi[2::3] = sym_edgep1_pi
        sym_pi = np.hstack((sym_edge_pi, sym_vertex_pi))

        syms.append((sym_board, sym_pi))

    return syms


# from __future__ import print_function
# import sys
# import numpy as np
# import networkx as nx
#
# from Game import Game
# from .TangledLogic import Board, calculateScore
# from .TangledVariants import *
#
# sys.path.append('..')
#
#
# class TangledGame(Game):
#     def __init__(self, name):
#         self.name = name
#
#         if self.name == "K4":
#             v, edges, adj_matrix, aut = create_k4_graph()
#         elif self.name == "C4":
#             v, edges, adj_matrix, aut = create_c4_graph()
#         elif self.name == "Petersen":
#             v, edges, adj_matrix, aut = create_petersen_graph()
#         elif self.name == "Q3":
#             v, edges, adj_matrix, aut = create_q3_graph()
#         elif self.name == "Q4":
#             v, edges, adj_matrix, aut = create_q4_graph()
#         elif self.name == "Q5":
#             v, edges, adj_matrix, aut = create_q5_graph()
#         else:
#             v, edges, adj_matrix, aut = create_k3_graph()
#
#         self.board = Board(v, edges, adj_matrix, aut)
#
#     def getInitBoard(self):
#         # return initial board (numpy board)
#         self.board = Board(self.board.v, self.board.edges, self.board.adj_matrix, self.board.aut)
#         return self.board.pieces
#
#     def getBoardSize(self):
#         # (a,b) tuple
#         return (2*self.board.v, self.board.v)
#
#     def getActionSize(self):
#         # return number of actions: three colors per edge, or select one of the vertices
#         return 3 * self.board.e + self.board.v
#
#     def getNextState(self, board, player, action):
#         # if player takes action on board, return next (board,player)
#         # action must be a valid move
#         b = Board(self.board.v, self.board.edges, self.board.adj_matrix, self.board.aut)
#         b.pieces = np.copy(board)
#         # b = self.board
#         # b.pieces = np.copy(board)
#
#         b.execute_move(action, player)
#
#         return b.pieces, -player
#
#     def getValidMoves(self, board, player):
#         b = Board(self.board.v, self.board.edges, self.board.adj_matrix, self.board.aut)
#         b.pieces = np.copy(board)
#         # b = self.board
#         # b.pieces = np.copy(board)
#
#         return b.get_legal_moves(player)
#
#     def getGameEnded(self, board, player):
#         # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
#         # player = 1
#         b = Board(self.board.v, self.board.edges, self.board.adj_matrix, self.board.aut)
#         b.pieces = np.copy(board)
#         # b = self.board
#         # b.pieces = np.copy(board)
#
#         if b.has_legal_moves():
#             return 0
#         else:
#             score = calculateScore(b.pieces, b.v)
#             if score > 0:
#                 return 1  # player 1 won
#             elif score < 0:
#                 return -1  # player 1 lost
#             else:
#                 return 1e-4  # draw
#
#     def getCanonicalForm(self, board, player):
#         # return state if player==1, else return -state if player==-1
#         b = Board(self.board.v, self.board.edges, self.board.adj_matrix, self.board.aut)
#         b.pieces = np.copy(board)
#         # b = self.board
#         # b.pieces = np.copy(board)
#
#         if player == -1:
#             for v in range(self.board.v):
#                 b.pieces[v, v] = -b.pieces[v, v]
#
#         return b.pieces
#
#     def getSymmetries(self, board, pi):
#         syms = []
#         n = self.board.v
#
#         # Split pi into edge and vertex probabilities
#         edge_pi = pi[:3 * self.board.e]
#         vertex_pi = np.array(pi[3 * self.board.e:])  # Convert to NumPy array for advanced indexing
#
#         # Initialize a 3D matrix for edge probabilities
#         pi_matrix = np.zeros((3, n, n))
#
#         # Fill the edge probabilities into pi_matrix
#         for idx, (x, y) in enumerate(self.board.edges):
#             pi_matrix[0, x, y] = edge_pi[3 * idx]
#             pi_matrix[0, y, x] = edge_pi[3 * idx]
#             pi_matrix[1, x, y] = edge_pi[3 * idx + 1]
#             pi_matrix[1, y, x] = edge_pi[3 * idx + 1]
#             pi_matrix[2, x, y] = edge_pi[3 * idx + 2]
#             pi_matrix[2, y, x] = edge_pi[3 * idx + 2]
#
#         for aut in self.board.aut:
#             # Create a permutation matrix based on the automorphism
#             perm_matrix = np.zeros((n, n))
#             for i in range(n):
#                 perm_matrix[i, aut[i]] = 1
#
#             # Apply the permutation to the board
#             sym_board1 = perm_matrix @ board[:self.board.v, :] @ perm_matrix.T
#             sym_board2 = perm_matrix @ board[self.board.v:, :] @ perm_matrix.T
#             sym_board = np.vstack((sym_board1, sym_board2))
#
#             # Apply permutation to edge probabilities
#             sym_edgen1_pi_matrix = perm_matrix @ pi_matrix[0, :, :] @ perm_matrix.T
#             sym_edge0_pi_matrix = perm_matrix @ pi_matrix[1, :, :] @ perm_matrix.T
#             sym_edgep1_pi_matrix = perm_matrix @ pi_matrix[2, :, :] @ perm_matrix.T
#
#             # Convert the automorphism to a list of indices if it's not already
#             aut_indices = [aut[i] for i in range(n)]
#
#             # Apply permutation to vertex probabilities using advanced indexing
#             sym_vertex_pi = vertex_pi[aut_indices]
#
#             # Collect symmetric edge probabilities
#             sym_edgen1_pi = np.zeros(self.board.e)
#             sym_edge0_pi = np.zeros(self.board.e)
#             sym_edgep1_pi = np.zeros(self.board.e)
#
#             for idx, (x, y) in enumerate(self.board.edges):
#                 sym_edgen1_pi[idx] = sym_edgen1_pi_matrix[x, y]
#                 sym_edge0_pi[idx] = sym_edge0_pi_matrix[x, y]
#                 sym_edgep1_pi[idx] = sym_edgep1_pi_matrix[x, y]
#
#             # Combine symmetric edge and vertex probabilities
#             sym_edge_pi = np.hstack((sym_edgen1_pi, sym_edge0_pi, sym_edgep1_pi)).reshape(-1)
#             sym_pi = np.hstack((sym_edge_pi, sym_vertex_pi))
#
#             syms.append((sym_board, sym_pi))
#
#         return syms
#
#     def stringRepresentation(self, board):
#         return board.tobytes()
#
#     @staticmethod
#     def display(board):
#         print("Board: ")
#         print(board)
