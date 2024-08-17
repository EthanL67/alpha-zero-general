from __future__ import print_function
import sys
import itertools
import numpy as np

from Game import Game
from .TangledLogic import Board, calculateScore
from .TangledVariants import *

sys.path.append('..')


class TangledGame(Game):
    def __init__(self, name):
        self.name = name
        self.getInitBoard()

    def getInitBoard(self):
        # return initial board (numpy board)
        if self.name == "K4":
            v, edges, adj_matrix = create_k4_graph()
        elif self.name == "C4":
            v, edges, adj_matrix = create_c4_graph()
        elif self.name == "Petersen":
            v, edges, adj_matrix = create_petersen_graph()
        elif self.name == "Q3":
            v, edges, adj_matrix = create_q3_graph()
        elif self.name == "Q4":
            v, edges, adj_matrix = create_q4_graph()
        elif self.name == "Q5":
            v, edges, adj_matrix = create_q5_graph()
        else:
            v, edges, adj_matrix = create_k3_graph()

        self.board = Board(v, edges, adj_matrix)

        return self.board.pieces

    def getBoardSize(self):
        # (a,b) tuple
        return (2*self.board.v, self.board.v)

    def getActionSize(self):
        # return number of actions: three colors per edge, or select one of the vertices
        return 3 * self.board.e + self.board.v

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.board.v, self.board.edges, self.board.adj_matrix)
        b.pieces = np.copy(board)
        # b = self.board
        # b.pieces = np.copy(board)

        b.execute_move(action, player)

        return b.pieces, -player

    def getValidMoves(self, board, player):
        b = Board(self.board.v, self.board.edges, self.board.adj_matrix)
        b.pieces = np.copy(board)
        # b = self.board
        # b.pieces = np.copy(board)

        return b.get_legal_moves(player)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.board.v, self.board.edges, self.board.adj_matrix)
        b.pieces = np.copy(board)
        # b = self.board
        # b.pieces = np.copy(board)

        if b.has_legal_moves():
            return 0
        else:
            score = calculateScore(b.pieces, b.v)
            if score > 0:
                return 1  # player 1 won
            elif score < 0:
                return -1  # player 1 lost
            else:
                return 1e-4  # draw

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        b = Board(self.board.v, self.board.edges, self.board.adj_matrix)
        b.pieces = np.copy(board)
        # b = self.board
        # b.pieces = np.copy(board)

        if player == -1:
            for v in range(self.board.v):
                b.pieces[v, v] = -b.pieces[v, v]

        return b.pieces

    def getSymmetries(self, board, pi):
        syms = [(board, pi)]

        # edge_idx_map = {edge: idx for idx, edge in enumerate(list(self.board.edges))}
        #
        # pieces = board[:self.board.v, :]
        # spaces = board[self.board.v:, :]
        #
        # edge_pi = pi[:3*self.board.e]
        # vertex_pi = pi[3*self.board.e:]
        #
        # for _ in range(self.board.v):
        #     # roll matrix
        #     sym_pieces = np.roll(pieces, axis=0, shift=1)
        #     sym_pieces = np.roll(sym_pieces, axis=1, shift=1)
        #     sym_spaces = np.roll(spaces, axis=0, shift=1)
        #     sym_spaces = np.roll(sym_spaces, axis=1, shift=1)
        #
        #     # roll pi
        #     sym_edge_pi = np.roll(edge_pi, axis=0, shift=3)
        #     sym_vertex_pi = np.roll(vertex_pi, axis=0, shift=1)
        #
        #     # reassemble the symmetric board
        #     sym_board = np.vstack((sym_pieces, sym_spaces))
        #
        #     # reassemble pi
        #     sym_pi = np.hstack((sym_edge_pi, sym_vertex_pi))
        #
        #     syms += [(sym_board, sym_pi)]

        # edge_pi = pi[:3*self.board.e]
        # vertex_pi = pi[3*self.board.e:]
        #
        # # Generate all permutations of the rows of the identity matrix
        # permutations = itertools.permutations(np.eye(self.board.v, dtype=int))
        #
        # for perm in permutations:
        #     perm_matrix = np.array(perm)
        #
        #     # transform adjacency matrix
        #     sym_adj_matrix = perm_matrix @ self.board.adj_matrix @ np.transpose(perm_matrix)
        #
        #     # transform edge pieces and spaces
        #     sym_pieces = perm_matrix @ self.board.pieces[:self.board.v, :] @ np.transpose(perm_matrix)
        #     sym_spaces = perm_matrix @ self.board.pieces[self.board.v:, :] @ np.transpose(perm_matrix)
        #
        #     # reassemble the symmetric board
        #     sym_board = np.vstack((sym_pieces, sym_spaces))
        #
        #     # find pi edge mapping
        #     edge_map_rows = np.any(self.board.adj_matrix != 0, axis=1)
        #     edge_map_cols = np.any(self.board.adj_matrix != 0, axis=0)
        #     edge_map = sym_adj_matrix[edge_map_rows][:, edge_map_cols]
        #
        #     # transform pi
        #     sym_edgen1_pi = edge_map @ edge_pi[0::3]
        #     sym_edge0_pi = edge_map @ edge_pi[1::3]
        #     sym_edgep1_pi = edge_map @ edge_pi[2::3]
        #     sym_vertex_pi = perm_matrix @ vertex_pi

        return syms

    def stringRepresentation(self, board):
        return board.tobytes()

    @staticmethod
    def display(board):
        print("Board: ")
        print(board)
