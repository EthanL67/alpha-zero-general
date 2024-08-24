import Arena
from MCTS import MCTS
from tangled.TangledGame import TangledGame
from tangled.TanlgedPlayers import *
from tangled.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *
import os
import logging
import warnings
from numba import NumbaPerformanceWarning

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = False
gv = "K3"

g = TangledGame(gv)

# all players
rp = RandomPlayer(g).play
hp = HumanTangledPlayer(g).play

os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
logging.getLogger('numba').setLevel(logging.ERROR)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp', gv + '_best.pth.tar')

args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.5})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

player1 = n1p

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./temp/', gv + '_best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.5})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = rp  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(player1, player2, g, display=TangledGame.display)

print(arena.playGames(50, verbose=True))
