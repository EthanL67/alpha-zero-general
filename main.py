import logging
import os
import warnings

import coloredlogs
from numba import NumbaPerformanceWarning

from Coach import Coach
from tangled.TangledGame import TangledGame as Game
from tangled.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
logging.getLogger('numba').setLevel(logging.ERROR)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

args = dotdict({
    'numIters': 1000,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 3,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/', '_best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'game_variant': "Q5"    # The type of graph to play on. Options: K3, K4, C4, Petersen, Q3, Q4, Q5
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args.game_variant)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
