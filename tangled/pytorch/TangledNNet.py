import sys

sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class TangledNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(TangledNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(args.num_channels, args.num_channels) for _ in range(4)])

        self.fc1 = nn.Linear(args.num_channels * self.board_x * self.board_y, 2048)
        self.fc_bn1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 1024)
        self.fc_bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, self.action_size)

        self.fc4 = nn.Linear(1024, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y

        # Pass through residual blocks
        for block in self.residual_blocks:
            s = block(s)  # batch_size x num_channels x board_x x board_y

        s = s.view(-1, self.args.num_channels * self.board_x * self.board_y)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 2048
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 1024

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

# import sys
# sys.path.append('..')
# from utils import *
#
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# class OthelloNNet(nn.Module):
#     def __init__(self, game, args):
#         # game params
#         self.board_x, self.board_y = game.getBoardSize()
#         self.action_size = game.getActionSize()
#         self.args = args
#
#         super(OthelloNNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
#         self.conv5 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
#         self.conv6 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
#
#         self.bn1 = nn.BatchNorm2d(args.num_channels)
#         self.bn2 = nn.BatchNorm2d(args.num_channels)
#         self.bn3 = nn.BatchNorm2d(args.num_channels)
#         self.bn4 = nn.BatchNorm2d(args.num_channels)
#         self.bn5 = nn.BatchNorm2d(args.num_channels)
#         self.bn6 = nn.BatchNorm2d(args.num_channels)
#
#         self.fc1 = nn.Linear(args.num_channels*self.board_x*self.board_y, 2048)
#         self.fc_bn1 = nn.BatchNorm1d(2048)
#
#         self.fc2 = nn.Linear(2048, 1024)
#         self.fc_bn2 = nn.BatchNorm1d(1024)
#
#         self.fc3 = nn.Linear(1024, self.action_size)
#
#         self.fc4 = nn.Linear(1024, 1)
#
#     def forward(self, s):
#         #                                                           s: batch_size x board_x x board_y
#         s = s.view(-1, 1, self.board_x, self.board_y)               # batch_size x 1 x board_x x board_y
#         s = F.relu(self.bn1(self.conv1(s)))                         # batch_size x num_channels x board_x x board_y
#         s = F.relu(self.bn2(self.conv2(s)))                         # batch_size x num_channels x board_x x board_y
#         s = F.relu(self.bn3(self.conv3(s)))                         # batch_size x num_channels x board_x x board_y
#         s = F.relu(self.bn4(self.conv4(s)))                         # batch_size x num_channels x board_x x board_y
#         s = F.relu(self.bn5(self.conv5(s)))                         # batch_size x num_channels x board_x x board_y
#         s = F.relu(self.bn6(self.conv6(s)))                         # batch_size x num_channels x board_x x board_y
#         s = s.view(-1, self.args.num_channels*self.board_x*self.board_y)
#
#         s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
#         s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512
#
#         pi = self.fc3(s)                                                                         # batch_size x action_size
#         v = self.fc4(s)                                                                          # batch_size x 1
#
#         return F.log_softmax(pi, dim=1), torch.tanh(v)

# import sys
# sys.path.append('..')
# from utils import *
#
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, batch_norm=True):
#         super(ResidualBlock, self).__init__()
#         self.batch_norm = batch_norm
#         self.fc1 = nn.Linear(in_channels, out_channels)
#         self.fc2 = nn.Linear(out_channels, out_channels)
#
#         if self.batch_norm:
#             self.bn1 = nn.BatchNorm1d(out_channels)
#             self.bn2 = nn.BatchNorm1d(out_channels)
#
#         self.skip_connection = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
#
#     def forward(self, x):
#         identity = x
#         if self.skip_connection is not None:
#             identity = self.skip_connection(x)
#
#         out = F.relu(self.fc1(x))
#         if self.batch_norm:
#             out = self.bn1(out)
#         out = F.relu(self.fc2(out))
#         if self.batch_norm:
#             out = self.bn2(out)
#         out += identity
#         return out
#
# class TangledNNet(nn.Module):
#     def __init__(self, game, args):
#         super(TangledNNet, self).__init__()
#         self.board_x, self.board_y = game.getBoardSize()
#         self.action_size = game.getActionSize()
#         self.args = args
#
#         self.input_dim = self.board_x * self.board_y
#
#         self.res_block1 = ResidualBlock(self.input_dim, 2048)
#         self.res_block2 = ResidualBlock(2048, 2048)
#         self.res_block3 = ResidualBlock(2048, 2048)
#         self.res_block4 = ResidualBlock(2048, 2048)
#
#         self.fc5 = nn.Linear(2048, 1024)
#         self.bn5 = nn.BatchNorm1d(1024)
#
#         self.fc6 = nn.Linear(1024, 512)
#         self.bn6 = nn.BatchNorm1d(512)
#
#         self.fc7 = nn.Linear(512, self.action_size)
#         self.fc8 = nn.Linear(512, 1)
#
#     def forward(self, s):
#         s = s.view(-1, self.input_dim)
#
#         s = F.dropout(F.relu(self.res_block1(s)), p=self.args.dropout, training=self.training)
#         s = F.dropout(F.relu(self.res_block2(s)), p=self.args.dropout, training=self.training)
#         s = F.dropout(F.relu(self.res_block3(s)), p=self.args.dropout, training=self.training)
#         s = F.dropout(F.relu(self.res_block4(s)), p=self.args.dropout, training=self.training)
#
#         s = F.dropout(F.relu(self.bn5(self.fc5(s))), p=self.args.dropout, training=self.training)
#         s = F.dropout(F.relu(self.bn6(self.fc6(s))), p=self.args.dropout, training=self.training)
#
#         pi = self.fc7(s)
#         v = self.fc8(s)
#
#         return F.log_softmax(pi, dim=1), torch.tanh(v)
