import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.encoding = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
#         self.encoding[:, 0::2] = torch.sin(position * div_term)
#         self.encoding[:, 1::2] = torch.cos(position * div_term)
#         self.encoding = self.encoding.unsqueeze(0)
#
#     def forward(self, x):
#         device = x.device  # Get the device of the input tensor
#         return x + self.encoding[:, :x.size(1)].to(device)


# class TangledNNet(nn.Module):
#     def __init__(self, game, args):
#         super(TangledNNet, self).__init__()
#         self.board_x, self.board_y = game.getBoardSize()
#         self.action_size = game.getActionSize()
#         self.args = args
#
#         self.input_dim = self.board_x * self.board_y
#         self.embedding_dim = input_dim
#         self.num_heads = 8
#         self.num_layers = 4
#         self.hidden_dim = 1024
#
#         # Define the embedding layer
#         self.embedding = nn.Linear(self.input_dim, self.embedding_dim)
#         self.positional_encoding = PositionalEncoding(self.embedding_dim)
#
#         # Define Transformer Encoder Layers
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embedding_dim,
#             nhead=self.num_heads,
#             dim_feedforward=self.hidden_dim
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=self.num_layers
#         )
#
#         # Define the output layers
#         self.fc1 = nn.Linear(self.embedding_dim, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#
#
#         self.fc2 = nn.Linear(512, self.action_size)
#         self.fc3 = nn.Linear(512, 1)
#
#     def forward(self, s):
#         s = s.view(-1, self.input_dim)
#         s = F.relu(self.embedding(s))
#         s = self.positional_encoding(s)
#
#         s = s.transpose(0, 1)  # Transformer expects input shape [seq_len, batch_size, features]
#         s = self.transformer_encoder(s)
#         s = s.transpose(0, 1)  # Restore shape to [batch_size, seq_len, features]
#
#         s = s.mean(dim=1)  # Aggregate across sequence length
#
#         s = F.dropout(F.relu(self.bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
#
#         pi = self.fc2(s)
#         v = self.fc3(s)
#
#         return F.log_softmax(pi, dim=1), torch.tanh(v)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.batch_norm = batch_norm
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)

        self.skip_connection = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        if self.skip_connection is not None:
            identity = self.skip_connection(x)

        out = F.relu(self.fc1(x))
        if self.batch_norm:
            out = self.bn1(out)
        out = F.relu(self.fc2(out))
        if self.batch_norm:
            out = self.bn2(out)
        out += identity
        return out

class TangledNNet(nn.Module):
    def __init__(self, game, args):
        super(TangledNNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.input_dim = self.board_x * self.board_y

        self.res_block1 = ResidualBlock(self.input_dim, 2048)
        self.res_block2 = ResidualBlock(2048, 2048)
        self.res_block3 = ResidualBlock(2048, 2048)
        self.res_block4 = ResidualBlock(2048, 2048)

        self.fc5 = nn.Linear(2048, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc6 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)

        self.fc7 = nn.Linear(512, self.action_size)
        self.fc8 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, self.input_dim)

        s = F.dropout(F.relu(self.res_block1(s)), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.res_block2(s)), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.res_block3(s)), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.res_block4(s)), p=self.args.dropout, training=self.training)

        s = F.dropout(F.relu(self.bn5(self.fc5(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.bn6(self.fc6(s))), p=self.args.dropout, training=self.training)

        pi = self.fc7(s)
        v = self.fc8(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
