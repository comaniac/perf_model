import functools
import torch as th
import torch.nn as nn
import numpy as np
import logging
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from .losses import approxNDCGLoss, listMLE, lambdaLoss


def get_activation(act_type):
    if act_type == 'leaky':
        return nn.LeakyReLU()
    elif act_type == 'elu':
        return nn.ELU()
    elif act_type == 'relu':
        return nn.ReLU()
    else:
        raise NotImplementedError


def get_ranking_loss(loss_type):
    if loss_type == 'approx_ndcg':
        return approxNDCGLoss
    elif loss_type == 'list_mle':
        return listMLE
    elif loss_type == 'lambda_rank':
        return lambdaLoss
    elif loss_type == 'lambda_rank_hinge':
        return functools.partial(lambdaLoss, use_hinge_loss=True)
    else:
        raise NotImplementedError


class LinearBlock(nn.Module):
    def __init__(self, in_units, units, act_type, dropout, use_gate_net=False):
        super(LinearBlock, self).__init__()
        self.use_gate_net = use_gate_net
        self.linear1 = nn.Linear(in_features=in_units,
                                 out_features=units,
                                 bias=False)
        if use_gate_net:
            logging.info('Use Gate')
            self.gate_net = nn.Sequential(
                nn.Linear(in_features=in_units,
                          out_features=units,
                          bias=False),
                nn.Sigmoid())
        self.bn = nn.BatchNorm1d(units)
        self.act = get_activation(act_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn(out)
        out = self.act(out)
        if self.use_gate_net:
            gate = self.gate_net(x)
            out = out * gate
        out = self.dropout(out)
        return out


class RankingModel(nn.Module):
    def __init__(self, in_units, units=128, num_layers=3,
                 dropout=0.05, use_gate=True,
                 use_residual=False, feature_importance=False,
                 act_type='leaky'):
        super(RankingModel, self).__init__()
        blocks = []
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.feature_importance = feature_importance
        logging.info('Use Gate={}'.format(use_gate))
        if self.feature_importance:
            self.feature_importance_net = \
                nn.Sequential(
                    LinearBlock(in_units=in_units,
                                units=units,
                                act_type=act_type,
                                dropout=dropout,
                                use_gate_net=use_gate),
                    LinearBlock(in_units=units,
                                units=units,
                                act_type=act_type,
                                dropout=dropout,
                                use_gate_net=use_gate),
                    nn.Linear(in_features=units,
                              out_features=in_units),
                    nn.Sigmoid()
                )
        for i in range(num_layers):
            blocks.append(LinearBlock(in_units=in_units,
                                      units=units,
                                      act_type=act_type,
                                      dropout=dropout,
                                      use_gate_net=use_gate))
            in_units = units
        self.out_layer = nn.Sequential(
            nn.Linear(in_features=units,
                      out_features=1,
                      bias=True))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, X):
        """MLP with residual connection

        Parameters
        ----------
        X
            Shape (batch_size, in_units)

        Returns
        -------
        scores
            Shape (batch_size,)
        """
        if self.feature_importance:
            feature_importance = self.feature_importance_net(X)
            X = feature_importance * X
        data_in = self.blocks[0](X)
        for i in range(1, self.num_layers):
            out = self.blocks[i](data_in)
            if self.use_residual:
                out = out + data_in
        out = self.out_layer(out)
        return out[:, 0]


class RegressionSampler:
    def __init__(self, thrpt, regression_batch_size=2048,
                 neg_mult=5):
        self._regression_batch_size = regression_batch_size
        self._thrpt = thrpt
        self._neg_mult = neg_mult
        self._valid_indices = (thrpt > 0).nonzero()[0]
        self._invalid_indices = (thrpt == 0).nonzero()[0]
        self._generator = np.random.default_rng()

    def __iter__(self):
        while True:
            if self._neg_mult > 0:
                valid_batch_size = int(np.ceil(self._regression_batch_size / (1 + self._neg_mult)))
                valid_batch_size = min(valid_batch_size, len(self._valid_indices))
                invalid_batch_size = self._regression_batch_size - valid_batch_size
                valid_indices = self._generator.choice(len(self._valid_indices),
                                                       valid_batch_size,
                                                       replace=True)
                invalid_indices = self._generator.choice(len(self._invalid_indices),
                                                         invalid_batch_size,
                                                         replace=True)
                indices = np.hstack([valid_indices, invalid_indices])
            else:
                indices = self._generator.choice(len(self._thrpt), self._regression_batch_size,
                                                 replace=True)
            yield indices


class RankGroupSampler:
    def __init__(self, thrpt, regression_batch_size=1024,
                 rank_batch_size=512, group_size=10,
                 beta_params=(3.0, 1.0)):
        self._rank_batch_size = min(rank_batch_size, len(thrpt))
        self._regression_batch_size = min(regression_batch_size, len(thrpt))
        self._num_samples = len(thrpt)
        self._thrpt = thrpt
        self._group_size = group_size
        self._valid_indices = (thrpt > 0).nonzero()[0]
        self._invalid_indices = (thrpt == 0).nonzero()[0]
        # The mixture of dirichlet
        self._beta_params = beta_params
        self._generator = np.random.default_rng()

    @property
    def regression_batch_size(self):
        return self._regression_batch_size

    @property
    def rank_batch_size(self):
        return self._rank_batch_size

    def __iter__(self):
        """

        Returns
        -------
        indices
            List with shape (regression_batch_size + batch_size * group_size,)
        """
        while True:
            # regression_indices = np.random.choice(len(self._thrpt),
            #                                       self.regression_batch_size,
            #                                       replace=False)
            taus = np.random.beta(a=self._beta_params[0],
                                  b=self._beta_params[1],
                                  size=(self._rank_batch_size,))
            valid_nums = np.minimum(np.ceil(taus * self._group_size).astype(np.int32),
                                    len(self._valid_indices))
            invalid_nums = self._group_size - valid_nums
            if len(self._invalid_indices) == 0:
                valid_nums[:] = self._group_size
                invalid_nums[:] = 0
            rank_batch_indices = []
            for i in range(self._rank_batch_size):
                invalid_indices = self._generator.choice(self._valid_indices, valid_nums[i],
                                                         replace=False)
                valid_indices = self._generator.choice(self._invalid_indices, invalid_nums[i],
                                                       replace=False)
                rank_batch_indices.append(np.hstack([invalid_indices, valid_indices]))
            rank_batch_indices = np.vstack(rank_batch_indices)
            # batch_indices = np.hstack([regression_indices, rank_batch_indices.reshape((-1,))])
            # yield batch_indices
            yield rank_batch_indices.reshape((-1,))
