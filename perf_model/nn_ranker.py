import functools
import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from .losses import approxNDCGLoss, listMLE, lambdaLoss


def get_activation(act_type):
    if act_type == 'leaky':
        return nn.LeakyReLU(0.1)
    elif act_type == 'elu':
        return nn.ELU()
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


class RankingModel(nn.Module):
    def __init__(self, in_units, units=128, num_layers=3,
                 dropout=0.05, use_bn=True, act_type='leaky'):
        super(RankingModel, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_features=in_units,
                                    out_features=units,
                                    bias=False))
            in_units = units
            if use_bn:
                layers.append(nn.BatchNorm1d(in_units))
            layers.append(get_activation(act_type))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features=in_units,
                                out_features=1,
                                bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        """

        Parameters
        ----------
        X
            Shape (batch_size, units)

        Returns
        -------
        scores
            Shape (batch_size,)
        """
        return self.net(X)[:, 0]


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
