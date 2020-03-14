"""Use BERT as the performance model."""
# pylint: disable=invalid-name
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import d2l
import tqdm
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from mxnet.gluon.utils import split_and_load

import logger

npx.set_np()

log = logger.get_logger('BERT')


### Class Declarations


class BERTEncoder(nn.Block):
    """BERT Encoder class."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, num_layers,
                 dropout, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(
                d2l.EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads,
                                 dropout))

    def forward(self, features):  # pylint: disable=arguments-differ
        X = features
        for blk in self.blks:
            X = blk(X, None)
        return X


class ThrptPred(nn.Block):
    """The network to predict throughput."""
    def __init__(self, hiddens, **kwargs):
        super(ThrptPred, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for hidden in hiddens:
            self.net.add(nn.Dense(hidden, activation='relu'))
        self.net.add(nn.Dense(1, activation='relu'))

    def forward(self, X):  # pylint: disable=arguments-differ
        # 0 is the index of the CLS token
        X = X[:, 0, :]
        # X shape: (batch size, num_hiddens)
        return self.net(X)


class ValidPred(nn.Block):
    """The network to predict if the config is valid or not."""
    def __init__(self, hiddens, **kwargs):
        super(ValidPred, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for hidden in hiddens:
            self.net.add(nn.Dense(hidden, activation='relu'))
        self.net.add(nn.Dense(1, activation='sigmoid'))

    def forward(self, X):  # pylint: disable=arguments-differ
        # X shape: (batch size, sequence length, num_hiddens)
        return self.net(X)


class BERTModel(nn.Block):
    """Define a BERT model."""
    def __init__(self, center_hiddens, center_ffn_hiddens, center_heads,
                 center_layers, reg_hiddens, cls_hiddens, dropout):
        super(BERTModel, self).__init__()
        self.center = BERTEncoder(center_hiddens, center_ffn_hiddens, center_heads,
                                  center_layers, dropout)
        # TODO(comaniac): Simplify this layer first.
        # self.reg_encoder = BERTEncoder(num_hiddens, ffn_num_hiddens, num_heads,
        #                                num_layers, dropout)
        self.reg_predictor = ThrptPred(reg_hiddens)
        self.valid_predictor = ValidPred(cls_hiddens)

    def forward(self, features):  # pylint: disable=arguments-differ
        encoded_X = self.center(features)
        valid_preds = self.valid_predictor(encoded_X)
        thrpt_preds = self.reg_predictor(encoded_X)
        return (valid_preds, thrpt_preds)

## Functions


def expand_hidden(feature, num_hiddens):
    """Expand features with number of hidden layers."""
    #pylint: disable=redefined-outer-name
    import numpy as np
    expand_feat = np.expand_dims(np.array(feature), axis=0)
    return np.broadcast_to(expand_feat.T, shape=(len(feature), num_hiddens))


def load_data(file_path, batch_size, num_hiddens):
    """Load tabular feature data."""
    log.info('Parsing file...')
    with open(file_path, 'r') as filep:
        next(filep)  # Get rid of headers

        # Parse features to sequence. num_seq = num_feature + 1
        features = []
        thrpts = []
        for line in tqdm.tqdm(filep):
            tokens = line.replace('\n', '').split(',')

            # initial <CLS> to 0
            features.append([0] + [float(v) for v in tokens[:-1]])
            thrpts.append(float(tokens[-1]))

    # Expand featues to (batch, sequence, hidden)
    log.info('Expanding features...')
    with ProcessPoolExecutor(max_workers=8) as pool:
        expand_features = []
        for start in tqdm.tqdm(
                range(0, len(features), 8),
                bar_format='{desc}{percentage:3.0f}%|{bar:50}{r_bar}'):
            futures = [
                pool.submit(expand_hidden, feature=feature, num_hiddens=num_hiddens)
                for feature in features[start:min(start + 8, len(features))]
            ]
            for future in as_completed(futures):
                expand_features.append(future.result())
        features = expand_features

    log.info('Total data size %d', len(features))

    # 70% for training.
    # 10% for validation.
    # 20% for testing.
    splitter1 = int(len(features) * 0.7)
    splitter2 = int(len(features) * 0.8)
    train_feats = np.array(features[:splitter1])
    train_thrpts = np.array(thrpts[:splitter1])
    train_iter = gluon.data.DataLoader(
        gluon.data.ArrayDataset(train_feats, train_thrpts),
        batch_size,
        shuffle=True,
        num_workers=d2l.get_dataloader_workers())
    validate_feats = np.array(features[splitter1:splitter2])
    validate_thrpts = np.array(thrpts[splitter1:splitter2])
    test_feats = np.array(features[splitter2:])
    test_thrpts = np.array(thrpts[splitter2:])
    return train_iter, validate_feats, validate_thrpts, test_feats, test_thrpts


def get_batch_loss(net, valid_loss, thrpt_loss, segments_X_shards, nsp_y_shards):
    """Get loss of a batch."""
    alpha = 0.7 # FIXME: A more reasonable number.

    # ls = []
    # valid_preds, thrpt_preds = net(segments_X_shards)
    # for valid_pred, thrpt_pred, nsp_y_shard in zip(valid_preds, thrpt_preds, nsp_y_shards):
    #     label = np.array([1 if nsp_y_shard > 0 else 0], ctx=valid_pred.ctx)
    #     vl = valid_loss(valid_pred, label).mean()
    #     tl = thrpt_loss(np.log(thrpt_pred + 1e-6), np.log(nsp_y_shard + 1e-6)).mean()
    #     if nsp_y_shard == 0:
    #         ls.append(vl)
    #     else:
    #         ls.append(vl + alpha * tl)
    # return np.array(ls, ctx=nsp_y_shards.ctx).mean()

    ls = []
    for segments_X_shard, nsp_y_shard in zip(segments_X_shards, nsp_y_shards):
        valid_pred, thrpt_pred = net(np.expand_dims(segments_X_shard, axis=0))
        label = np.array([1 if nsp_y_shard > 0 else 0], ctx=valid_pred.ctx)
        vl = valid_loss(valid_pred, label).mean()
        tl = thrpt_loss(np.log(thrpt_pred + 1e-6), np.log(nsp_y_shard + 1e-6)).mean()
        if nsp_y_shard == 0:
            ls.append(vl)
        else:
            ls.append(vl + alpha * tl)
    return ls



def test_acc(net, test_feats, test_thrpts, print_log=True):
    """Test the model accuracy."""
    valid_error = 0
    thrpt_errors = []
    for start in range(0, len(test_feats), 32):
        valid_preds, thrpt_preds = net(test_feats[start:min(start + 32, len(test_feats))])

        for idx, (valid_pred, thrpt_pred) in enumerate(zip(valid_preds, thrpt_preds)):
            valid_pred = valid_pred.tolist()[0]
            thrpt_pred = thrpt_pred.tolist()[0]
            real = test_thrpts[start + idx].tolist()
            valid_error += (valid_pred > 0.5 and real == 0) or (valid_pred <= 0.5 and real > 0)
            if valid_pred > 0.5:
                if real > 0:
                    error = abs(thrpt_pred - real) / real
                else:
                    error = thrpt_pred
                thrpt_errors.append(error)
                if print_log:
                    log.info('Pred %.6f, Expected %.6f, Error %.2f%%',
                             thrpt_pred, real, 100 * error)
            elif print_log:
                log.info('Pred invalid, expected %.6f', real)
    valid_err_rate = valid_error / len(test_feats)
    thrpt_err = (np.array(thrpt_errors).mean().tolist(), np.array(thrpt_errors).std().tolist())
    if print_log:
        log.info('Valid error %.2f%%', valid_err_rate)
        log.info('Average error: %.2f (std %.2f)', thrpt_err[0], thrpt_err[1])
    return (valid_err_rate, thrpt_err)


def train_bert(args, reporter=None):
    """Training process."""
    logger.enable_log_file('train-{}'.format(
        os.path.basename(args.data_file).replace('.csv', '.log')))

    ctx = d2l.try_all_gpus()

    log.info('Loading data from %s...', args.data_file)
    train_iter, validate_feats, validate_thrpts, test_feats, test_thrpts = load_data(
        args.data_file, args.batch_size, args.center_hiddens)
    validate_feats = np.array(validate_feats, ctx=ctx[0])
    validate_thrpts = np.array(validate_thrpts, ctx=ctx[0])
    test_feats = np.array(test_feats, ctx=ctx[0])
    test_thrpts = np.array(test_thrpts, ctx=ctx[0])


    fnet = [
        'Center Heads: {}'.format(args.center_heads),
        'Center FFN Hiddens: {}'.format(args.center_ffn_hiddens),
        'Center Layers: {}'.format(args.center_layers),
        'Valid Hiddens: {}'.format(args.cls_hiddens),
        'Regression Hiddens: {}'.format(args.reg_hiddens),
    ]
    log.info('Network\n\t%s', '\n\t'.join(fnet))
    fparams = [
        'Batch Size: {}'.format(args.batch_size),
        'Dropout: {}'.format(args.dropout),
        'Epochs: {}'.format(args.epochs),
        'Learning Rate: {}'.format(args.lr),
        'Weight Decay: {}'.format(args.wd)
    ]
    log.info('Hyper-Parameters:\n\t%s', '\n\t'.join(fparams))

    net = BERTModel(args.center_hiddens, args.center_ffn_hiddens,
                    args.center_heads, args.center_layers, args.reg_hiddens,
                    args.cls_hiddens, args.dropout)
    net.initialize(init.Xavier(), ctx=ctx)
    log.info('Model initialized on %s', str(ctx))

    # Initialize trainer.
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': args.lr,
        'wd': args.wd
    })

    # Imbalance dataset processing.
    num_valid = len(test_thrpts.nonzero()[0])
    num_invalid = len(test_thrpts) - num_valid
    pos_weight = num_invalid / num_valid
    log.info('Positive weight for SidmodBCELoss: %.2f', pos_weight)

    # Initialize loss functions.
    valid_loss = gluon.loss.SigmoidBCELoss(from_sigmoid=True, weight=pos_weight)
    thrpt_loss = gluon.loss.L2Loss()

    log.info('Training...')
    metric = d2l.Accumulator(2)
    for epoch in range(args.epochs):
        if reporter is None:
            log.info('Epoch %d', epoch)
            progress = tqdm.tqdm(
                train_iter, bar_format='{desc}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            progress = train_iter
        for iter_idx, (batch_feat, batch_thrpt) in enumerate(progress):
            np_feat = split_and_load(batch_feat, ctx, even_split=False)[0]
            np_thrpt = split_and_load(batch_thrpt, ctx, even_split=False)[0]
            with autograd.record():
                ls = get_batch_loss(net, valid_loss, thrpt_loss, np_feat, np_thrpt)
            for l in ls:
                l.backward()
            trainer.step(1)
            l_mean = sum([float(l) for l in ls]) / len(ls)
            metric.add(l_mean, 1)
            if reporter is None: # and iter_idx % 30 == 0:
                progress.set_description_str(desc='Loss {:.3f}'.format(l_mean),
                                             refresh=True)
        val_acc = test_acc(net, validate_feats, validate_thrpts, print_log=False)
        if reporter is None:
            log.info('Epoch %d: Loss %.3f, Valid Error %.2f%%, Thrpt Error %.3f (std %.3f)',
                     epoch, metric[0] / metric[1], val_acc[0], val_acc[1][0], val_acc[1][1])
        else:
            # FIXME: Not working now
            reporter(epoch=epoch, accuracy=val_acc)

    if reporter is None:
        log.info('Final loss %.3f', metric[0] / metric[1])

    log.info('Testing...')
    test_acc(net, test_feats, test_thrpts)
    return net


if __name__ == "__main__":
    main_args = argparse.Namespace()

    main_args.center_heads = 8
    main_args.center_hiddens = main_args.center_heads  # Force 1-to-1 attention
    main_args.center_layers = 6
    main_args.center_ffn_hiddens = 1024
    main_args.reg_hiddens = [1024]
    main_args.cls_hiddens = [512, 32]

    main_args.data_file = sys.argv[1]
    main_args.batch_size = 256
    main_args.dropout = 0.2
    main_args.epochs = 40
    main_args.lr = 1e-4
    main_args.wd = 1
    file_name = sys.argv[2]
    net = train_bert(main_args)
    net.save_parameters(file_name)
